#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    # add scaling based on the notes
    pool3_out_scaled = tf.multiply(layer3_out, 0.0001, name = 'pool3_out_scaled')
    pool4_out_scaled = tf.multiply(layer4_out, 0.01, name = 'pool4_out_scaled')
    # return image_input, keep_prob, layer3_out, layer4_out, layer7_out
    return image_input, keep_prob, pool3_out_scaled, pool4_out_scaled, layer7_out

# tests.test_load_vgg(load_vgg, tf) will pass it use the following return:
#     `return image_input, keep_prob, layer3_out, layer4_out, layer7_out`

print("load_vgg test passed.")

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    ## 1by1 conv and conv tranpose
    conv7_1by1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides = (1,1), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    output_layer7 = tf.layers.conv2d_transpose(conv7_1by1, num_classes, 4, strides = (2,2), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # one skip connection
    ## 1by1 conv
    conv4_1by1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides = (1,1), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    input_skip4 = tf.add(output_layer7, conv4_1by1)
    ## conv trans
    output_skip4 = tf.layers.conv2d_transpose(input_skip4, num_classes, 4, strides=(2, 2), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # a second skip connection
    conv3_1by1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides = (1,1), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    output_skip3 = tf.add(output_skip4, conv3_1by1)
    # Input = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8))
    output = tf.layers.conv2d_transpose(output_skip3, num_classes, 16, strides=(8, 8), padding = 'same',
                                     kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    return output 
tests.test_layers(layers)
print("layer function test passed.")

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    ## reshape the nn_last_layer to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    
    # add L2 regularization into cost manually or explicitly...
    vars   = tf.trainable_variables() 
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001
    
    cross_entropy_loss_l2 = cross_entropy_loss + lossL2
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)
print("optimize function test passed.")

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    cost_result_epoch = []
    import numpy as np
    import matplotlib.pyplot as plt


    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-3})
        print("at current {} epoch, loss = {}".format(epoch+1, loss))
        cost_result_epoch.append(loss)
    
    # plot loss versus epoch, is it decreasing?
    N = len(cost_result_epoch)
    x = np.linspace(0, N, N)

    plt.scatter(x, cost_result_epoch)
    plt.plot(x, cost_result_epoch, '-o')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss_regularized')
    plt.show()
    pass

tests.test_train_nn(train_nn)
print("train_nn function test passed.")

def run():
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    saver_dir = './saver/model_test.'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    tf.reset_default_graph()
    with tf.Session() as sess:
        
        #sess.run(tf.global_variables_initializer())
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        print("first step is to load vgg")
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        # placeholders for optimize
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        
        epochs = 64
        batch_size = 8
        
        ## tf saver initialized
        saver = tf.train.Saver()
        
        ## sess.run(tr.global_variables_initializer())
        print("get ready for training in train_nn:")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        
        save_path = saver.save(sess, saver_dir)
        
        # load saved model for additional processing like video frames
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./saver/'))
        
        print("Trained model with {} epochs and batch_size = {} Saved to ".format(epochs, batch_size) + save_path)
        

if __name__ == '__main__':
    run()
