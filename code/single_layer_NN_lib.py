#########################################################################
#CSE574 | Project 3 | Part 2
#Description: Functions to be used for single hidden layer NN using tensorflow
#########################################################################

import tensorflow as tf

#Creates a single hidden layer Neural Network model
def create_single_hidden_layer_nn(number_hidden_units):

    n_hidden_1 = number_hidden_units #number of neurons in hidden layer
    n_input = 784
    n_classes = 10

    #Create nodes for input images and target labels
    x = tf.placeholder(tf.float32, shape = [None, n_input]) #784 for 28x28 image
    actual_y = tf.placeholder(tf.float32, shape = [None, n_classes]) #10 labels for 10 digits (0-9)

    # Hidden layer with RELU activation
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    bias1 = tf.Variable(tf.random_normal([n_hidden_1]))
    layer_1 = tf.matmul(x, W1) + bias1
    layer_1 = tf.nn.relu(layer_1)
    
    #Output layer with linear activation
    W_out = tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    bias_out = tf.Variable(tf.random_normal([n_classes]))
    out_layer = tf.matmul(layer_1, W_out) + bias_out

    return out_layer, x, actual_y