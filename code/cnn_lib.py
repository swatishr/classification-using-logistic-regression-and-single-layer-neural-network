#########################################################################
#CSE574 | Project 3 | Part 3
#Description: Functions to be used for CNN implementation using tensorflow
#########################################################################

import tensorflow as tf

#Weight Variable initialization
def weight_init(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#Bias Variable initialization
def bias_init(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#Function to convolve input with given weights, stride=1 and zero padding
def convolution(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#Function for 2D max pooling
def maxpool(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
