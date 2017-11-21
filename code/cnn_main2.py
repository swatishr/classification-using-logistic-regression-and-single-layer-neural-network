#########################################################################
#CSE574 | Project 3 | Part 3
#Description: To classify images by building convolutional neural network
#using tensorflow
#########################################################################

from cnn_lib import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from USPS_data_extraction import *
print('Running CNN...\n\n\n')
#Download, extract and read MNIST data in numpy array
mnistData = input_data.read_data_sets('MNIST_Data', one_hot=True)

#Create nodes for input images and target labels
x = tf.placeholder(tf.float32, shape = [None, 784]) #784 for 28x28 image
actual_y = tf.placeholder(tf.float32, shape = [None, 10]) #10 labels for 10 digits (0-9)

#-----------Convolution layer 1 with 32 features applied on 5x5 patch of image-----------

#Intialization of weights and bias
W1 = weight_init([5, 5, 1, 32])
bias1 = bias_init([32])

#reshape data into 4d
x_4d = tf.reshape(x, [-1, 28, 28, 1])

#convolve the image, add bias, apply relu activation function
conv1_output = tf.nn.relu(convolution(x_4d, W1) + bias1)

#apply max pooling
pool1_output = maxpool(conv1_output)


#-----------Convolution layer 2 with 64 features applied on 5x5 patch of image-----------

#Intialization of weights and bias
W2 = weight_init([5, 5, 32, 64])
bias2 = bias_init([64])

#convolve the layer 1 output, add bias, apply relu activation function
conv2_output = tf.nn.relu(convolution(pool1_output, W2) + bias2)

#apply max pooling
pool2_output = maxpool(conv2_output)


#------------Fully Connected Layer with 1024 neurons--------------

#Intialization of weights and bias
W_fc = weight_init([7*7*64, 1024])
bias_fc = bias_init([1024])

#flatten the pool2 output, multiply it with weights, add bias and then apply relu function
pool2_flat = tf.reshape(pool2_output, [-1, 7*7*64])

fc1_output = tf.nn.relu(tf.matmul(pool2_flat, W_fc) + bias_fc)

#dropout only during training to prevent overfitting, not during testing
no_drop_prob = tf.placeholder(tf.float32)  #probability of not dropping out the neurons output

fc1_output_drop = tf.nn.dropout(fc1_output, no_drop_prob)


#------------Logit Layer--------------
W_logit = weight_init([1024, 10])
bias_logit = bias_init([10])

logit_output = tf.matmul(fc1_output_drop, W_logit) + bias_logit


#-------Cross entropy loss function------
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = actual_y, logits = logit_output))

#training using optimizers: AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#gives a boolean vector for whether the actual and predicted output match (1 if true, 0 if false)
right_prediction = tf.equal(tf.argmax(logit_output, 1), tf.argmax(actual_y, 1))

#get accuracy
accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))


#train and evaluate the model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# saver = tf.train.Saver()
	for i in range(20000):
		batch = mnistData.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], actual_y: batch[1], no_drop_prob: 1.0})
			print("At step %d, training accuracy: %.2f" %(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], actual_y: batch[1], no_drop_prob: 0.5})

	#Run on MNIST test data
	accuracy_mnist = accuracy.eval(feed_dict={x: mnistData.test.images, actual_y: mnistData.test.labels, no_drop_prob: 1.0})
	print("MNIST test accuracy: %.2f" %(accuracy_mnist*100))

	#Run on USPS test data
	usps_test_images, usps_test_labels = extract_usps_data(0)
	accuracy_usps1 = accuracy.eval(feed_dict={x: usps_test_images[0:10000,:], actual_y: usps_test_labels[0:10000,:], no_drop_prob: 1.0})
	print("The accuracy on USPS test set1: %.2f" %(accuracy_usps1*100))
	accuracy_usps2 = accuracy.eval(feed_dict={x: usps_test_images[10000:19999,:], actual_y: usps_test_labels[10000:19999,:], no_drop_prob: 1.0})
	print("The accuracy on USPS test set2: %.2f" %(accuracy_usps2*100))

	print("The accuracy on USPS test set: %.2f" %(((accuracy_usps1+accuracy_usps2)/2)*100))