#########################################################################
#CSE574 | Project 3 | Part 2
#Description: To classify images by building single hidden layer NN
#using tensorflow
#########################################################################

import tensorflow as tf
from single_layer_NN_lib import *
from tensorflow.examples.tutorials.mnist import input_data

#Initialize Parameters
learning_rate = 0.01
training_epochs = 20000
batch_size = 50

#Construct NN model
predicted_y, x, actual_y = create_single_hidden_layer_nn()

#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= predicted_y, labels = actual_y))

#Optimizer for training
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#gives a boolean vector for whether the actual and predicted output match (1 if true, 0 if false)
right_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(actual_y, 1))

#get accuracy
accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))

#Download, extract and read MNIST data in numpy array
mnistData = input_data.read_data_sets('MNIST_Data', one_hot=True)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#training
	for i in range(training_epochs):
		batch = mnistData.train.next_batch(batch_size)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], actual_y: batch[1]})
			print("At step %d, training accuracy: %.2f" %(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], actual_y: batch[1]})

	#Run on MNIST test data	
	test_accuracy = accuracy.eval(feed_dict={x: mnistData.test.images, actual_y: mnistData.test.labels})
	print("MNIST test accuracy: %.2f" %(test_accuracy))