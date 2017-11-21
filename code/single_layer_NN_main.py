#########################################################################
#CSE574 | Project 3 | Part 2
#Description: To classify images by building single hidden layer NN
#using tensorflow
#########################################################################

import tensorflow as tf
from single_layer_NN_lib import *
from tensorflow.examples.tutorials.mnist import input_data
from USPS_data_extraction import *
print('Running Single Hidden Layer Neural Network...\n\n\n')
#Initialize Parameters
learning_rate = 0.01
training_epochs = 20000
batch_size = 50
number_hidden_units = 784

#Download, extract and read MNIST data in numpy array
mnistData = input_data.read_data_sets('MNIST_Data', one_hot=True)

#Extract USPS data
usps_test_images, usps_test_labels = extract_usps_data(0)

#below commented for loops iis for hypertuning
# for number_hidden_units in range(784,1030, 80):
# 	for learning_rate in np.arange(0.01,0.06,0.01):
#Construct NN model
predicted_y, x, actual_y = create_single_hidden_layer_nn(number_hidden_units)

#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= predicted_y, labels = actual_y))

#Optimizer for training
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#gives a boolean vector for whether the actual and predicted output match (1 if true, 0 if false)
right_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(actual_y, 1))

#get accuracy
accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#training
	for i in range(training_epochs):
		batch = mnistData.train.next_batch(batch_size)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], actual_y: batch[1]})
			print("At step %d, training accuracy: %.2f" %(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], actual_y: batch[1]})

	#Run on MNIST training data	
	accuracy_mnist_train = accuracy.eval(feed_dict={x: mnistData.train.images, actual_y: mnistData.train.labels})
	print("MNIST validation accuracy: %.2f" %(accuracy_mnist_train*100))

	#Run on MNIST validation data	
	accuracy_mnist_val = accuracy.eval(feed_dict={x: mnistData.validation.images, actual_y: mnistData.validation.labels})
	print("MNIST validation accuracy: %.2f" %(accuracy_mnist_val*100))

	#Run on MNIST test data	
	accuracy_mnist_test = accuracy.eval(feed_dict={x: mnistData.test.images, actual_y: mnistData.test.labels})
	print("MNIST test accuracy: %.2f" %(accuracy_mnist_test*100))

	#Run on USPS test data
	accuracy_usps = accuracy.eval(feed_dict={x: usps_test_images, actual_y: usps_test_labels})
	print("The accuracy on USPS test set: %.2f" %(accuracy_usps*100))

	#print("%d %.2f %.2f %.2f %.2f %.2f" %(number_hidden_units, learning_rate, accuracy_mnist_train*100, accuracy_mnist_val*100, accuracy_mnist_test*100, accuracy_usps*100))
print('\n\n')