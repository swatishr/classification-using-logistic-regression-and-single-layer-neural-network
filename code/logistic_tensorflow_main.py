
#########################################################################
#CSE574 | Project 3 | Part 1 using Tensorflow
#Description: To classify images with single linear layer model using tensorflow
#########################################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Download, extract and read MNIST data in numpy array
mnistData = input_data.read_data_sets('MNIST_Data', one_hot=True)

#Start a session
sess = tf.InteractiveSession()

#Create nodes for imput images and target labels

x = tf.placeholder(tf.float32, shape = [None, 784]) #784 for 28x28 image
actual_y = tf.placeholder(tf.float32, shape = [None, 10]) #10 labels for 10 digits (0-9)

#Define parameters of model: weight and bias
W = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#To predict class of given image
predicted_y = tf.nn.softmax(tf.matmul(x, W) + bias)

#loss function
loss = tf.reduce_mean(-tf.reduce_sum(actual_y * tf.log(predicted_y), reduction_indices=[1]))

#train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#run training on train data of batches of 100
for _ in range(10000):
	batch = mnistData.train.next_batch(100)
	#print(mnistData.train._index_in_epoch)
	train_step.run(feed_dict={x: batch[0], actual_y: batch[1]})

#Evaluate the model
match_predictions = tf.equal(tf.argmax(actual_y, 1), tf.argmax(predicted_y, 1))

accuracy = tf.reduce_mean(tf.cast(match_predictions, tf.float32))

ans = accuracy.eval(feed_dict={x: mnistData.test.images, actual_y: mnistData.test.labels})

print("The accuracy on MNIST test set: %.2f" %(ans*100))