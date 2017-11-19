#########################################################################################
#Description:
# Read the traning data and perform the calculations
#
#########################################################################################
import numpy as np
from libs import *

try:
	data = np.load("trainData.npz")
	trains_images = data['trains_images']
	train_images_label = data['train_images_label']
except FileNotFoundError:
	images = gzip.open('MNIST_Data/train-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('MNIST_Data/train-labels-idx1-ubyte.gz', 'rb')
	(trains_images,train_images_label) =read_gz(images, labels);
	np.savez("trainData.npz", trains_images=trains_images, train_images_label=train_images_label)
try:
	data = np.load("testData.npz")
	test_images = data['test_images']
	test_images_label = data['test_images_label']
except FileNotFoundError:
	images = gzip.open('MNIST_Data/t10k-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('MNIST_Data/t10k-labels-idx1-ubyte.gz', 'rb')
	(test_images,test_images_label) =read_gz(images,labels);
	np.savez("testData.npz", test_images=test_images, test_images_label=test_images_label)
# print(trains_images.shape)
# print(train_images_label.shape)
# print(test_images.shape)
# print(test_images_label.shape)
# view_image(trains_images[9999,:,:], train_images_label[9999])
trains_images = trains_images.reshape([60000,784])
test_images = test_images.reshape([10000,784])
# print(trains_images.shape)
# print(train_images_label.shape)
# print(test_images.shape)
# print(test_images_label.shape)

################################Preparing the weights and feature matrices##################################
W = np.ones((10, 785), dtype=float32)  # Initialize numpy array #784+1
# W = np.random.randn(10, 785) * 0.001
# trains_images = np.insert(trains_images, 1, values=1, axis=1)#adding the extra column in feature matrix
trains_images = np.insert(trains_images, 0, 1, axis=1)#adding the extra column in feature matrix, 785 features now
validation_images = trains_images[55000:60000]
validation_labels = train_images_label[55000:60000,:]
trains_images = trains_images[0:55000]
train_images_label = train_images_label[0:55000,:]
# print(trains_images.shape)
# print(train_images_label.shape)
# print(validation_images.shape)
# print(validation_labels.shape)
train_images_label_target_mat = np.zeros((55000, 10), dtype=uint8)
train_images_label_target_mat[np.arange(55000), train_images_label.T] = 1#hot vector
# print(train_images_label_target_mat.shape)#one hot vector
# z = np.matmul(trains_images,theta)
# print(loss_grad_softmax_naive(W, trains_images, train_images_label_target_mat, 0))
try:
	data = np.load("weights.npz")
	W = data['W']
except FileNotFoundError:
	for epoch in range(11):
		loss, grad = loss_grad_softmax_naive(W, trains_images, train_images_label, 0.5)
		W -= 0.05 * grad # [K x D]
		if(epoch % 10 == 0):
			print ('iteration %d/%d: loss %0.3f' % (epoch, 1000, loss[0]))
	np.savez("weights.npz", W=W)
yDash = predict(W, trains_images)
print(W)
# for i in range(55000):
# 	print("predicted label : %d Actual Label %d" %(yDash[i], train_images_label[i]))
# h = yDash(trains_images, W)
# # for i in range(0,55000):#repeat 50000 times
# # 	# print(trains_images[i,:].shape)
# # 	a = np.matmul(W,trains_images[i,:])
# # 	# print(a)
# # 	# print(trains_images[i,:])
# # 	h[i,:] =  softmax(a.T)
# # 	# print(softmax(a.T))
# # 	# break
# print('done')
# print(h[1,:])
# print(cross_entropy(h, train_images_label_target_mat))
# print('performing SGD')
# W = sgd_solution(W, 0.5, 55000, 2, 0.1, trains_images, train_images_label_target_mat, h)
# print(W.shape)
# h = yDash(trains_images, W)
# print(cross_entropy(h, train_images_label_target_mat))
# print(h[1:10,:])
# print(softmaxx([2, 3, 5, 6]))
# cross_entropy(h,train_images_label_target_mat)