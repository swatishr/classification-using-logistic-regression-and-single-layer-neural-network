#########################################################################################
#Description:
# Read the traning data and perform the calculations
#
#########################################################################################
import numpy as np
from data_loader import *

try:
	data = np.load("trainData.npz")
	trains_images = data['trains_images']
	train_images_label = data['train_images_label']
except FileNotFoundError:
	images = gzip.open('../MNIST-data/train-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('../MNIST-data/train-labels-idx1-ubyte.gz', 'rb')
	(trains_images,train_images_label) =read_gz(images, labels);
	np.savez("trainData.npz", trains_images=trains_images, train_images_label=train_images_label)
try:
	data = np.load("testData.npz")
	test_images = data['test_images']
	test_images_label = data['test_images_label']
except FileNotFoundError:
	images = gzip.open('../MNIST-data/t10k-images-idx3-ubyte.gz', 'rb')
	labels = gzip.open('../MNIST-data/t10k-labels-idx1-ubyte.gz', 'rb')
	(test_images,test_images_label) =read_gz(images,labels);
	np.savez("testData.npz", test_images=test_images, test_images_label=test_images_label)
print(trains_images.shape)
print(train_images_label.shape)
print(test_images.shape)
print(test_images_label.shape)