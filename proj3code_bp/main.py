#Single__Hidden_layer_BP.py
#########################################################################################
#Description:
# Read the training data and perform the calculations
#
#########################################################################################
import numpy as np
from SNLibs import *
from USPS_data_extraction import *
from sklearn import preprocessing
print('Running Single__Hidden_layer with Back-Propogation Regression...\n')
try:#training data
	data = np.load("trainData.npz")
	trains_images = data['trains_images']
	train_images_label = data['train_images_label']
except FileNotFoundError:#read from mnist file if presaved data not found
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
try:#usps data
	data = np.load("USPStestData-logistic.npz")
	USPS_test_images = data['USPS_test_images']
	USPS_test_images_label = data['USPS_test_images_label']
except FileNotFoundError:
	(USPS_test_images,USPS_test_images_label) =extract_usps_data(1)
	np.savez("USPStestData-logistic.npz", USPS_test_images=USPS_test_images, USPS_test_images_label=USPS_test_images_label)
# print(trains_images.shape)
# print(train_images_label.shape)
# print(USPS_test_images.shape)
# print(USPS_test_images_label)

# view_image(trains_images[9999,:,:], train_images_label[9999])
trains_images = trains_images.reshape([60000,784])#flattening the input array
test_images = test_images.reshape([10000,784])
trains_images = preprocessing.scale(trains_images)#standardising the image data set with zero mean and unit standard deviation
test_images = preprocessing.scale(test_images)
USPS_test_images = preprocessing.scale(USPS_test_images)
# trains_images = (trains_images - trains_images.min())/(trains_images.max()-trains_images.min())
# test_images = (test_images - test_images.min())/(test_images.max()-test_images.min())
# print(trains_images.shape)
# print(train_images_label.shape)
# print(test_images.shape)
# print(test_images_label.shape)

################################Preparing the weights and feature matrices##################################
W = np.ones((10, 785), dtype=float32)  # Initialize numpy array #784+1 for the weight with one, also added the bias column too
# W = np.random.randn(10, 785) * 0.001
# trains_images = np.insert(trains_images, 1, values=1, axis=1)#adding the extra column in feature matrix
# trains_images = np.insert(trains_images, 0, 1, axis=1)#adding the extra column in feature matrix, 785 features now
validation_images = trains_images[55000:60000]
validation_labels = train_images_label[55000:60000,:]
trains_images = trains_images[0:55000]
train_images_label = train_images_label[0:55000,:]
# test_images = np.insert(test_images, 0, 1, axis=1)#adding the extra column in feature matrix, 785 features now
# USPS_test_images = np.insert(USPS_test_images, 0, 1, axis=1)#adding the extra column in feature matrix, 785 features now
train_images_label_target_mat = np.zeros((55000, 10), dtype=uint8)
train_images_label_target_mat[np.arange(55000), train_images_label.T] = 1#hot vector ofr the input label

# model = build_model(3, 200, True, trains_images[0:200,:], train_images_label[0:200,:], 0, 0.05) 
model = build_model(30, 200, trains_images, train_images_label, 0.05, 0.05, train_images_label_target_mat) 
yDashTrain = predict(model, trains_images)
count = 0;
for i in range(55000):
	# print("predicted label : %d Actual Label %d" %(yDashTrain[i], train_images_label[i]))
	if(yDashTrain[i] == train_images_label[i]):
		count = count + 1
print("training set Accuracy is %f" %(count/55000))
yDashVal = predict(model, validation_images)
count = 0
for i in range(5000):
	# print("predicted label : %d Actual Label %d" %(yDash[i], validation_labels[i]))
	if(yDashVal[i] == validation_labels[i]):
		count = count + 1
print("validation set Accuracy is %f"%(count/5000))
yDashTest = predict(model, test_images)
count = 0
for i in range(10000):
	# print("predicted label : %d Actual Label %d" %(yDash[i], test_images_label[i]))
	if(yDashTest[i] == test_images_label[i]):
		count = count + 1
print("Test set Accuracy is %f" %(count/10000))
yDashTest = predict(model, USPS_test_images)
count = 0
for i in range(19999):
	# print("predicted label : %d Actual Label %d" %(yDash[i], USPS_test_images_label[i]))
	# print(USPS_test_images_label[i])
	# print(np.where( USPS_test_images_label[i]==1))
	if(yDashTest[i] == np.where( USPS_test_images_label[i]==1)):
		count = count + 1

print("USPS set Accuracy is %f" %(count/19999))

