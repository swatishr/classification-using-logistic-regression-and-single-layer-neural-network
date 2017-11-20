#data-loader.py
#loads the training , validation and test data from the MNIST data files
from struct import unpack
import gzip
from pylab import imshow, show, cm, matmul
# from math import log

# Third-party libraries
import numpy as np
from numpy import zeros, uint8, float32, exp, max, log2, sum, log
def read_gz(images,labels):

	"""Read input-vector (image) and target class (label, 0-9) and return
	   it as list of tuples.
	"""
	# Open the images with gzip in read binary mode
	# images = gzip.open('../MNIST-data/train-images-idx3-ubyte.gz', 'rb')
	# labels = gzip.open('../MNIST-data/train-labels-idx1-ubyte.gz', 'rb')

	# Read the binary data

	# We have to get big endian unsigned int. So we need '>I'

	# Get metadata for images
	images.read(4)  # skip the magic_number
	number_of_images = images.read(4)
	number_of_images = unpack('>I', number_of_images)[0]
	rows = images.read(4)
	rows = unpack('>I', rows)[0]#28
	cols = images.read(4)
	cols = unpack('>I', cols)[0]#28

	# Get metadata for labels
	labels.read(4)  # skip the magic_number
	N = labels.read(4)
	N = unpack('>I', N)[0] #60000
	# print(number_of_images);

	if number_of_images != N:
	    raise Exception('number of labels did not match the number of images')

	# Get the data
	x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array #60000X28X28
	y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
	for i in range(N):
	    if i % 1000 == 0:
	        print("i: %i" % i)
	    for row in range(rows):
	        for col in range(cols):
	            tmp_pixel = images.read(1)  # Just a single byte
	            tmp_pixel = unpack('>B', tmp_pixel)[0]
	            x[i][row][col] = tmp_pixel
	    tmp_label = labels.read(1)
	    y[i] = unpack('>B', tmp_label)[0]
	    # print(y.shape)#60000X1
	return (x, y)

def view_image(image, label=""):
	"""View a single image."""
	print("Label: %s" % label)
	imshow(image, cmap=cm.gray)
	show()

#calculating the predicted values
def yDash(trains_images, W):
	[N, D] = trains_images.shape
	h = zeros((N, 10), dtype=float32);
	for i in range(0,N):#repeat 55000 times
		# print(trains_images[i,:].shape)
		# a = matmul(W,trains_images[i,:])
		h[i,:] = W.dot(trains_images[i,:])
		# print(h[i,:].shape)
		# print(h[i,:])
		# print(trains_images[i,:])
		h[i,:] =  softmax(h[i,:])
		# break
		# print(softmax(a.T))
		# break
	return h

#hello softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = exp(x - max(x))
    return e_x / e_x.sum()


def sgd(W, train_images, T, L2_lambda, epochNo, learning_rate):
	N, D = train_images.shape
	for epoch in range(epochNo):
		loss = cross_entropy(W,train_images, T, L2_lambda)
		grad = np.zeros_like(W)
		K = W.shape[0]#number of classes
		for i in range(N):
			ydash = np.zeros(K) # [K, 1] unnormalized score
			ydash = W.dot(train_images[i,:])
			y = T[i]
			normalised_yDash = softmax(ydash)
			for j in range(K):
			    grad[j, :] += normalised_yDash[j] * train_images[i,:]
			grad[np.where( y==1), :] -= train_images[i,:] # deal with the correct class
		grad /= N
		grad += L2_lambda * W
		if(epoch % 10 == 0):
			print ('iteration %d/%d: loss %0.3f' % (epoch, epochNo, loss))
		W -= learning_rate * grad # [K x D]
	return W

#cross entropy error function
def cross_entropy(W, X, T, L2_lambda):
	[N, D] = X.shape
	# ydash = W.dot(X)
	Error = 0
	E_D = 0
	for i in range(N):
		ydash = np.zeros(10) # [K, 1] unnormalized score
		ydash = W.dot(X[i,:])
		y = T[i]
		normalised_yDash = softmax(ydash)
		E_D = -1*y.dot(log(normalised_yDash.T))
		# E_D = -np.log(corr_cls_exp_score / sum_exp_scores)
		Error += E_D
	Error /= N
	Error += 0.5 * L2_lambda * np.sum(W * W) # add regularization
	# Y_dash = design_matrix.dot(W)
	# print(Y_dash[1:10])
	# print("min Y max Y = %0.4f  %0.4f"%(np.min(Y_dash), np.max(Y_dash)))
	return Error
#prediction of the labels
def predict(W, X):
        X = X.T
        Y_dash = np.zeros(X.shape[1])
        scores = W.dot(X)
        Y_dash = np.argmax(scores, axis=0)
        return Y_dash