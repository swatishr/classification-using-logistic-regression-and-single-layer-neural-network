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
	# for i in range(0,N):#repeat 55000 times
	# 	scores = zeros(10)
	# 	for cls in range(10):
	# 		w = W[cls, :]
	# 		scores[cls] = w.dot(trains_images[i,:])
	# 	scores -= max(scores)
	# 	# correct_class = y[i]
	# 	sum_exp_scores = sum(exp(scores))

	# 	# corr_cls_exp_score = np.exp(scores[correct_class])
	# 	h[i,:] = exp(scores) / sum_exp_scores
	# 	# print(h[i,:].shape)
	# 	# print(h[i,:])
	# 	# break
	# return h


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
def softmaxx(inputs):
	"""
	Calculate the softmax for the give inputs (array)
	:param inputs:
	:return:
	"""
	return exp(inputs) / float(sum(exp(inputs)))
#cross entropy error function
def cross_entropy(h, T):
	[N, D] = h.shape
	# Y_dash = design_matrix.dot(W)
	# print(Y_dash[1:10])
	# print("min Y max Y = %0.4f  %0.4f"%(np.min(Y_dash), np.max(Y_dash)))
	Error =0;
	tempError = 0;
	for i in range(0,N):
		Error = Error + -1*T[i,:].dot(log(h[i,:].T))
		# for j in range(0,10):
		# 	tempError +=-1*T[i,j]*log2(h[i,j])
		# Error =Error+tempError
		# Error =  Error + -log(h[i,T[i,:].tolist().index(1)])
		# print(Error)
		# print(h[i,T[i,:]])
		# tempError = 0
		# break
		# break
	# Erms = np.sqrt((2 * Error)/N)
	return Error/N

#sgd back again, ahaa
def sgd_solution(W, learning_rate, train_images, T, h):
	[N, D]=train_images.shape
	E=0
	loss = 0
	# weights = np.zeros([10,D])
	weights = W
	# delta = zeros([10,D])
	delta = np.zeros_like(W)
	tempWeight = 0
	# h=h.T
	# b = zeros((N,))
	# print(T.shape)
	# for epoch in range(num_epochs):
	# 	print('epoch is ',epoch+1)
	# print("weights ", weights)
	# for i in range(0,N):
	for j in range(0,10):
		# print (j)
		for i in range(0,N):
			tempWeight += -(h[i,j]-T[i,j])*train_images[i,:]
			# print((h[i,j]-T[i,j]))
			# print(tempWeight)
			# break

		# break
		delta[j,:] = tempWeight/N
		tempWeight = 0
		# weights[j,:] = weights[j,:] - learning_rate*delta[j,:]
	weights = weights - learning_rate*delta
	# print(weights)

	# for i in range(int(N/minibatch_size)):
	# 	lower_bound = i * minibatch_size
	# 	upper_bound = min((i+1)*minibatch_size, N)
	# 	Phi = trains_images[lower_bound : upper_bound, :]
	# 	t = T[lower_bound : upper_bound, :]
	# 	E_D = np.matmul((np.matmul(Phi, weights.T)-t).T, Phi)
	# 	E = (E_D + L2_lambda * weights) / minibatch_size
	# 	weights = weights - learning_rate * E
	# print (weights.shape)
	# print(np.linalg.norm(E))
	# print (weights)
	return weights

def loss_grad_softmax_naive(W, train_images, T, reg):
	"""
	Compute the loss and gradients using softmax function 
	with loop, which is slow.

	Parameters
	----------
	W: (K, D) array of weights, K is the number of classes and D is the dimension of one sample.
	X: (D, N) array of training data, each column is a training sample with D-dimension.
	y: (N, ) 1-dimension array of target data with length N with lables 0,1, ... K-1, for K classes
	reg: (float) regularization strength for optimization.

	Returns
	-------
	a tuple of two items (loss, grad)
	loss: (float)
	grad: (K, D) with respect to W
	"""
	X = train_images.T
	loss = 0
	grad = np.zeros_like(W)
	dim, num_train = X.shape
	num_classes = W.shape[0]
	for i in range(num_train):
		sample_x = X[:, i]
		scores = np.zeros(num_classes) # [K, 1] unnormalized score
		for cls in range(num_classes):
		    w = W[cls, :]
		    scores[cls] = w.dot(sample_x)
		# Shift the scores so that the highest value is 0
		scores -= np.max(scores)
		correct_class = T[i]
		sum_exp_scores = np.sum(np.exp(scores))

		corr_cls_exp_score = np.exp(scores[correct_class])
		loss_x = -np.log(corr_cls_exp_score / sum_exp_scores)
		loss += loss_x

		# compute the gradient
		percent_exp_score = np.exp(scores) / sum_exp_scores
		for j in range(num_classes):
		    grad[j, :] += percent_exp_score[j] * sample_x


		grad[correct_class, :] -= sample_x # deal with the correct class

	loss /= num_train
	loss += 0.5 * reg * np.sum(W * W) # add regularization
	grad /= num_train
	grad += reg * W
	return loss, grad

def loss_grad_softmax_vectorized(W, X, y, reg):
	""" Compute the loss and gradients using softmax with vectorized version"""
	X =X.T
	loss = 0 
	grad = np.zeros_like(W)
	dim, num_train = X.shape

	scores = W.dot(X) # [K, N]
	# Shift scores so that the highest value is 0
	scores -= np.max(scores)
	scores_exp = np.exp(scores)
	# print(scores_exp.shape)
	# correct_scores_exp = scores_exp[y, range(num_train)] # [N, ]
	scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
	# loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
	# loss /= num_train
	# loss += 0.5 * reg * np.sum(W * W)

	scores_exp_normalized = scores_exp / scores_exp_sum
	# deal with the correct class
	scores_exp_normalized[y, range(num_train)] -= 1 # [K, N]
	grad = scores_exp_normalized.dot(X.T)
	grad /= num_train
	grad += reg * W

	return 0, grad
#prediction of the labels
def predict(W, X):
        """
        Predict value of y using trained weights

        Parameters
        ----------
        X: (D x N) array of data, each column is a sample with D-dimension.

        Returns
        -------
        pred_ys: (N, ) 1-dimension array of y for N sampels
        """
        X = X.T
        Y_dash = np.zeros(X.shape[1])
        scores = W.dot(X)
        Y_dash = np.argmax(scores, axis=0)
        return Y_dash