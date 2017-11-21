#general library file, contains all the methods needed for performing the LR
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

#################################
#function for viewing the image and its label
#function view_image(image, label="")
#input : image array and its label
#output : displays image and its label on the console
def view_image(image, label=""):
	"""View a single image."""
	print("Label: %s" % label)
	imshow(image, cmap=cm.gray)
	show()


#################################
#function for performing softmax
#function softmax(X)
#input : X is the result of W.dot(input_image)
#output : returns softmax output
def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = exp(x - max(x))
	return e_x / e_x.sum()


#################################
#function for calculating the cross entropy error/loss
#function cross_entropy(W, X, T, L2_lambda)
#input : weights, input array for the image data, T is the one hot vector of the actual labels, L2_lambda is the regulariser
#output : returns error value between actual and predicted label
def cross_entropy(model, X, T, L2_lambda):
	# [N, D] = X.shape
	# ydash = W.dot(X)
	[num_examples, nn_input_dim] = X.shape
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
	probs = np.zeros((num_examples ,10),dtype=float32) # [K, 1] unnormalized score
	[num_examples, nn_input_dim] = X.shape
	# Forward propagation to calculate our predictions 
	z1 = X.dot(W1) + b1 
	a1 = np.tanh(z1) 
	z2 = a1.dot(W2) + b2 
	Error = 0
	E_D = 0
	for i in range(num_examples):
		ydash = np.zeros(10) # [K, 1] unnormalized score
		ydash = z2[i].T
		y = T[i]
		normalised_yDash = softmax(ydash)
		E_D = -1*y.dot(log(normalised_yDash.T))
		# E_D = -np.log(corr_cls_exp_score / sum_exp_scores)
		Error += E_D
	Error /= num_examples
	Error += 0.5 * L2_lambda *(np.sum(np.square(W1)) + np.sum(np.square(W2)))  # add regularization
	# Y_dash = design_matrix.dot(W)
	# print(Y_dash[1:10])
	# print("min Y max Y = %0.4f  %0.4f"%(np.min(Y_dash), np.max(Y_dash)))
	return Error

# Helper function to evaluate the total loss on the dataset 
def calculate_loss(model, X,y, reg_lambda): 
	[num_examples, nn_input_dim] = X.shape
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
	# Forward propagation to calculate our predictions 
	z1 = X.dot(W1) + b1 
	a1 = np.tanh(z1) 
	z2 = a1.dot(W2) + b2 
	exp_scores = np.exp(z2) 
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
	# Calculating the loss 
	corect_logprobs = -np.log(probs[range(num_examples), y]) 
	data_loss = np.sum(corect_logprobs) 
	# Add regulatization term to loss (optional) 
	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
	return 1./num_examples * data_loss 	
#################################
#function for calculating the cross entropy error/loss
#function predict(model, x)
#input : weight matrix of the model, x is the input data set of the image
#output : return the column with the maximum probabiliy value (softmax)
def predict(model, x): 
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
	# Forward propagation 
	z1 = x.dot(W1) + b1 
	a1 = np.tanh(z1) 
	z2 = a1.dot(W2) + b2 
	exp_scores = np.exp(z2) 
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
	return np.argmax(probs, axis=1) 

# %% 16 
# This function learns parameters for the neural network and returns the model. 
# - nn_hdim: Number of nodes in the hidden layer 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 10 iterations 
def build_model(nn_hdim, num_passes, X, y, reg_lambda, learning_rate, T): 
 
	# Initialize the parameters to random values. We need to learn these. 
	[num_examples, nn_input_dim] = X.shape
	np.random.seed(0) 
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
	b1 = np.zeros((1, nn_hdim)) 
	W2 = np.random.randn(nn_hdim, 10) / np.sqrt(nn_hdim) 
	b2 = np.zeros((1, 10)) 
 
	# This is what we return at the end 
	model = {} 
	probs = np.zeros((num_examples ,10),dtype=float32) # [K, 1] unnormalized score
	# Gradient descent. For each batch... 
	for i in range(0, num_passes): 
 
		# Forward propagation 
		z1 = X.dot(W1) + b1 
		a1 = np.tanh(z1) 
		z2 = a1.dot(W2) + b2 
		# print(z2.shape)
		# exp_scores = np.exp(z2) 
		# probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
		for m in range(num_examples):
			probs[m,:] = softmax(z2[m,:])
 
		# performing Backpropagation 
		delta3 = probs
		delta3 =probs - T 
		# for i in range(num_examples):
		# 	delta3[i, y[i]] = delta3[i, y[i]] -1
		dW2 = (a1.T).dot(delta3) 
		db2 = np.sum(delta3, axis=0, keepdims=True) 
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) 
		dW1 = np.dot(X.T, delta2) 
		db1 = np.sum(delta2, axis=0) 
 
		# Add regularization terms (b1 and b2 don't have regularization terms) 
		dW2 += reg_lambda * W2 
		dW1 += reg_lambda * W1 
 
		# Gradient descent parameter update 
		W1 += -learning_rate * dW1 
		b1 += -learning_rate * db1 
		W2 += -learning_rate * dW2 
		b2 += -learning_rate * db2 
 
		# Assign new parameters to the model 
		model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
 
		# Optionally print the loss. 
		# This is expensive because it uses the whole dataset, so we don't want to do it too often. 
		loss = cross_entropy(model,X, T, reg_lambda)
		if i % 10 == 0: 
		  # print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y, reg_lambda))) 
		  print("Loss after iteration %i: %f" %(i, loss)) 
 
	return model 