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
#function for performing the W.dot(X) with softmax
#function yDash(trains_images, W)
#input : train image array and the weight matrix
#output : returns the W.dot(X), that is the h matrix
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
#function for pperforming the gradient descent
#function sgd(W, train_images, T, L2_lambda, epochNo, learning_rate)
#input : W s weight matrix, train_image is the train data, T is the one hot vector of the train labels, L2_lambda is the regulariser, epchoNo is the iteration count for performing gradient descent, learning_rate is the step size
#output : returns the optimsed weights
def sgd(W, train_images, T, L2_lambda, epochNo, learning_rate):
	N, D = train_images.shape
	for epoch in range(epochNo):
		loss = cross_entropy(W,train_images, T, L2_lambda)
		grad = np.zeros_like(W)
		K = W.shape[0]#number of classes
		for i in range(N):
			ydash = np.zeros(K) # [K, 1] 
			ydash = W.dot(train_images[i,:]) #unnormalized predicted label
			y = T[i]
			normalised_yDash = softmax(ydash)
			for j in range(K):#performing the gradient descent
			    grad[j, :] += normalised_yDash[j] * train_images[i,:]
			grad[np.where( y==1), :] -= train_images[i,:] # deal with the correct label in the one hot vector of y
		grad /= N
		grad += L2_lambda * W
		if(epoch % 10 == 0):
			print ('iteration %d/%d: loss %0.3f' % (epoch, epochNo, loss))
		W -= learning_rate * grad # [K x D]#updating the weights
	return W

#################################
#function for calculating the cross entropy error/loss
#function cross_entropy(W, X, T, L2_lambda)
#input : weights, input array for the image data, T is the one hot vector of the actual labels, L2_lambda is the regulariser
#output : returns error value between actual and predicted label
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

# Helper function to evaluate the total loss on the dataset 
def calculate_loss(model, X, y): 
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

# %% 16 
# This function learns parameters for the neural network and returns the model. 
# - nn_hdim: Number of nodes in the hidden layer 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 1000 iterations 
def build_model(nn_hdim, num_passes, print_loss, X, y, reg_lambda, learning_rate): 
 
    # Initialize the parameters to random values. We need to learn these. 
    [num_examples, nn_input_dim] = X.shape
    np.random.seed(0) 
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
    b1 = np.zeros((1, nn_hdim)) 
    W2 = np.random.randn(nn_hdim, 10) / np.sqrt(nn_hdim) 
    b2 = np.zeros((1, 10)) 
 
    # This is what we return at the end 
    model = {} 
 
    # Gradient descent. For each batch... 
    for i in range(0, num_passes): 
 
        # Forward propagation 
        z1 = X.dot(W1) + b1 
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
 
        # Backpropagation 
        delta3 = probs 
        # print(probs.shape)
        # print(y.shape)
        # print(range(10))
        # delta3[range(num_examples), y] -= 1 
        for i in range(num_examples):
        	delta3[i, y[i]] = delta3[i, y[i]] -1
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
        if print_loss and i % 10 == 0: 
          print("Loss after iteration %i: %f" %(i, calculate_loss(model))) 
 
    return model 