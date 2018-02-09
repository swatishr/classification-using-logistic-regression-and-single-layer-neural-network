# Classification using Logistic Regression, Single Hidden Layer Neural Network and Convolutional Neural Network

Authors: Swati Nair, Jayant Solanki

We have trained our classification model on MNIST data using logistic regression, single
hidden layer neural network, convolutional neural network and predicted the labels of
the digit images in both MNIST and USPS digit dataset.

-Logistic Regression:
We trained the model and tuned the hyperparameter i.e. learning rate, by using our own implementation of Logistic regression, we achieved an accuracy of 91.56% on MNIST test images and 45.15% on USPS test images at learning rate of 0.14 and lambda (regulariser) value of 0. Using tensorflow, we have achieved an accuracy of 92.41% on MNIST test images and 48.32% on USPS test images

- Single hidden layer neural network:
We trained the model and tuned the hyperparameter i.e. learning rate and
number of units in hidden layer, we achieved an accuracy of 97.76% on
MNIST test images and 64.6% on USPS test images.

- Convolutional Neural Network:
After training the model using CNN, we achieved an accuracy of 99.18 % on
MNIST test images and 75% on USPS test images.

Project Structure:
1. logistic_main.py : Run this file for execution of logistic regression model without using tensorflow

2. logistic_tensorflow_main.py : Run this file for execution of logistic regression model using tensorflow. It creates the model, trains it, tests it on MNIST validation, test set and USPS test set

3. single_layer_NN_main.py : Run this file for execution of single hidden layer NN model using tensorflow. It creates the model, trains it, tests it on MNIST validation, test set and USPS test set

4. cnn_main2.py : Run this file for execution of CNN model using tensorflow. It creates the model, trains it, tests it on MNIST validation, test set and USPS test set

5. libs.py :
a. read_gz(images,labels): To read MNIST gz data
b. view_image(image, label=""): to view single image from the MNSIT data
c. yDash(trains_images, W): for performing the W.dot(X)
d. softmax(x) : for calculating the softmax of each row in W.dot(X)
e. sgd(W, train_images, T, L2_lambda, epochNo, learning_rate): gradient descend for optimising the weights
f. cross_entropy(W, X, T, L2_lambda): cacluating the loss in the model
g. predict(W, X): predicting the labels from the output of the model

6. single_layer_NN_lib.py :
a. create_single_hidden_layer_nn(number_hidden_units): create input layer, one hidden layer with specified number of neurons and output layer
7. cnn_lib.py :
a. weight_init(shape): Initialize weight variables
b. bias_init(shape): Initialize bias variables
c. convolution(x, W): convolves input with given weights and stride 1 and with zero padding
d. maxpool(x): performs max pooling on window size of 2x2 and stride of 2 with zero padding

8. USPS_data_extraction.py :
a. make_square(im): To make the image with equal height and width
b. extract_usps_data(): Get USPS test images and labels. Usps_test_images is a Nx784 numpy array and Usps_test_labels is Nx10 numpy array (one-hot representation)

### References
1. Ublearns
2. Stackoverflow.com
3. Python, Numpy and TensorFlow documentations
4. https://cs231n.github.io/convolutional-networks/
5. http://yann.lecun.com/exdb/mnist/
6. https://martin-thoma.com/classify-mnist-with-pybrain/
