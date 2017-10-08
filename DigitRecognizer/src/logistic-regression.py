import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2  # to test only
import os
import pandas as pd
# import tflearn  # can simplify the process, for later

def testShowPicture():
    with open('../data/mnist_test.csv', 'r') as csv_file:
        for data in csv.reader(csv_file):
            # The first column is the label
            label = data[0]

            # The rest of columns are pixels
            pixels = data[1:]

            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='uint8')

            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((28, 28))

            # Plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            break # This stops the loop, I just want to see one


def readData(filename):
    df = pd.read_csv(filename, header=None)

    # For images
    images = df.iloc[:,1:].values
    images = images.astype(np.float)

    # For labels
    labels = df[[0]].values.ravel()
    labels_count = np.unique(labels).shape[0]

    return images, labels

def getHotVector(x, nlabel):
    size = x.shape[0]
    tmp=np.zeros((size,nlabel))
    tmp[np.arange(size),x.astype('int64')] = 1
    return tmp

# ------------ 
#     Main     
# ------------ 

# testShowPicture()

# Define parameters
nlabel = 10
learning_rate=0.01
training_epochs=1000

# Load Data
file_train_set =  "../data/mnist_train.csv"
file_test_set =  "../data/mnist_test.csv"
x_train, y_train = readData(file_train_set)
x_test, y_test = readData(file_test_set)

x_train = np.multiply(x_train, 1.0 / 255.0)
x_test = np.multiply(x_test, 1.0 / 255.0)

# Transform Data 
y_train = getHotVector(y_train, nlabel)
y_test  = getHotVector(y_test, nlabel)

# Create TensorFlow variable
x = tf.placeholder(tf.float32, shape=[None, 784]) 
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
y_prediction = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_prediction), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

# debug
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(training_epochs):
		sess.run(optimizer, feed_dict={x: x_train, y_: y_train})
		if i%10 == 0:
			print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) )


# save results
# np.savetxt('submission.csv', 
# 	   np.c_[range(1,len(y_test)+1),y_prediction], 
# 	   delimiter=',', 
# 	   header = 'ImageId,Label', 
# 	   comments = '', 
# 	   fmt='%d')

sess.close()


# ------------ 
#   Not used   
# ------------ 
# dataset = tf.contrib.data.Dataset.csv('../data/mnist_test.csv')
# dir_path = os.path.dirname(os.path.realpath(__file__ + "../"))
# filename = dir_path + "/data/mnist_test.csv"

# labels = data[[0]].values.ravel().astype(np.uint8)
# X_test = images[:2000].reshape(-1, 1, 28, 28)
# x = tf.placeholder(tf.float32,shape=(150,5))
# x = data

# b=np.zeros((60000,10))
# b[np.arange(60000),y_train]=1
# y_train = b

# b=np.zeros((10000,10))
# b[np.arange(10000),y_test]=1
# y_test = b
