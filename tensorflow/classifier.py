import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

class Classifier:

    def __init__(self, batchSize, numEpoch):
        self.batchSize = batchSize
        self.numEpoch = numEpoch
        
        self.trainedModelPath = "./trained_model/model.ckpt"
        
        # single flattened 28 by 28 pixel MNIST image 
        self.x  = tf.placeholder(tf.float32, shape=[None, 784])
        # label output (take value from 0 to 9)
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # start a session for
        self.session = tf.InteractiveSession()
        
    def train(self):
        self.loadMNISTDataset()
    
    def predict(self, filename):
        # load trained model
        self.loadModel()
        
        # read input image into (1, 784) array
        x_image = self.readImageIntoArray(filename)
        
        # forward pass, and find the index of maximum value in y
        prediction = tf.argmax(self.y, 1)
        label = prediction.eval(feed_dict = {self.x: x_image})
        return label[0]
        
    def evaluate(self, images, labels):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy.eval(feed_dict = {self.x : images,
                                          self.y_: labels})
        
    def loadMNISTDataset(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    
    def saveModel(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, self.trainedModelPath)
        print("Model saved in file: %s" % save_path)
        
    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(self.session, self.trainedModelPath)
        
    def readImageIntoArray(self, filename):
        # read & convert image to Grayscale
        image = Image.open(filename, 'r').convert('L')
        
        # convert to numpy array
        array = np.asarray(image.getdata(), dtype=np.float32)
        return np.reshape(array, (-1, 784))
        
class SoftmaxClassifier(Classifier):
    
    def __init__(self, batchSize = 100, numEpoch = 1000):
        Classifier.__init__(self, batchSize, numEpoch)
        
        # parameters: W and b
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        
        # regression model
        self.y = tf.matmul(self.x, self.W) + self.b
        
        self.session.run(tf.global_variables_initializer())
        
    def train(self):
        Classifier.train(self)
        
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
        )
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        for i in range(self.numEpoch):
            batch = self.mnist.train.next_batch(self.batchSize)
            train_step.run(feed_dict={self.x : batch[0],
                                      self.y_: batch[1]})
            
        # evaluate trained model on test data
        print("Accuracy on test data: ", self.evaluate(self.mnist.test.images,
                                                       self.mnist.test.labels))
        
        # save trained model
        self.saveModel()
        
class CNNClassifier(Classifier):
    
    def __init__(self, batchSize = 100, numEpoch = 30000):
        Classifier.__init__(self, batchSize, numEpoch)
        
        # define CNN layers & their connections
        self.buildCNNLayers()
    
    def buildCNNLayers(self):
        # 1st Layer
        self.W_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)
        
        # 2nd layer
        self.W_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)
        
        # Densely Connected Layer
        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        
        # Readout Layer
        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        
        self.session.run(tf.global_variables_initializer())
        
    def train(self):
        Classifier.train(self)
        
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
        )
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self.session.run(tf.global_variables_initializer())
        for i in range(self.numEpoch):
            batch = self.mnist.train.next_batch(self.batchSize)
            
            # evaluate model after every 200 step
            if i % 200 == 0:
                print("step %d, training accuracy %g" % \
                      (i, self.evaluate(batch[0], batch[1])))
                train_step.run(feed_dict={self.x : batch[0],
                                          self.y_: batch[1]})

        # evaluate trained model on test data
        print("Accuracy on test data: ", self.evaluate(self.mnist.test.images,
                                                       self.mnist.test.labels))
        
        # save trained model
        self.saveModel()
    
    # CNN Helpers
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

# Usage:
#classifier = SoftmaxClassifier()
#label = classifier.predict("./test/9.jpg")
#print('Predicted label:', label)

classifier = CNNClassifier()
#classifier.train()
label = classifier.predict("./test/4.png")
print('Predicted label:', label)
