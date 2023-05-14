#!/usr/bin/env python3
#
from keras.datasets import mnist
import numpy as np

from nn import NeuralNet

def target_ohe(k):
    x = np.zeros((10, 1))
    x[k] = 1
    return x


if __name__ == '__main__':
    (trainX, trainy), (testX, testy) = mnist.load_data()
    trainX, testX = np.reshape(trainX, (-1, 784, 1)), np.reshape(testX, (-1, 784, 1))
    trainy = map(target_ohe, trainy)
    # Inefficient, but break it into row by row
    train_data, test_data = list(zip(trainX, trainy)), list(zip(testX, testy))
    NeuralNet([784, 30, 10]).train(train_data, test_data, epochs=50, batchsize=32, eta=1)
