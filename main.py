import matplotlib.pyplot as plt
import csv
import pathlib
from math import pi, sin
import random
import numpy as np
from NNBackprop import NeuralNetwork
import datamanip as dm

def function(x1, x2):
    '''
    Input: 2 decimal numbers from 0 to 1 inclusive
    Ouput: 1 decimal number
    time: O(1), space: O(1)
    '''
    return (sin(2*pi*x1) * sin(2*pi*x2))

def genData(dataset, n=5):
    '''
    Input: filename to store data in, function to generate the data,
            n for generating n + 1 points with
    Output: None
    time: O((n + 1)^2), space: O((n + 1)^2)
    '''
    for i in range(n + 1):
        for j in range(n + 1):
            x1 = i/n
            x2 = j/n
            Y = function(x1, x2)
            dataset.append((float(x1), float(x2), float(Y)))

def scramble(old):
    '''return a scrambled list based on old list'''
    new = old[:]
    random.shuffle(new)
    return new


def genRandData(dataset, n=50):
    '''
    Input: filename to store data in, function to generate the data,
            n for generating n + 1 points with
    Output: None
    time: O((n + 1)^2), space: O((n + 1)^2)
    '''

    for _ in range(n):
        x1 = random.random()
        x2 = random.random()
        Y = function(x1, x2)
        dataset.append((float(x1), float(x2), float(Y)))

    dataset = np.round(dataset)


if __name__ == "__main__":

    randomset = []
    genRandData(randomset)

    dataset = []
    genData(dataset)

    dataset = scramble(dataset)

    print(dataset)

    NN = NeuralNetwork()  # create NN with 2 hidden neurons

    NN.train(dataset, 1000)  # train the neural network

    NN.test(randomset)