# Jupyter style runtime for report test cases

# %% Prep
import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi
import random
import pandas as pd
from NNBackprop import NeuralNetwork

def function(x1, x2):
    '''Function that the neural net is trying to learn'''
    return (sin(2*pi*x1) * sin(2*pi*x2))

def genData(dataset, n=5):
    '''Function that generates a dataset  based on the n and store it in the list that is passed'''
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


def genRandData(dataset, n=10):
    '''Generate random data and store it'''

    for _ in range(n):
        x1 = random.random()
        x2 = random.random()
        Y = function(x1, x2)
        dataset.append((float(x1), float(x2), float(Y)))

    dataset = np.round(dataset)

# %% Part A : i/5 j/5 generate data
''' Part A'''

dataset = []
genData(dataset)

dataset = scramble(dataset)

# %% Train for 1000 epoch
lastIteration = []    #last iteration of training input and output

NN = NeuralNetwork(3)  # create NN with 3 hidden units

NN.train(dataset, lastIteration, 1000)  # train the neural network

li = pd.DataFrame(lastIteration, columns = ['x1', 'x2', 'A', 't', 'error'])

print(li)

li.to_excel("/data/1000epoch/train.xlsx")
# print(dataset)

# %% Train for 10000 epoch
lastIteration = []    #last iteration of training input and output

NN = NeuralNetwork(3)  # create NN with 3 hidden units

NN.train(dataset, lastIteration, 10000)  # train the neural network

li = pd.DataFrame(lastIteration, columns = ['x1', 'x2', 'A', 't', 'error'])

print(li)

li.to_excel("/data/10000epoch/train.xlsx")

# %% test on i/10 j/10 generate data

testset = []
genData(testset, 10)

# %% test for 1000 epoch
testlist = []

NN.test(testset,testlist)  # test the neural network

tl = pd.DataFrame(testlist, columns = ['x1', 'x2', 'A', 't', 'error'])

# %% test for 10000 epoch

testlist = []

NN.test(testset,testlist)  # test the neural network

tl = pd.DataFrame(testlist, columns = ['x1', 'x2', 'A', 't', 'error'])

# %% Part C random x1, and x2
'''Part C'''

dataset = []
genRandData(dataset, 5)

# %% Train for 1000 epoch
lastIteration = []    #last iteration of training input and output

NN = NeuralNetwork(3)  # create NN with 3 hidden units

NN.train(dataset, lastIteration, 1000)  # train the neural network

li = pd.DataFrame(lastIteration, columns = ['x1', 'x2', 'A', 't', 'error'])

print(li)


# %% Train for 10000 epoch
lastIteration = []    #last iteration of training input and output

NN = NeuralNetwork(3)  # create NN with 3 hidden units

NN.train(dataset, lastIteration, 10000)  # train the neural network

li = pd.DataFrame(lastIteration, columns = ['x1', 'x2', 'A', 't', 'error'])

print(li)

# %% test on i/10 j/10 

testset = []
genData(testset, 10)

# %% test for 1000 epoch

testlist = []

NN.test(testset,testlist)  # test the neural network

tl = pd.DataFrame(testlist, columns = ['x1', 'x2', 'A', 't', 'error'])

# %% test for 10000 epoch

testlist = []

NN.test(testset,testlist)  # test the neural network

tl = pd.DataFrame(testlist, columns = ['x1', 'x2', 'A', 't', 'error'])

