# Jupyter style runtime for report test cases

# %% Prep
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import numpy


def function(x1, x2):
    '''
    Input: 2 decimal numbers from 0 to 1 inclusive
    Ouput: 1 decimal number
    time: O(1), space: O(1)
    '''
    return (math.sin(2*math.pi*x1) * math.sin(2*math.pi*x2))


def genData(filename, n=5):
    '''
    Input: filename to store data in, function to generate the data,
            n for generating n + 1 points with
    Output: None
    time: O((n + 1)^2), space: O((n + 1)^2)
    '''

    with open(filename, 'w+') as F:
        for i in range(n + 1):
            for j in range(n + 1):
                x1 = i/n
                x2 = j/n
                Y = function(x1, x2)
                F.write(f'{x1:.2f},{x2:.2f},{Y:.2f}\n')


def genRandData(filename, n=50):
    '''
    Input: filename to store data in, function to generate the data,
            n for generating n + 1 points with
    Output: None
    time: O((n + 1)^2), space: O((n + 1)^2)
    '''

    with open(filename, 'w+') as F:
        for _ in range(n):
            x1 = random.random()
            x2 = random.random()
            Y = function(x1, x2)
            F.write(f'{x1:.2f},{x2:.2f},{Y:.2f}\n')


def parseData(filename, dataset):
    '''read data from file and insert it into a cached dataset'''
    with open(filename, 'r') as F:
        for line in F:
            x1, x2, Y = line.split(',')
            dataset.append((float(x1), float(x2), float(Y)))


def scramble(old):
    '''return a scrambled list based on old list'''
    new = old[:]
    random.shuffle(new)
    return new


def dispLoss(errorList, epochList):
    plt.plot(epochList, errorList)
    plt.xlabel('Number of Epoch')
    plt.ylabel('Mean Square Error')
    plt.show()


def avg(pylist):
    '''return the average of list'''
    return sum(pylist) / len(pylist)


def tabulate(csvfile, x1, x2, Y, t):
    with open(csvfile, mode='a+') as F:
        F_writer = csv.writer(
            F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        F_writer.writerow([x1, x2, t, Y])


class NeuralNetwork:
    def __init__(self):
        # np.random.seed(1) ????

        self.W1 = np.zeros((2,2)) #initialize weights to first layer
        self.W2 =  np.zeros((2,2)) #initialize weights to output
        self.bLayer = np.zeros((2,1)) #initialize hidden layer bias
        self.bOutput = np.zeros((1,1)) #initialize output bias

        self.P = None #input
        self.outputlayer = None #output of hidden layer
        self.A = None # final output
        self.n1 = None #net output N = WP n1 = w11p1 + w21p2
        self.n2 = None #net output of last layer
        

        print('Initial Weights and Biases')
        print(f'w1 = {self.W1}')
        print(f'b1 = {self.bLayer}')
        print(f'w2 = {self.W2}')
        print(f'b2 = {self.bOutput}')

    def s1(self, t, s2):
        ''' sensitivity of first layer '''
        return self.dsig(self.n1) @ np.transpose(self.W2) @ s2

    def s2(self, t):
        ''' sensitivity of last layer -2*F(n)*E '''
        return -2 * (self.dlin(self.n2) @ self.error(t, self.A))

    def error (self, t, A):
        ''' error function '''
        return (t - A)

    def sig(self, n):
        '''Sigmoid Activation Function'''
        return 1 / (1 + np.exp(-n))

    def dsig(self, n):
        '''Derivative of last layer which is the diagonals only'''
        res = np.zeros((n.size, n.size))

        for i in range (n.size):
            res[i][i] = self.sig(n[i][0]) * (1 - self.sig(n[i][0]))

        return res

    def lin(self, n):
        '''pure line function'''
        return n

    def dlin(self, n):
        '''derivative of linear function which is just the identity matrix'''
        return np.identity(n.size)


    def feedfoward(self, P):
        ''' feed foward function to compute layers and return the output '''
        self.P = P
        self.n1 = (self.W1 @ self.P) + self.bLayer
        self.outputlayer = self.sig(self.n1)
        self.n2 = (self.W2 @ self.outputlayer) + self.bOutput
        self.A = self.lin(self.n2)

        return self.A

    def backpropagrate(self, t, alpha=0.5):
        '''weights update'''
        s2 = self.s2(t)
        s1 = self.s1(t, s2)

        '''Wnew = Wold + (-alpha)*S(A_transpose)  '''
        self.W2 = self.W2 + -alpha * (s2 @ np.transpose(self.outputlayer))
        self.bOutput = self.bOutput + -alpha * s2
        self.W1 = self.W1 + -alpha * (s1 @ np.transpose(self.P))
        self.bLayer = self.bLayer + -alpha * s1


    def train(self, dataset, epoch = 1000):
        ''' train network '''
        errorAvgMin = 1
        epochList = []
        errorList = []
        errorSum = 0
        count = 0

        for i in range (epoch):
            for x1, x2, t in dataset:
                p = np.array([[x1], [x2]])
                self.feedfoward(p)
                ''' loss function returns 1x1 vector so used [0][0] '''
                errorSum += self.Loss(t, self.A)[0][0]
                tabulate('training.csv', x1, x2, (self.A), t)

                count += 1
                self.backpropagrate(t)

            epochList.append(i+1)
            errorAvg = errorSum / count
            errorList.append(errorAvg)
            
            if errorAvg < errorAvgMin:
                errorAvgMin = errorAvg

        errorSum = 0 #reset sum
        count = 0 #reset count
    
    def predict(self, x1, x2):
        ''' predict the output based on input '''
        p = np.array([[x1], [x2]])
        A = self.feedfoward(p)

        return A   

    def Loss(self, t, A):
        ''' return the error squared  '''
        MSE = (t - A)**2
        
        return MSE

    def test(self, testdata):
        ''' test the performance'''
        count = 0
        for x1, x2, t in testdata:
            A = self.predict(x1,x2)
            error = self.Loss(t, A)
            tabulate('data/test.csv', x1, x2, A, error)


# %% Case 1 : i/5 j/5
'''Case 1'''

datafile = 'data/trainingdata.txt'
testfile = 'data/test.txt'

genData(datafile, 5)
genRandData(testfile, 50)

dataset = []
parseData(datafile, dataset)
dataset = scramble(dataset)

# print(dataset)

NN = NeuralNetwork()  # create NN with 3 hidden neurons

NN.train(dataset, 1000)  # train the neural network

testset = []
parseData(testfile, testset)

NN.test(testset)

# %% Case 2 i/10 j/10
'''Case 2 '''

datafile = './data/data.txt'
testfile = './data/test.txt'

genData(datafile, 10)
genRandData(testfile, 50)

dataset = []
parseData(datafile, dataset)
dataset = scramble(dataset)

# print(dataset)

NN = NeuralNet()  # create NN with 3 hidden neurons

NN.train(dataset, 500)  # train the neural network

testset = []
parseData(testfile, testset)

NN.test(testset)


# %% Case 3 random x1, and x2
'''Case 3'''


datafile = './data/dataR.txt'
testfile = './data/testR.txt'

genRandData(datafile, 100)
genRandData(testfile, 50)

dataset = []
parseData(datafile, dataset)
dataset = scramble(dataset)

# print(dataset)

NN = NeuralNet(2)  # create NN with 3 hidden neurons

NN.train(dataset, 500)  # train the neural network

testset = []
parseData(testfile, testset)

NN.test(testset)