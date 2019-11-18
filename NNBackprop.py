import numpy as np
import csv
from datamanip import tabulate

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

    