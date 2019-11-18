import matplotlib.pyplot as plt
from math import pi, sin
import random
import csv


def function(x1, x2):
    '''
    Input: 2 decimal numbers from 0 to 1 inclusive
    Ouput: 1 decimal number
    time: O(1), space: O(1)
    '''
    return (sin(2*pi*x1) * sin(2*pi*x2))


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
        for i in range(n):
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