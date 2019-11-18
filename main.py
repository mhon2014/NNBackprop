from neuralNet import NeuralNet
import datamanip as dm


def main():
    '''Experiment on NN tos simulate y= f(x1, x2) = sin(2*pi*x1)*sin(2*pi*x2)'''

    datafile = 'data.txt'
    # datafile2 = './data/data2.txt'
    # datafile3 = './data/dataR.txt'
    testfile = 'test.txt'
    # testfile2 = './data/testR.txt'

    dm.genData(datafile, 5)
    # dm.genData(datafile2, 10)
    # dm.genRandData(datafile3, 25)

    dm.genRandData(testfile, 10)

    dataset = []
    dm.parseData(datafile, dataset)
    # dataset = dm.scramble(dataset)

    # print(dataset)

    NN = NeuralNet(3)  # create NN with 3 hidden neurons (best results)

    NN.train(dataset, 500)  # train the neural network

    testset = []
    dm.parseData(testfile, testset)

    NN.test(testset)

if __name__ == "__main__":
    print("LESGOOOOOO")
    pass