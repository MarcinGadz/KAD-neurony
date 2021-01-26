import numpy as np

import matplotlib.pyplot as plt

from Model import Model


def trainfirstmodel():
    ins = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outs = [[0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]]
    rate = 0.1
    neu = Model(np.asarray(ins), np.asarray(outs), False, [2, 3], learnrate=rate)
    neu.train(2000)
    # startweightsfirst, startweightssecond = neu.startweights

    plt.plot(neu.epochs, neu.errors)
    plt.title("Learning rate: " + str(rate))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

    # plt.plot(neu.epochs, neu.orerror)
    # plt.title("OR")
    # plt.xlabel('Epoch')
    # plt.ylabel('Error')
    # plt.show()
    #
    # plt.plot(neu.epochs, neu.anderror)
    # plt.title("AND")
    # plt.xlabel('Epoch')
    # plt.ylabel('Error')
    # plt.show()
    #
    # plt.plot(neu.epochs, neu.xorerror)
    # plt.title("XOR")
    # plt.xlabel('Epoch')
    # plt.ylabel('Error')
    # plt.show()


def trainsecondmodel():
    ins = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    outs = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    rate = 4
    neu = Model(np.asarray(ins), np.asarray(outs), False, [2, 4], learnrate=rate)
    neu.train(5000)

    plt.plot(neu.epochs, neu.errors)
    plt.title("Learning rate: " + str(rate))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    trainsecondmodel()
