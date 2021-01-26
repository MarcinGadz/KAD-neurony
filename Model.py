import numpy as np
import random
from tabulate import tabulate


class Model:
    def __init__(self, inputs, outputs, bias, layers, learnrate=0.2):
        """

        :param inputs: data to be treated as input for learning examples
        :type inputs: nd.array
        :param outputs: expected outputs for proper inputs
        :type outputs: nd.array
        :param bias: specify if neurons should have bias
        :type bias: bool
        :param layers: list containing number of neurons in each layer from input to output
        :type layers: list
        :param learnrate: specify learning ratio
        :type learnrate: float
        """
        self.inputs = inputs
        self.outputs = outputs
        self.bias = bias
        self.layers = layers
        self.learnrate = learnrate
        self.firstlayerres = np.empty([1, self.layers[0]])
        self.seclayerres = np.empty([1, self.layers[1]])
        self.weightsfirstlayer = np.zeros((self.layers[0], len(self.inputs[0])))
        self.weightssecondlayer = np.zeros([self.layers[1], self.layers[0]])

        for i in range(self.layers[0]):
            for j in range(len(self.inputs[0])):
                self.weightsfirstlayer[i][j] = random.uniform(-0.1, 0.1)
        for i in range(self.layers[1]):
            for j in range(self.layers[0]):
                self.weightssecondlayer[i][j] = random.uniform(-0.1, 0.1)

        if bias:
            self.biasesfirstlayer = np.ones(self.layers[0])
            self.biasesseclayer = np.ones(self.layers[1])
        else:
            self.biasesfirstlayer = np.zeros(self.layers[0])
            self.biasesseclayer = np.zeros(self.layers[1])

        self.startweights = [self.weightsfirstlayer, self.weightssecondlayer]
        self.err = []
        self.errors = []
        self.epochs = []
        # self.orerror = []
        # self.xorerror = []
        # self.anderror = []
        self.printstartweights()

    def logisticactivation(self, data):
        return 1.0 / (1 + np.exp(-data))

    def derivative(self, data):
        return data * (1 - data)

    def calcfirstlayer(self, dataset):
        res = np.dot(dataset, self.weightsfirstlayer.T)
        for i in range(self.layers[0]):
            res[i] += self.biasesfirstlayer[i]
        self.firstlayerres = self.logisticactivation(res)

    def calcseclayer(self):
        res = np.dot(self.firstlayerres, self.weightssecondlayer.T)
        for i in range(self.layers[1]):
            res[i] += self.biasesseclayer[i]
        self.seclayerres = self.logisticactivation(res)

    def updateweights(self, outputs, inputs):
        """

        :param outputs: expected outputs
        :type outputs: list
        :param inputs: specified inputs
        :type inputs: list
        """
        self.err = np.sum((outputs - self.seclayerres) ** 2)
        errors = outputs - self.seclayerres
        # self.orerror.append(abs(errors[0]))
        # self.xorerror.append(abs(errors[1]))
        # self.anderror.append(abs(errors[2]))
        for i in range(self.layers[1]):
            for j in range(self.layers[0]):
                self.weightssecondlayer[i][j] += self.learnrate * self.derivative(self.seclayerres[i]) * \
                                                 self.firstlayerres[j] * errors[i]
        firstlayererr = np.zeros((self.layers[0], 1))
        for i in range(self.layers[0]):
            for j in range(self.layers[1]):
                firstlayererr[i] += self.weightssecondlayer[j][i] * errors[j]
        for i in range(self.layers[0]):
            for j in range(len(self.inputs[0])):
                self.weightsfirstlayer[i][j] += self.learnrate * self.derivative(self.firstlayerres[i]) * inputs[j] * \
                                                firstlayererr[i]
        if self.bias:
            for i in range(self.layers[1]):
                self.biasesseclayer[i] += self.learnrate * self.derivative(self.seclayerres[i]) * errors[i]
            for i in range(self.layers[0]):
                self.biasesfirstlayer[i] += self.learnrate * self.derivative(self.firstlayerres[i]) * firstlayererr[i]

    def train(self, epochs=2000):
        for e in range(epochs):
            datasets = [0, 1, 2, 3]
            random.shuffle(datasets)
            tmperrors = []
            for dataset in datasets:
                self.calcfirstlayer(self.inputs[dataset])
                self.calcseclayer()
                self.updateweights(self.outputs[dataset], self.inputs[dataset])
                tmperrors.append(np.average(self.err))
            self.epochs.append(e)
            self.errors.append(np.average(tmperrors))

    def calcresult(self, inputs):
        self.calcfirstlayer(inputs)
        self.calcseclayer()
        return self.seclayerres

    def printstartweights(self):
        print("Warstwa 1")
        print(tabulate(self.startweights[0].T, headers=['Neuron 1', 'Neuron 2'], tablefmt='latex'))
        print("Warstwa 2")
        print(tabulate(self.startweights[1].T, headers=['Neuron 1', 'Neuron 2', 'Neuron 3'], tablefmt='latex'))
