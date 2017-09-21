import numpy as np


class HiddenLayer(object):
    def __init__(self, inputSize, layerSize, regularizationRate, learningRate):
        self.weights = 0.1 * np.random.randn(inputSize, layerSize)
        self.bias = np.zeros((1, layerSize))
        self.reg = regularizationRate
        self.learningRate = learningRate

    def forwardPropagation(self, X):
        self.signals = np.dot(X, self.weights) + self.bias
        self.activations = np.maximum(0, self.signals)
        return self.activations

    def backwardPropagation(self, nextLayer):
        self.gradient = np.dot(nextLayer.gradient, nextLayer.weights.T)
        self.gradient[self.activations <= 0] = 0

    def adjustWeights(self, X):
        self.dweights = np.dot(X.T, self.gradient)
        self.dbias = np.sum(self.gradient, axis=0, keepdims=True)
        self.dweights += self.reg * self.weights

        self.weights = self.weights - self.learningRate * self.dweights
        self.bias = self.bias - self.learningRate * self.dbias
