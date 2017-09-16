import numpy as np
import Functions as fn


class HiddenLayer(object):
    def __init__(self, inputs, nodes):
        self.weights = np.random.rand(nodes, inputs)
        self.bias = np.random.rand(nodes, 1)
        self.gradient = np.random.rand(nodes, 1)

    def forwardPropagation(self, X):
        self.signals = np.add(np.dot(self.weights, X), self.bias)
        self.activations = fn.sigmoid(self.signals)

    def backwardPropagation(self, nextLayer):
        self.gradient = np.multiply(np.dot(nextLayer.weights.transpose(), nextLayer.gradient), self.selfGradient())

    def selfGradient(self):
        return np.multiply(1.0 - self.activations, self.activations)

    def adjustWeights(self, alpha, X):
        self.weights = self.weights - alpha * np.dot(self.gradient, X.transpose())
