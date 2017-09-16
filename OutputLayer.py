import numpy as np
import Functions as fn


class OutputLayer(object):
    def __init__(self, inputs):
        self.weights = np.random.rand(1, inputs)
        self.bias = np.random.randint(0,10)

    def forwardPropagation(self, X):
        self.signals = np.add(np.dot(self.weights, X), self.bias)
        self.activations = fn.sigmoid(self.signals)

    def backwardPropagation(self, expected):
        self.gradient = self.selfGradient() * (self.activations - expected)

    def selfGradient(self):
        return (1 - self.activations) * self.activations

    def adjustWeights(self, alpha, X):
        self.weights = self.weights - alpha * self.gradient * X.transpose()
