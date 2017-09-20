import numpy as np


class OutputLayer(object):
    def __init__(self, inputSize, classSize, regularizationRate, learningRate):
        self.weights = 0.01 * np.random.randn(inputSize, classSize)
        self.bias = np.zeros((1, classSize))
        self.reg = regularizationRate
        self.learningRate = learningRate

    def forwardPropagation(self, X):
        self.score = np.dot(X, self.weights) + self.bias
        self.exp_score = np.exp(self.score)
        self.probability = self.exp_score / np.sum(self.exp_score, axis=1, keepdims=True)
        predicted_class = np.argmax(self.score, axis=1)
        return predicted_class

    def backwardPropagation(self, y):
        self.gradient = self.probability
        self.gradient[range(len(self.gradient)), y] -= 1
        self.gradient = self.gradient / len(self.gradient)

    def adjustWeights(self, X):
        self.dweights = X.T.dot(self.gradient)
        self.dbias = np.sum(self.gradient, axis=0, keepdims=True)
        self.dweights += self.reg * self.weights

        self.weights = self.weights - self.learningRate * self.dweights
        self.bias = self.bias - self.learningRate * self.dbias

    def calculateLoss(self, Y):
        # number of examples
        n = len(self.probability)

        logProbability = -np.log(self.probability[range(n), Y])
        loss = np.sum(logProbability) / n + 0.5 * self.reg * np.sum(self.weights * self.weights)
        return loss
