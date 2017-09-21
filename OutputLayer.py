import numpy as np


class OutputLayer(object):
    def __init__(self, inputSize, classSize, regularizationRate, learningRate):
        self.weights = np.random.randn(inputSize, classSize).astype(np.float64)
        self.bias = np.zeros((1, classSize))
        self.reg = regularizationRate
        self.learningRate = learningRate

    def forwardPropagation(self, X):
        self.score = np.dot(X, self.weights) + self.bias
        self.score = self.score - np.amax(self.score, axis=1, keepdims=True)
        self.exp_score = np.exp(self.score)
        self.probability = self.exp_score / np.sum(self.exp_score, axis=1, keepdims=True)
        predicted_class = np.argmax(self.score, axis=1)
        return predicted_class

    def backwardPropagation(self, Y):
        self.gradient = self.probability
        self.gradient[range(len(self.gradient)), Y] -= 1
        self.gradient = self.gradient / len(self.gradient)

    def adjustWeights(self, X):
        self.d_weights = X.T.dot(self.gradient)
        self.d_bias = np.sum(self.gradient, axis=0, keepdims=True)
        self.d_weights += self.reg * self.weights

        self.weights = self.weights - self.learningRate * self.d_weights
        self.bias = self.bias - self.learningRate * self.d_bias

    def calculateLoss(self, Y):
        # number of examples
        n = len(self.probability)

        logProbability = -self.score[range(n), Y] + np.log(np.sum(self.exp_score, axis=1, keepdims=True))
        loss = (np.sum(logProbability) / n + 0.5 * self.reg * np.sum(self.weights * self.weights)).astype(np.int64)
        return loss
