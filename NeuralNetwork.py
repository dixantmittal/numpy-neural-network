import numpy as np

import HiddenLayer as hidden
import OutputLayer as out


class NeuralNetwork(object):
    def __init__(self, classSize, hiddenLayersSizes, batchSize, learningRate, regularizationRate, metricsRate=10):
        self.hiddenLayersSizes = hiddenLayersSizes
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.regularizationRate = regularizationRate
        self.classSize = classSize
        self.trainingLoss = []
        self.testingLoss = []
        self.trainingAccuracy = []
        self.testingAccuracy = []
        self.start = 0
        self.end = 0
        self.metricsRate = metricsRate

    def initLayers(self, inputSize):
        self.hiddenLayers = []
        for i in self.hiddenLayersSizes:
            self.hiddenLayers.append(
                hidden.HiddenLayer(inputSize=inputSize, layerSize=i,
                                   regularizationRate=self.regularizationRate,
                                   learningRate=self.learningRate))
            inputSize = i

        self.outputLayer = out.OutputLayer(inputSize=inputSize, classSize=self.classSize,
                                           regularizationRate=self.regularizationRate,
                                           learningRate=self.learningRate)

    def train(self, X_train, Y_train, X_test, Y_test, maxIterations):

        n_samples, n_features = X_train.shape

        self.initLayers(n_features)

        # forward propagate and calculate score
        i = 0
        while i < maxIterations:

            X_batch, Y_batch = self.getBatch(X_train, Y_train, self.batchSize)

            X = X_batch
            for hl in self.hiddenLayers:
                X = hl.forwardPropagation(X)

            self.outputLayer.forwardPropagation(X)

            # calculating training loss
            if i % self.metricsRate == 0:
                self.trainingLoss.append(self.outputLayer.calculateLoss(Y=Y_batch))

            self.outputLayer.backwardPropagation(Y=Y_batch)

            nextLayer = self.outputLayer
            for hl in reversed(self.hiddenLayers):
                hl.backwardPropagation(nextLayer)
                nextLayer = hl

            X = X_batch
            for hl in self.hiddenLayers:
                hl.adjustWeights(X)
                X = hl.activations

            self.outputLayer.adjustWeights(X)

            i += 1
            if i % (maxIterations / 10) == 0:
                print(int(i * 100 / maxIterations), "% complete")

            if i % self.metricsRate == 0:
                # calculating training accuracy
                self.calculateAccuracy(self.trainingAccuracy, X_batch, Y_batch)

                # calculating testing loss
                self.calculateLoss(self.testingLoss, X_test, Y_test)

                # calculating testing accuracy
                self.calculateAccuracy(self.testingAccuracy, X_test, Y_test)

        if (i + 1 == maxIterations):
            print("epoch reached and network has not converged yet")

    def calculateAccuracy(self, history, X_test, Y_test):
        history.append(np.mean(self.predict(X_test) == Y_test) * 100)

    def calculateLoss(self, loss, X_test, Y_test):
        X = X_test
        for hl in self.hiddenLayers:
            X = hl.forwardPropagation(X)
        self.outputLayer.forwardPropagation(X)
        loss.append(self.outputLayer.calculateLoss(Y=Y_test))

    def predict(self, testX):
        X = testX
        for hl in self.hiddenLayers:
            hl.forwardPropagation(X)
            X = hl.activations

        predictedClass = self.outputLayer.forwardPropagation(X=X)
        return predictedClass

    def getBatch(self, X_train, Y_train, batchSize):
        n_samples = len(X_train)
        self.start = self.end
        if self.start >= n_samples:
            self.start = 0
        self.end = self.start + batchSize
        if self.end > n_samples:
            self.end = n_samples

        X = X_train[self.start:self.end]
        Y = Y_train[self.start:self.end]
        return (X, Y)
