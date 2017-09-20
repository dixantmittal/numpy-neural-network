import numpy as np

import HiddenLayer as hidden
import OutputLayer as out


class NeuralNetwork(object):
    def __init__(self, classSize, hiddenLayersSizes, batchSize, learningRate, regularizationRate):
        self.hiddenLayersSizes = hiddenLayersSizes
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.regularizationRate = regularizationRate
        self.classSize = classSize

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

    def train(self, X_train, Y_train, epoch):

        self.initLayers(len(X_train[0]))

        # init prev loss to a big number
        prevloss = 10000

        # forward propagate and calculate score
        i = 0
        for i in range(epoch):

            msk = np.random.rand(len(X_train)) < (self.batchSize / len(X_train))
            X = X_train[msk]
            for hl in self.hiddenLayers:
                X = hl.forwardPropagation(X)

            self.outputLayer.forwardPropagation(X)

            loss = self.outputLayer.calculateLoss(Y=Y_train[msk])
            print(loss)

            if abs(prevloss - loss) < 0.00000001:
                print(i)
                break
            prevloss = loss

            self.outputLayer.backwardPropagation(Y_train[msk])

            nextLayer = self.outputLayer
            for hl in reversed(self.hiddenLayers):
                hl.backwardPropagation(nextLayer)
                nextLayer = hl

            X = X_train[msk]
            for hl in self.hiddenLayers:
                hl.adjustWeights(X)
                X = hl.activations

            self.outputLayer.adjustWeights(X)

        if (i + 1 == epoch):
            print("epoch reached and network has not converged yet")

    def predict(self, testX):
        X = testX
        for hl in self.hiddenLayers:
            hl.forwardPropagation(X)
            X = hl.activations

        predictedClass = self.outputLayer.forwardPropagation(X=X)
        return predictedClass

    def calculateLoss(self, Y):
        self.outputLayer.calculateLoss(Y)
