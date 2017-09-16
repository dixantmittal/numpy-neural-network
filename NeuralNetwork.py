import HiddenLayer as hidden
import OutputLayer as out
import numpy as np


class NeuralNetwork(object):
    isTrained = False

    def __init__(self, numberOfInputs, hiddenLayersNodes):
        self.layers = []
        inputs = numberOfInputs
        for i in range(0, len(hiddenLayersNodes)):
            self.layers.append(hidden.HiddenLayer(inputs, hiddenLayersNodes[i]))
            inputs = hiddenLayersNodes[i]

        self.layers.append(out.OutputLayer(inputs=inputs))

    def train(self, trainingX, trainingY, epoch, learningRate):
        i = 0
        while True:
            j = 0
            for j in range(0, len(trainingX)):

                X = np.asmatrix(trainingX[j, :]).transpose()
                for hl in self.layers:
                    hl.forwardPropagation(X)
                    X = hl.activations

                expected = trainingY[j]
                for hl in reversed(self.layers):
                    hl.backwardPropagation(expected)
                    expected = hl

                X = np.asmatrix(trainingX[j, :]).transpose()
                for hl in self.layers:
                    hl.adjustWeights(learningRate, X)
                    X = hl.activations
            i += j
            if i > epoch:
                break

        for hl in self.layers:
            print(hl.weights)
        self.isTrained = True

    def test(self, testX):
        if self.isTrained != True:
            print("Network not trained!")
            return

        testY = []
        for i in range(0, len(testX)):
            X = np.asmatrix(testX[i, :]).transpose()
            for hl in self.layers:
                hl.forwardPropagation(X)
                X = hl.activations
            testY.append(X)

        return testY
