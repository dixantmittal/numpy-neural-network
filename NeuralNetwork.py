import HiddenLayer as hidden
import OutputLayer as out


class NeuralNetwork(object):
    def __init__(self, classSize, hiddenLayersSizes, batchSize, learningRate, regularizationRate):
        self.hiddenLayersSizes = hiddenLayersSizes
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.regularizationRate = regularizationRate
        self.classSize = classSize
        self.loss = []
        self.start = 0
        self.end = 0

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

    def train(self, X_train, Y_train, maxIterations):

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

            loss = self.outputLayer.calculateLoss(Y=Y_batch)
            self.loss.append(loss)

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

            if i != 0 and i % (maxIterations / 10) == 0:
                print(i)
                # plt.plot(range(len(self.loss)), self.loss, "ro")
                # plt.xlabel("Iterations"   )
                # plt.ylabel("logLoss")
                # plt.show()
            i += 1

        if (i + 1 == maxIterations):
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
