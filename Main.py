import NeuralNetwork as nn
import numpy as np
import AccuracyTest as at

# samples = 1000
# Data, Y = dt.make_classification(n_samples=samples, n_features=2, n_redundant=0)
# msk = np.random.rand(len(Data)) < 0.75
#
# trainData = Data[msk]
# testData = Data[~msk]
# trainY = Y[msk]
# testY = Y[~msk]

trainData = np.load("/Users/dixantmittal/Downloads/TrainingData.npy")
testData = np.load("/Users/dixantmittal/Downloads/TestingData.npy")
trainY = np.load("/Users/dixantmittal/Downloads/TrainingY.npy")
testY = np.load("/Users/dixantmittal/Downloads/TestingY.npy")

network = nn.NeuralNetwork(2, [3, 3, 3])
network.train(trainingX=trainData, trainingY=trainY, epoch=10000000, learningRate=0.03)
output = network.test(testData)

print("Train Accuracy: ", at.getAccuracy(network.test(trainData), trainY))
print("Test Accuracy: ", at.getAccuracy(output, testY))
