import OutputLayer as ol
import HiddenLayer as hl
import numpy as np
from sklearn import datasets as dt
import random
import AccuracyTest as at

samples = 200
Data, Y = dt.make_classification(n_samples=samples, n_features=2, n_redundant=0)
msk = np.random.rand(len(Data)) < 0.75

trainData = Data[msk]
testData = Data[~msk]
trainY = Y[msk]
testY = Y[~msk]

print("Split: ", len(trainData))

HL1 = hl.HiddenLayer(2, 2)
OL = ol.OutputLayer(2)

for i in range(0, 10000):
    n = random.randint(0, len(trainData) - 1)
    X = np.asmatrix(trainData[n, :]).transpose()

    HL1.forwardPropagation(X)
    OL.forwardPropagation(HL1.activations)

    OL.backwardPropagation(trainY[n])
    HL1.backwardPropagation(OL)

    HL1.adjustWeights(0.3, X)
    OL.adjustWeights(0.3, HL1.activations)

print(HL1.weights)
print(OL.weights)

at.getAccuracy(testData, testY, HL1, OL)