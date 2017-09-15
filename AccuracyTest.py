import random
import numpy as np


def getAccuracy(Test, Y, HL1, OL):
    totalErrors = 0

    for i in range(0, len(Test) - 1):

        X = np.asmatrix(Test[i, :]).transpose()

        HL1.forwardPropagation(X)
        OL.forwardPropagation(HL1.activations)

        if OL.activations > 0.5:
            output = 1
        else:
            output = 0

        if Y[i] != output:
            totalErrors += 1

    print("Accuracy: ", (len(Test) - totalErrors) / len(Test) * 100)
