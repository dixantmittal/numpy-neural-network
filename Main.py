import matplotlib.pyplot as plt
import numpy as np

import NeuralNetwork as nn


def getData():
    print("Loading data...")
    X_data = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/x_train.csv', delimiter=',',
                           dtype=int)
    Y_data = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/y_train.csv', delimiter=',',
                           dtype=int)
    print("Data loaded!")

    print("Splitting data 80-20 for validations.")
    msk = np.random.rand(len(X_data)) < 0.80
    X_train = X_data[msk]
    X_validate = X_data[~msk]
    Y_train = Y_data[msk]
    Y_validate = Y_data[~msk]
    X_test = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/x_test.csv', delimiter=',',
                           dtype=int)
    Y_test = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/y_test.csv', delimiter=',',
                           dtype=int)
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


X_train, Y_train, X_validate, Y_validate, X_test, Y_test = getData()

commonBatchSize = 128
commonIterations = 2000
commonAlpha = 1e-0

# ---------------------------------------Neural net with 2 hidden layers [100,40]---------------------------------------
network = nn.NeuralNetwork(classSize=4, hiddenLayersSizes=[100, 40], batchSize=commonBatchSize,
                           learningRate=commonAlpha,
                           regularizationRate=1e-3)

network.train(X_train=X_train, Y_train=Y_train, maxIterations=commonIterations)

print('training accuracy: %.2f' % (np.mean(network.predict(X_train) == Y_train) * 100))
print('validation accuracy: %.2f' % (np.mean(network.predict(X_validate) == Y_validate) * 100))
print('test accuracy: %.2f' % (np.mean(network.predict(X_test) == Y_test) * 100))
net_100_40_loss = network.loss
plt.plot(range(len(net_100_40_loss)), net_100_40_loss, "ro", ms=1, label="100-40 net")
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------Neural net with 6 hidden layers [28,28,28,28,28,28]---------------------------------
network = nn.NeuralNetwork(classSize=4, hiddenLayersSizes=[28, 28, 28, 28, 28, 28], batchSize=commonBatchSize,
                           learningRate=commonAlpha,
                           regularizationRate=1e-3)

network.train(X_train=X_train, Y_train=Y_train, maxIterations=commonIterations)

print('training accuracy: %.2f' % (np.mean(network.predict(X_train) == Y_train) * 100))
print('validation accuracy: %.2f' % (np.mean(network.predict(X_validate) == Y_validate) * 100))
print('test accuracy: %.2f' % (np.mean(network.predict(X_test) == Y_test) * 100))

net_6x28_loss = network.loss
plt.plot(range(len(net_6x28_loss)), net_6x28_loss, "go", ms=1, label="6x28 net")
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------Neural net with 28 hidden layers 28x14----------------------------------------
network = nn.NeuralNetwork(classSize=4,
                           hiddenLayersSizes=[14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                                              14, 14, 14, 14, 14, 14, 14, 14, 14, 14], batchSize=commonBatchSize,
                           learningRate=commonAlpha,
                           regularizationRate=1e-3)

network.train(X_train=X_train, Y_train=Y_train, maxIterations=commonIterations)

print('training accuracy: %.2f' % (np.mean(network.predict(X_train) == Y_train) * 100))
print('validation accuracy: %.2f' % (np.mean(network.predict(X_validate) == Y_validate) * 100))
print('test accuracy: %.2f' % (np.mean(network.predict(X_test) == Y_test) * 100))

net_28x14_loss = network.loss
plt.plot(range(len(net_28x14_loss)), net_28x14_loss, "bo", ms=1, label="28x14 net")
# ----------------------------------------------------------------------------------------------------------------------



plt.xlabel("Iterations")
plt.ylabel("log(Loss)")
plt.legend()
plt.show()
