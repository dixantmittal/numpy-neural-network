import numpy as np

import NeuralNetwork as nn

# samples = 1000
# Data, Y = dt.make_classification(n_samples=samples, n_features=2, n_redundant=0)
# msk = np.random.rand(len(Data)) < 0.75
#
# from sklearn import datasets as dt
#
# Data, Y = dt.load_breast_cancer(return_X_y=True)
# msk = np.random.rand(len(Data)) < 0.80
# Data = Data[:,0:2]
#
# np.save("/Users/dixantmittal/Downloads/TrainingData", Data[msk])
# np.save("/Users/dixantmittal/Downloads/TestingData", Data[~msk])
# np.save("/Users/dixantmittal/Downloads/TrainingY", Y[msk])
# np.save("/Users/dixantmittal/Downloads/TestingY", Y[~msk])
#
# print(len(Data[msk]))

# print("Loading data...")
# trainX = np.load("/Users/dixantmittal/Downloads/TrainingData.npy")
# testX = np.load("/Users/dixantmittal/Downloads/TestingData.npy")
# trainY = np.load("/Users/dixantmittal/Downloads/TrainingY.npy")
# testY = np.load("/Users/dixantmittal/Downloads/TestingY.npy")
# print("Data loaded!")
#


# N = 100  # number of points per class
# D = 2  # dimensionality
# K = 3  # number of classes
# X = np.zeros((N * K, D))  # data matrix (each row = single example)
# y = np.zeros(N * K, dtype='uint8')  # class labels
# for j in range(K):
#     ix = range(N * j, N * (j + 1))
#     r = np.linspace(0.0, 1, N)  # radius
#     t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
#     X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
#     y[ix] = j

print("Loading data...")
X_data = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/x_train.csv', delimiter=',', dtype=int)
Y_data = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/y_train.csv', delimiter=',', dtype=int)
# Y_data = Y_data.reshape((len(Y_data), 1))
print("Data loaded!")

print("Splitting data 80-20 for validations.")
msk = np.random.rand(len(X_data)) < 0.80
X_train = X_data[msk]
X_validate = X_data[~msk]
Y_train = Y_data[msk]
Y_validate = Y_data[~msk]
X_test = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/x_test.csv', delimiter=',', dtype=int)
Y_test = np.genfromtxt('/Users/dixantmittal/Downloads/assignment1/Question2_123/y_test.csv', delimiter=',', dtype=int)

# # Neural net with 2 hidden layers [100,40]
# network = nn.NeuralNetwork(classSize=4, hiddenLayersSizes=[100, 40], batchSize=128, learningRate=1e-1,
#                            regularizationRate=1e-4)
#
# network.train(X_train=X_train, Y_train=Y_train, epoch=10000)

# Neural net with 2 hidden layers [28,28,28,28,28,28]
network = nn.NeuralNetwork(classSize=4, hiddenLayersSizes=[28, 28, 28, 28, 28, 28], batchSize=128, learningRate=1e0,
                           regularizationRate=0)

network.train(X_train=X_train, Y_train=Y_train, epoch=10000)

# network = nn.NeuralNetwork(classSize=4,
#                            hiddenLayersSizes=[14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
#                                               14, 14, 14, 14, 14, 14, 14, 14, 14, 14], batchSize=1000, learningRate=1e0,
#                            regularizationRate=0)
#
# network.train(X_train=X_train, Y_train=Y_train, epoch=5000)

print('training accuracy: %.2f' % (np.mean(network.predict(X_train) == Y_train) * 100))
print('validation accuracy: %.2f' % (np.mean(network.predict(X_validate) == Y_validate) * 100))
print('test accuracy: %.2f' % (np.mean(network.predict(X_test) == Y_test) * 100))
