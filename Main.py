import matplotlib.pyplot as plt
import numpy as np

import neural_network as nn

print("Loading data...")
X = np.load('/Users/dixantmittal/Downloads/X.npy')
y = np.load('/Users/dixantmittal/Downloads/y.npy')
print("Data loaded!")

network = nn.NeuralNetwork(n_features=30, n_classes=4, architecture=[100])
network.train(X, y, epochs=10, verbose=False, optimizer='adam', alpha=1e-1)

plt.plot(range(len(network.training_loss)), network.training_loss, "ro", ms=1, label="training loss")
plt.show()
