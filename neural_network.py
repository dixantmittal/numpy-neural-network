from tqdm import tqdm

from layer_utils import *
from layers import *


class NeuralNetwork(object):
    def __init__(self, n_features, n_classes, architecture, dropout=0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.architecture = architecture
        self.dropout = dropout
        self.n_layers = len(architecture) + 1

        # initialize weights map
        self.weights, self.bias = init_layers(n_features, n_classes, architecture)

        # metrics
        self.training_loss = []

    def train(self, X, y, epochs=10, alpha=3e-3, delta=1e-4, batch_size=128, optimizer='adam',
              optimizer_params={'beta1': 0.9, 'beta2': 0.999},
              verbose=True):

        X_train, y_train, X_dev, y_dev = split_data(X, y)
        n_samples, n_features = X_train.shape

        # forward propagate and calculate score
        n_iterations = epochs * n_samples // batch_size

        # reset adam params
        m_w, v_w, m_b, v_b = {}, {}, {}, {}

        for iteration in tqdm(range(n_iterations)):

            X_batch, y_batch = get_batch(X_train, y_train, batch_size)

            # forward pass
            activation, cache = {}, {}
            h = X_batch
            for step in range(self.n_layers - 1):
                activation[step], cache[step] = relu_forward(h, self.weights[step], self.bias[step])
                h = activation[step]

            # calculate softmax score
            score, cache[self.n_layers - 1] = affine_forward(h, self.weights[self.n_layers - 1],
                                                             self.bias[self.n_layers - 1])

            # calculate score
            loss, dout = softmax_loss(score, y_batch)

            # Backpropagation
            dW, db = {}, {}
            dout, dW[self.n_layers - 1], db[self.n_layers - 1] = affine_backward(dout, cache[self.n_layers - 1])
            for step in reversed(range(self.n_layers - 1)):
                dout, dW[step], db[step] = relu_backward(dout, cache[step])

            # adjust weights
            for step in range(self.n_layers - 1):

                if optimizer == 'adam':
                    # adam optimizer
                    m_w[step] = m_w.get(step, 0) * optimizer_params['beta1'] + (1 - optimizer_params['beta1']) * dW[
                        step]
                    m_b[step] = m_b.get(step, 0) * optimizer_params['beta1'] + (1 - optimizer_params['beta1']) * db[
                        step]

                    v_w[step] = v_w.get(step, 0) * optimizer_params['beta2'] + (1 - optimizer_params['beta2']) * (
                        dW[step] ** 2)
                    v_b[step] = v_b.get(step, 0) * optimizer_params['beta2'] + (1 - optimizer_params['beta2']) * (
                        db[step] ** 2)

                    self.weights[step] = self.weights[step] - alpha * m_w[step] / (np.sqrt(v_w[step]) + 1e-10)
                    self.bias[step] = self.bias[step] - alpha * m_b[step] / (np.sqrt(v_b[step]) + 1e-10)

                elif optimizer == 'sgd':
                    self.weights[step] = self.weights[step] - alpha * dW[step]
                    self.bias[step] = self.bias[step] - alpha * db[step]
                else:
                    print('invalid optimizer')

            self.training_loss.append(loss)
            if verbose and iteration % 100 == 0:
                print('---------------------')
                print('iteration: ', iteration)
                print("loss: ", loss)

        if verbose:
            print('training accuracy: ', np.mean(self.predict(X_train) == y_train) * 100)
            print('validation accuracy: ', np.mean(self.predict(X_dev) == y_dev) * 100)

    def predict(self, X):

        # forward pass
        h = X
        for step in range(self.n_layers - 1):
            h, _ = relu_forward(h, self.weights[step], self.bias[step])

        # calculate softmax score
        score, _ = affine_forward(h, self.weights[self.n_layers - 1],
                                  self.bias[self.n_layers - 1])

        label = np.argmax(score, axis=1)

        return label
