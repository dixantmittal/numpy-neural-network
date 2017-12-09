import numpy as np


def init_layers(n_features, n_classes, architecture):
    weights = {}
    bias = {}

    n_layers = len(architecture)

    input_dim = n_features

    for i in range(n_layers):
        hidden_dim = architecture[i]

        # Xavier's initialization
        weights[i] = np.random.randn(input_dim, hidden_dim) * (2 / (input_dim + hidden_dim))
        bias[i] = np.zeros((1, hidden_dim))

        input_dim = hidden_dim

    # final layer
    weights[n_layers] = np.random.randn(input_dim, n_classes) * (2 / (input_dim + n_classes))
    bias[n_layers] = np.zeros((1, n_classes))

    return weights, bias


def split_data(X, y, split=.95):
    n_samples, _ = X.shape
    mask = np.random.rand(n_samples) < split

    X_train = X[mask]
    y_train = y[mask]
    X_dev = X[~mask]
    y_dev = y[~mask]

    return X_train, y_train, X_dev, y_dev


def get_batch(X, y, batch_size):
    n_samples, _ = X.shape
    batch_mask = np.random.choice(n_samples, batch_size)

    X_batch = X[batch_mask]
    y_batch = y[batch_mask]

    return X_batch, y_batch
