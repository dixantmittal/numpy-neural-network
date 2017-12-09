import numpy as np


def relu_forward(X, W, b):
    # calculate dot
    h = np.dot(X, W) + b

    # apply relu activation
    a = np.maximum(h, 0)

    return a, (X, W, b)


def relu_backward(dout, cache):
    X, W, b = cache

    dW = np.dot(X.transpose(), dout)
    db = np.sum(dout, axis=0)

    dx = np.dot(dout, W.transpose())
    dx[X == 0] = 0

    return dx, dW, db


def affine_forward(X, W, b):
    # calculate dot
    a = np.dot(X, W) + b

    return a, (X, W, b)


def affine_backward(dout, cache):
    X, W, b = cache

    dW = np.dot(X.transpose(), dout)
    db = np.sum(dout, axis=0)

    dx = np.dot(dout, W.transpose())

    return dx, dW, db


def dropout_forward(X, dropout):
    # find dropout masks
    p = np.random.rand(*X.shape) < (1 - dropout)

    # turn masked values off
    a = X * p / (1 - dropout)
    return a, (X, p, dropout)


def dropout_backward(dout, cache):
    X, p, dropout = cache
    dx = dout * p * (1 - dropout)
    return dx


def softmax_loss(score, y):
    n_samples = len(y)
    shifted_score = score - np.max(score, axis=1, keepdims=True)
    exp_score = np.exp(shifted_score)
    sum_exp_score = np.sum(exp_score, axis=1, keepdims=True)

    prob = exp_score / sum_exp_score
    log_prob = shifted_score - sum_exp_score

    loss = -np.sum(log_prob[range(n_samples), y]) / n_samples

    dout = prob
    dout[range(n_samples), y] -= 1
    dout = dout / n_samples

    return loss, dout
