import numpy as np


def soft_max(x: np.array, derivative: bool = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


def sigmoid(x: np.array, derivative: bool = False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))


def relu(x: np.array, derivative: bool = False):
    if not derivative:
        return np.maximum(0, x)
    else:
        return np.where(x <= 0, 0, 1)
