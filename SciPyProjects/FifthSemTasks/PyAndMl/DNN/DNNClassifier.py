import os

import numpy as np
import pandas as pd


def soft_max(x: np.array, derivative: bool = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


class DeepNeuralNetwork:
    def __init__(self, topology: [int]):
        self.topology = topology
        self.weights = {}
        for i in range(0, len(topology) - 1):
            self.weights[f'W{i}'] = np.random.random((topology[i + 1], topology[i])) * np.sqrt(1. / topology[i + 1])

    def feed_forward(self, x_vector: np.array):
        params = {'A0': x_vector}

        for i in range(1, len(self.topology)):
            params[f'Z{i}'] = np.dot(self.weights[f'W{i - 1}'], params[f'A{i - 1}'])
            params[f'A{i}'] = soft_max(params[f'Z{i}'])

        return params

    def back_propagation(self, params: dict, real_y_val: np.array):
        change_w = {}
        len_last = len(self.topology) - 1
        output = params[f'A{len_last}']

        error = 2 * (output - real_y_val) / self.topology[-1] * soft_max(params[f'Z{len_last}'], derivative=True)
        change_w[f'W{len_last}'] = np.outer(error, params[f'A{len_last - 1}'])

        for i in reversed(range(2, len(self.topology))):
            error = np.dot(self.weights[f'W{i - 1}'].T, error) * soft_max(params[f'Z{i - 1}'], derivative=True)
            change_w[f'W{i - 1}'] = np.outer(error, params[f'A{i - 2}'])

        return change_w

    def 


def vectorize(y: int):
    res = np.zeros(10)
    res[y] = 1.0
    return res


if __name__ == '__main__':
    mnist = pd.read_csv("..\\Resources\\mnist_test.csv")

    dnn = DeepNeuralNetwork([784,512, 256, 128, 64, 10])
    res_foward = dnn.feed_forward(mnist.values[0][1:])
    back_proop = dnn.back_propagation(res_foward, vectorize(mnist.values[0][1]))
    print(res_foward)
