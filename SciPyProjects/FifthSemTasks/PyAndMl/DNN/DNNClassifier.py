import os
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def soft_max(x: np.array, derivative: bool = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


def vector_from_val(y: int):
    res = np.zeros(10)
    res[int(y)] = 1.0
    return res


class DeepNeuralNetwork:
    def __init__(self, topology: [int], epochs: int = 1, learn_rate: float = 0.001):
        self.learn_rate = learn_rate
        self.epochs = epochs
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

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.weights[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.feed_forward(x)
                changes_to_w = self.back_propagation(output, vector_from_val(y))
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_time, accuracy * 100
            ))


if __name__ == '__main__':
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    x = (x / 255).astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    dnn = DeepNeuralNetwork([784, 512, 256, 128, 64, 10])
    dnn.train(x_train, y_train, x_val, y_val)

