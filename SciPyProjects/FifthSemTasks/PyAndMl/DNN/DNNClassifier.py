import os
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from numba import jit, prange


def soft_max(x: np.array, derivative: bool = False):
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


def sigmoid(x: np.array, derivative: bool = False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))


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
            self.weights[f'W{i}'] = np.random.randn(topology[i + 1], topology[i]) * np.sqrt(1. / topology[i + 1])

    def feed_forward(self, x_vector: np.array):
        params = {'A0': x_vector}

        for i in range(1, len(self.topology)):
            params[f'Z{i}'] = np.dot(self.weights[f'W{i - 1}'], params[f'A{i - 1}'])
            params[f'A{i}'] = sigmoid(params[f'Z{i}'])

        return params

    def back_propagation(self, params: dict, real_y_val: np.array):
        change_w = {}
        len_last = len(self.topology) - 1
        output = params[f'A{len_last}']

        error = 2 * (output - real_y_val) / self.topology[-1] * soft_max(params[f'Z{len_last}'], derivative=True)
        change_w[f'W{len_last}'] = np.outer(error, params[f'A{len_last - 1}'])

        for i in reversed(range(2, len(self.topology))):
            error = np.dot(self.weights[f'W{i - 1}'].T, error) * sigmoid(params[f'Z{i - 1}'], derivative=True)
            change_w[f'W{i - 1}'] = np.outer(error, params[f'A{i - 2}'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.weights[key[0] + str(int(key[1]) - 1)] -= self.learn_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            len_last = len(self.topology) - 1
            output = self.feed_forward(x)[f'A{len_last}']
            pred = np.argmax(output)
            predictions.append(pred == y)

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        len_last = len(self.topology) - 1
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.feed_forward(x)
                changes_to_w = self.back_propagation(output, vector_from_val(y))
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_time, accuracy * 100
            ))

    def predict(self, x_value: np.array) -> int:
        return self.feed_forward(x_value).max


def x_y_split_data_frame(data_frame, random_state: bool = False):
    x, y = data_frame.iloc[:, 1:].values.astype('float64'), data_frame.iloc[:, 0:1].values
    if random_state:
        x += np.random.normal(0, 1, x.shape)
    return x / 255, y.reshape(len(y))


if __name__ == '__main__':
    train = pd.read_csv('..\\Resources\\mnist_train.csv')
    test = pd.read_csv('..\\Resources\\mnist_test.csv')

    x_train, y_train = x_y_split_data_frame(train)
    x_val, y_val = x_y_split_data_frame(test)

    dnn = DeepNeuralNetwork([784, 512, 256, 128, 32, 64, 10], epochs=20)

    dnn.train(x_train, y_train, x_val, y_val)

