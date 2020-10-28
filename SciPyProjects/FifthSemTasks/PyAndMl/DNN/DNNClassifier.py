import os
import time
from typing import Callable

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from joblib import parallel_backend

import numpy as np
import pandas as pd
from numba import njit, prange

from PyAndMl.MathUtils.ActivationFunctions import *


def vector_from_val(y: int):
    res = np.zeros(10)
    res[int(y)] = 1.0
    return res


class DeepNeuralNetwork:
    def __init__(self, topology: [(int, Callable)], epochs: int = 1, learn_rate: float = 0.001):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layers_size = [i[0] for i in topology]
        self.layers_activations = [i[1] for i in topology]
        self.weights = {}
        for i in range(0, len(topology) - 1):
            self.weights[f'W{i}'] = np.random.randn(self.layers_size[i + 1], self.layers_size[i]) \
                                    * np.sqrt(1. / self.layers_size[i + 1])

    def feed_forward(self, x_vector: np.array):
        params = {'A0': x_vector}

        for i in range(1, len(self.layers_size)):
            params[f'Z{i}'] = np.dot(self.weights[f'W{i - 1}'], params[f'A{i - 1}'])
            params[f'A{i}'] = self.layers_activations[i - 1](params[f'Z{i}'])

        return params

    def back_propagation(self, params: dict, real_y_val: np.array):
        change_w = {}
        len_last = len(self.layers_size) - 1
        output = params[f'A{len_last}']

        error = 2 * (output - real_y_val) / self.layers_size[-1] * self.layers_activations[-1](params[f'Z{len_last}'],
                                                                                               derivative=True)
        change_w[f'W{len_last}'] = np.outer(error, params[f'A{len_last - 1}'])

        for i in reversed(range(2, len(self.layers_size))):
            error = np.dot(self.weights[f'W{i - 1}'].T, error) * self.layers_activations[i - 2](params[f'Z{i - 1}'],
                                                                                                derivative=True)
            change_w[f'W{i - 1}'] = np.outer(error, params[f'A{i - 2}'])

        return change_w

    def update_network_parameters(self, changes_to_w: dict):
        for key, value in changes_to_w.items():
            self.weights[key[0] + str(int(key[1]) - 1)] -= self.learn_rate * value

    def compute_accuracy(self, x_values: np.array, y_values: np.array):
        predictions = []

        for x, y in zip(x_values, y_values):
            len_last = len(self.layers_size) - 1
            output = self.feed_forward(x)[f'A{len_last}']
            pred = np.argmax(output)
            predictions.append(pred == y)

        return np.mean(predictions)

    def train(self, x_train_dataset: np.array, y_train_labels: np.array, x_values: np.array, y_values: np.array):
        start_time = time.time()
        for iteration in range(self.epochs):
            for i in range(len(x_train_dataset)):
                output = self.feed_forward(x_train_dataset[i])
                changes_to_w = self.back_propagation(output, vector_from_val(y_train_labels[i]))
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_values, y_values)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_time, accuracy * 100
            ))

    def predict(self, x_value: np.array) -> np.ndarray[int]:
        len_last = len(self.layers_size) - 1
        return np.argmax(self.feed_forward(x_value)[f'A{len_last}'])


def x_y_split_data_frame(data_frame: pd.DataFrame, random_state: bool = False):
    x, y = data_frame.iloc[:, 1:].values.astype('float64'), data_frame.iloc[:, 0:1].values
    if random_state:
        x += np.random.normal(0, 1, x.shape)
    return x / 255, y.reshape(len(y))


if __name__ == '__main__':
    train = pd.read_csv('..\\Resources\\mnist_train.csv')
    test = pd.read_csv('..\\Resources\\mnist_test.csv')

    x_train, y_train = x_y_split_data_frame(train)
    x_val, y_val = x_y_split_data_frame(test)

    dnn = DeepNeuralNetwork([(784, sigmoid),
                             (256, sigmoid),
                             (32, relu),
                             (10, soft_max)],
                            epochs=5, learn_rate=0.1)

    dnn.train(x_train, y_train, x_val, y_val)

    for i in range(len(x_val)):
        print(f'Real image type is "{y_val[i]}", DeepNeuralNetwork predict "{dnn.predict(x_val[i])}"')

