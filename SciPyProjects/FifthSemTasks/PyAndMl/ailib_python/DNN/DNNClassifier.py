import time
from typing import Callable

import numpy
import pandas as pd

from PyAndMl.ailib_python.MathUtils.ActivationFunctions import *
import cupy as np




def vector_from_val(y: int):
    res = np.zeros(10)
    res[int(y)] = 1.0
    return res


def array_vectors_from_val(y_dataset_labels: np.array):
    result = []
    for item in y_dataset_labels:
        result.append(vector_from_val(item))
    return np.array(result)


class DeepNeuralNetwork:
    def __init__(self, topology: [(int, Callable)], epochs: int = 1, learn_rate: float = 0.001):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layers_size = [i[0] for i in topology]
        self.layers_activations = [i[1] for i in topology]
        self.len_last = len(self.layers_size) - 1
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
        len_last = self.len_last
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
            output = self.feed_forward(x)[f'A{self.len_last}']
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(np.array(predictions))

    def cross_entropy(self, inputs, labels):
        out_num = labels.shape[0]
        p = np.sum(labels.reshape(1, out_num) * inputs)
        loss = -np.log(p)
        return loss

    def train(self, x_train_dataset: np.array, y_train_labels: np.array, x_values: np.array, y_values: np.array):
        start_time = time.time()

        for iteration in range(self.epochs):
            loss = 0
            for x, y in zip(x_train_dataset, y_train_labels):
                output = self.feed_forward(x)
                changes_to_w = self.back_propagation(output, y)
                self.update_network_parameters(changes_to_w)

                loss += self.cross_entropy(output[f'A{self.len_last}'], y)

            accuracy = self.compute_accuracy(x_values, y_values)
            loss /= len(x_train_dataset)

            print(f'Epoch: {iteration + 1}, Time Spent: {time.time() - start_time},'
                  f' Loss: {loss} Accuracy: {numpy.round(accuracy * 100, 2)}')

    def predict(self, x_value: np.array):
        len_last = len(self.layers_size) - 1
        return np.argmax(self.feed_forward(x_value)[f'A{len_last}'])


def x_y_split_data_frame(data_frame: pd.DataFrame, random_state: bool = False):
    x, y = data_frame.iloc[:, 1:].values.astype('float64'), data_frame.iloc[:, 0:1].values
    if random_state:
        x += np.random.normal(0, 1, x.shape)

    y = array_vectors_from_val(y.reshape(len(y)))

    return np.asarray(x / 255), np.asarray(y)


if __name__ == '__main__':
    train = pd.read_csv('../Resources/mnist_train.csv')
    test = pd.read_csv('../Resources/mnist_test.csv')

    x_train, y_train = x_y_split_data_frame(train)
    x_val, y_val = x_y_split_data_frame(test)

    dnn = DeepNeuralNetwork([(784, sigmoid),
                             (512, relu),
                             (10, soft_max)],
                            epochs=10, learn_rate=0.05)

    dnn.train(x_train, y_train, x_val, y_val)
    curr = 0
    for i in range(len(x_val)):
        real = np.argmax(y_val[i])
        pred = dnn.predict(x_val[i])
        curr += 1 if real == pred else 0
        print(f'Real image type is "{real}", DeepNeuralNetwork predict "{pred}"')

    print(curr / len(x_val))
