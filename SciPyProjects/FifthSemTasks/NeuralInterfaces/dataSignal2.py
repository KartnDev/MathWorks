from typing import Iterable
from scipy.stats.stats import pearsonr, spearmanr
import numpy as np


def function_square(x):
    return 3 * x ** 2


def function_line(x):
    return 2 * x + 3


def read_signals(path: str, index, last):
    res = []

    with open(path) as file:
        file_lines = file.read().split('\n')[index:last]

        for i in range(len(file_lines[0].split(' '))):
            res.append([])

        for line in file_lines:
            values = line.split(' ')
            for i, word in enumerate(values):
                res[i].append(float(word))
    return res


def get_interval_array_slice(array: Iterable, lower_bound: int, upper_bound: int):
    tickets_in_second = 30
    return array[tickets_in_second * lower_bound: tickets_in_second * upper_bound]


def correlation(x: Iterable, y: Iterable):
    x_sub = x - np.mean(x)
    y_sub = y - np.mean(y)

    return np.sum(x_sub * y_sub) / np.sqrt(np.sum(x_sub ** 2) * np.sum(y_sub ** 2))


if __name__ == '__main__':
    x_open, y_open, z_open = read_signals("Resource\\OpenEyes.asc")
    x_closed, y_closed, z_closed = read_signals("Resource\\ClosedEyes.asc")

    print("Analysing Signals")
    print("Pearson: ", pearsonr(x_open[3:105], y_open[3:105])[0])
    print("Spearman Correlation: ", spearmanr(x_open[3:105], y_open[3:105]).correlation)

    YSquare = function_square(x_open)
    YLine = function_line(x_open)
    
    print("Analysing First Function")
    print("Pearson: ", pearsonr(x_open[3:105], YSquare[3:105])[0])
    print("Spearman Correlation: ", spearmanr(x_open[3:105], YSquare[3:105]).correlation)

    print("\n Analysing Second Function")
    print("Pearson: ", pearsonr(x_open[3:105], YLine[3:105])[0])
    print("Spearman Correlation: ", spearmanr(x_open[3:105], YLine[3:105]).correlation)