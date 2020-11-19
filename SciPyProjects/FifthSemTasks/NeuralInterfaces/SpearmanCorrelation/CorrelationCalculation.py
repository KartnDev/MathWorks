import time

import numpy as np
from scipy.stats import spearmanr, rankdata

from NeuralInterfaces.dataSignal2 import read_signals

def full_ranks(x: list):
    res = {i: item for i, item in enumerate(x)}
    return {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}

def spearman(x_series: list, y_series: list):
    start_time = time.time()
    x_ranks = rankdata(x_series)
    y_ranks = rankdata(y_series)
    res = spearman_jit(x_ranks, y_ranks, len(x_ranks))
    return res


def spearman_jit(x_ranks, y_ranks, n):
    sum = 0.0
    for i in range(n):
        sum += (x_ranks[i] - y_ranks[i]) ** 2

    return 1 - (6 * sum) / (n ** 3 - n)


def matrix_correlation(cv):
    n = len(cv)

    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = spearmanr(cv[i], cv[j])[0]

    return result


if __name__ == '__main__':
    # x = np.array([i/1000 for i in range(256 * 256)])
    # y = np.array([(-np.sin(i) * i + i) for i in range(256 * 256)])
    #
    # sum = np.sum((x - x.mean()) * (y - y.mean()))
    #
    # error_x = np.sum((x - x.mean()) * (x - x.mean()))
    # error_y = np.sum((y - y.mean()) * (y - y.mean()))
    #
    # print(sum / np.sqrt(error_x * error_y))
    val = read_signals("C:\\Users\\Dmitry\\Desktop\\Newfolder\\NewSigCopy4(4).txt", 0, 1000)
    for l in val:
        print(set([x for x in l if l.count(x) > 1]))

    for i in range(0, 9):
        val = read_signals("C:\\Users\\Dmitry\\Desktop\\Newfolder\\NewSigCopy4(4).txt", i * 500, (i * 500) + 1000)

        print(np.round(matrix_correlation(val), 6))
