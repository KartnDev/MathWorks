import time

import numpy as np
from scipy.stats import spearmanr, rankdata
import numba

from NeuralInterfaces.dataSignal2 import read_signals


def spearman(x_series: list, y_series: list):
    start_time = time.time()
    x_ranks = rankdata(x_series)
    y_ranks = rankdata(y_series)
    res = spearman_jit(x_ranks, y_ranks, len(x_ranks))
    return np.round(res, 2)


def spearman_jit(x_ranks, y_ranks, n):
    sum = np.linalg.norm(x_ranks - y_ranks)**2

    return 1 - (6 * sum) / (n ** 3 - n)


def matrix_correlation(cv):
    n = len(cv)

    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = spearman(cv[i], cv[j])

    return result


if __name__ == '__main__':
    x_open, y_open, z_open = read_signals("..\\Resource\\OpenEyes.asc")
    x_close, y_close, z_close = read_signals("..\\Resource\\ClosedEyes.asc")
    english = np.random.randn(1000000) * 100
    math = np.random.randn(1000000) * 100

    last_len = len(x_close)

    print(matrix_correlation([x_open[:last_len], y_open[:last_len], z_open[:last_len], x_close, y_close, z_close]))

