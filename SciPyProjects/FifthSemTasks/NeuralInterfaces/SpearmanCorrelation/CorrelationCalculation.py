import numpy as np
from scipy.stats import spearmanr, rankdata


from NeuralInterfaces.dataSignal2 import read_signals


def spearman(x_series: list, y_series: list):

    x_ranks = rankdata(x_series)
    y_ranks = rankdata(y_series)

    d_squares_sum = 0.0

    n = len(x_series)

    for i in range(n):
        d_squares_sum += ((x_ranks[i]) - (y_ranks[i])) ** 2

    return 1 - (6 * d_squares_sum) / (n ** 3 - n)


if __name__ == '__main__':
    x_open, y_open, z_open = read_signals("..\\Resource\\OpenEyes.asc")
    x_close, y_close, z_close = read_signals("..\\Resource\\ClosedEyes.asc")
    english = np.random.randn(10000000) * 100
    math = np.random.randn(10000000) * 100

    print(spearmanr(x_open[:len(x_close)], x_close))
    print(spearman(x_open[:len(x_close)], x_close))