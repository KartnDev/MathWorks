from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def b_spline_kernel(x_val: float, n: int):
    if n == 1:
        if abs(x_val) <= 0.5:
            return 1
        else:
            return 0
    elif n == 2:
        if abs(x_val) < 1:
            return 1 - abs(x_val)
        else:
            return 0
    else:
        if abs(x_val) <= n / 2:
            lhs = (n + 2 * x_val) / (2 * n - 2)
            rhs = (n - 2 * x_val) / (2 * n - 2)
            return lhs * b_spline_kernel(x_val + 0.5, n - 1) + rhs * b_spline_kernel(x_val - 0.5, n - 1)
        else:
            return 0


def build_basis_spline(x_vector: Iterable):
    res = np.zeros(x_vector.shape)
    for i in range(len(x_vector)):
        res[i] = b_spline_kernel(x_vector[i], 10)

    return res


if __name__ == '__main__':
    N = 10
    h = 0.1
    X = np.arange(0, (N / 2) + h, h)

    result = build_basis_spline(np.exp(X) * np.sin(X))

    x_neg = ((-1 * X)[::-1])
    X = np.concatenate((x_neg, X[1:]))
    result = np.concatenate((result[::-1], result[1:]))

    plt.plot(X, result)
    plt.show()
