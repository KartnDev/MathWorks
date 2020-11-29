from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from DigitalSignalProcessing.BSpline import b_spline_kernel


def fundamental(x_val: float, p_count: int, n_count: int):
    phi_mh_step = np.zeros(n + 1)
    p_half = int(p / 2)
    d_temp = np.zeros(n + 1)
    basis_points = np.zeros(p_half + 1)

    for j in range(p_half + 1):
        basis_points[j] = b_spline_kernel(j, p)

    for m in range(0, n + 1):
        value = basis_points[0]
        sum_b = 0
        for j in range(1, p_half + 1):
            sum_b += basis_points[j] * np.cos(j * m * h)
        phi_mh_step[m] = value + 2 * sum_b
    d_multiply_h_step = 1 / phi_mh_step

    for k in range(0, n + 1):
        value = d_multiply_h_step[0]
        sum_b = 0
        for m in range(1, n + 1):
            sum_b += d_multiply_h_step[m] * np.cos(h * k * m)
        d_temp[k] = (value + 2 * sum_b) / N

    sum_res = 0
    for k in range(-n_count, n_count):
        sum_res += d_temp[np.abs(k)] * b_spline_kernel(x_val - k, p_count)
    return sum_res


if __name__ == '__main__':
    p = 10
    n = 100
    N = 2 * n + 1
    h = 2 * np.pi / N
    X = np.arange(-10, 10, 0.1)
    Y = np.zeros(len(X))

    for i in range(len(X)):
        Y[i] = fundamental(X[i], p, n)

    plt.plot(X, Y, color="red", label="N = 10")
    plt.show()
