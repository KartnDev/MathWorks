from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np


def thomas_solver_templated(under_diag: Iterable, mid_diag: Iterable, upper_diag: Iterable, rsh_vector: Iterable):
    size = len(under_diag)
    mid_diag[0] = mid_diag[0] / under_diag[0]
    rsh_vector[0] = rsh_vector[0] / under_diag[0]

    for i in range(1, size):
        v = mid_diag[i] / (-upper_diag[i] * mid_diag[i - 1] + under_diag[i])
        u = (rsh_vector[i] - upper_diag[i] * rsh_vector[i - 1]) / (-upper_diag[i] * mid_diag[i - 1] + under_diag[i])

        mid_diag[i] = v
        rsh_vector[i] = u

    Y = np.zeros(size)
    Y[size - 1] = rsh_vector[size - 1]
    for i in range(size - 2, -1, -1):
        Y[i] = -mid_diag[i] * Y[i + 1] + rsh_vector[i]
    return Y


def find_tri_diagonal_matrix(func_vector: Iterable, n_count: int):
    under_diag = np.zeros(n_count + 1)
    mid_diag = np.zeros(n_count + 1)
    upper_diag = np.zeros(n_count + 1)
    rhs_vector = np.zeros(n_count + 1)

    for inx in range(0, n_count):
        under_diag[inx] = 4
        mid_diag[inx] = 1
        upper_diag[inx] = 1
        rhs_vector[inx] = (6 * func_vector[inx - 1] + func_vector[inx + 1] - 2 * func_vector[inx]) / (h * h)

    under_diag[0] = 1

    rhs_vector[0] = (func_vector[0] + func_vector[2] - 2 * func_vector[1]) / (h * h)
    rhs_vector[n_count] = (func_vector[n_count - 3] + func_vector[n_count - 1] - 2 * func_vector[n_count - 2]) / (h * h)

    under_diag[n_count] = 1

    return under_diag, mid_diag, upper_diag, rhs_vector


def phi_one(t: float):
    return 1 - t


def phi_two(t: float):
    return t


def phi_three(t: float):
    return (-1 / 6) * t * (t - 1) * (t - 2)


def phi_four(t: float):
    return (1 / 6) * t * (t - 1) * (t + 1)


def build_spline(x: float, a_boundary: float, h_step: float, x_vector: Iterable, func_vector: Iterable):
    under, mid, upper, rsh_vector = find_tri_diagonal_matrix(f, n)

    M = thomas_solver_templated(under, mid, upper, rsh_vector)

    index = int((x - a_boundary) / h_step)
    t = (x - x_vector[index]) / h_step

    phi_phase_one = func_vector[index] * phi_one(t)
    phi_phase_two = func_vector[index + 1] * phi_two(t)
    phi_phase_three = M[index] * h_step * h_step * phi_three(t)
    phi_phase_four = M[index + 1] * h_step * h_step * phi_four(t)

    return phi_phase_one + phi_phase_two + phi_phase_three + phi_phase_four


def func(x):
    return x * np.sin(-x) * np.exp(-x) * np.tan(-x * 1.1)


if __name__ == '__main__':
    n = 25
    a = 0
    b = 10
    h = (b - a) / n
    X = np.arange(a, b + h, h)
    f = func(X)

    SRes = []

    for i in range(0, n):
        SRes.append(build_spline(X[i], 0, h, X, f))

    SRes = np.array(SRes)

    plt.plot(X[:n], SRes, linewidth=3)
    plt.plot(X[:n], f[:n], 'o', linewidth=1)
    plt.show()
