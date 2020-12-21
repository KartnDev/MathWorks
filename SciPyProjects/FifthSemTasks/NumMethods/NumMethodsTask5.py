from typing import Iterable

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


def find_tri_diagonal_matrix(func_vector: Iterable, h: float, n_count: int):
    under_diag = np.zeros(n_count + 1)
    mid_diag = np.zeros(n_count + 1)
    upper_diag = np.zeros(n_count + 1)
    rhs_vector = np.zeros(n_count + 1)

    for inx in range(0, n_count - 1):
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
    under, mid, upper, rsh_vector = find_tri_diagonal_matrix(func_vector, h_step, n)

    M = thomas_solver_templated(under, mid, upper, rsh_vector)

    index = int((x - a_boundary) / h_step)
    t = (x - x_vector[index]) / h_step

    phi_phase_one = func_vector[index] * phi_one(t)
    phi_phase_two = func_vector[index + 1] * phi_two(t)
    phi_phase_three = M[index] * h_step * h_step * phi_three(t)
    phi_phase_four = M[index + 1] * h_step * h_step * phi_four(t)

    return phi_phase_one + phi_phase_two + phi_phase_three + phi_phase_four


def generate_spline(x, y):
    h = (x[0] - x[-1]) / len(x)

    return np.array([build_spline(x[i], 0, h, x, y) for i in range(0, len(x))])


import matplotlib.pyplot as plt
import numpy as np

f = lambda x: x ** (x * np.cos(x))

a = 0
b = 3
res = []

for n in [5, 10, 20]:
    x = np.linspace(a, b, n)
    y = f(x)
    res.append(generate_spline(x, y))

xx = np.arange(a, b, 0.01)
plt.plot(xx, f(xx), 'b--')
for i, n in enumerate([5, 10, 20]):
    plt.plot(np.linspace(a, b, n), res[i])
plt.show()
