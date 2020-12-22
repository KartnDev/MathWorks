from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np


def thomas_solver_templated(under_diag: Iterable, mid_diag: Iterable, upper_diag: Iterable, rsh_vector: Iterable):
    size = len(under_diag)
    mid_diag[0] = mid_diag[0] / under_diag[0]
    rsh_vector[0] = rsh_vector[0] / under_diag[0]

    for i in range(0, size - 1):
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


def phi_one(x_value: float):
    return 1 - x_value


def phi_two(x_value: float):
    return x_value


def phi_three(x_value: float):
    return (-1 / 6) * x_value * (x_value - 1) * (x_value - 2)


def phi_four(x_value: float):
    return (1 / 6) * x_value * (x_value - 1) * (x_value + 1)


def build_spline(x: float, a_boundary: float, h_step: float, x_vector: Iterable, func_vector: Iterable):
    under, mid, upper, rsh_vector = find_tri_diagonal_matrix(func_vector, n)

    M = thomas_solver_templated(under, mid, upper, rsh_vector)

    index = int((x - a_boundary) / h_step)
    t = (x - x_vector[index]) / h_step

    phi_phase_one = func_vector[index] * phi_one(t)
    phi_phase_two = func_vector[index] * phi_two(t)
    phi_phase_three = M[index] * h_step * h_step * phi_three(t)
    phi_phase_four = M[index] * h_step * h_step * phi_four(t)

    return phi_phase_one + phi_phase_two + phi_phase_three + phi_phase_four


def generate_spline(x, y):
    assert len(x) == len(y), 'different sizes of input vectors'

    result = []
    h_step = (x[-1] - x[0]) / (len(x) - 1)
    for i in range(0, len(x)):
        result.append(build_spline(x[i], 0, h_step, x, y))

    return result


def func(x):
    return x ** (x * np.cos(x))


if __name__ == '__main__':

    a_bound = 0
    b_bound = 3
    for n in [5, 10, 20]:
        h = (b_bound - a_bound) / n
        x_range = np.arange(a_bound, b_bound + h, h)
        func_val = func(x_range)

        res = generate_spline(x_range, func_val)

        plt.plot(x_range, res, linewidth=3)
        plt.plot(x_range, func_val, 'o', linewidth=1)
        xx = np.arange(a_bound, b_bound + h, 0.01)
        plt.title("$ y = x^{x \cdot cosx} $ \t" + f"n_points: {n}")
        plt.plot(xx, func(xx))
        plt.show()
