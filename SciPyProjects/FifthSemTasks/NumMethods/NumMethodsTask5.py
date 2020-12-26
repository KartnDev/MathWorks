from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np

from NumMethods.NumMethodsTask4 import tridiagonal_matrix_solver, build_tridiagonal


def get_tridiagonal(h_array: Iterable) -> Iterable:
    size: int = len(h_array) + 1
    side_diagonal = np.array([h for h in h_array[1:-1]])
    mid_diagonal = np.array([2 * (h_array[i - 1] + h_array[i]) for i in range(1, size - 1)])

    return build_tridiagonal(side_diagonal, mid_diagonal, side_diagonal)


def calculate_phi(y_vector: Iterable, h_vector: Iterable) -> Iterable:
    length_array = len(y_vector)
    phi_rhs_vector = []

    for i in range(2, length_array):
        val = (y_vector[i] - y_vector[i - 1]) / h_vector[i - 1]
        val2 = (y_vector[i - 1] - y_vector[i - 2]) / h_vector[i - 2]
        phi_rhs_vector.append(6 * (val - val2))

    return np.array(phi_rhs_vector)


def generate_d_coefficients(c_coefficient: Iterable, h_array: Iterable) -> Iterable:
    d_vector_coefficients = [c_coefficient[0] / h_array[0]]
    for i in range(1, len(c_coefficient)):
        d_vector_coefficients.append((c_coefficient[i] - c_coefficient[i - 1]) / h_array[i])

    return np.array(d_vector_coefficients)


def generate_b_coefficients(y_vector: Iterable, h_array: Iterable, c_coefficients: Iterable, d_coefficients: Iterable) :
    b_vector_coefficients = []

    for i in range(1, len(y_vector)):
        val = (y_vector[i] - y_vector[i - 1]) / h_array[i - 1] \
              + (h_array[i - 1] / 2) * c_coefficients[i - 1] \
              - np.power(h_array[i - 1], 2) * d_coefficients[i - 1] / 6
        b_vector_coefficients.append(val)

    return np.array(b_vector_coefficients)


def generate_spline(X, Y) -> Callable:
    assert len(X) == len(Y), "x and y have different sizes!"

    h_array = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)])

    rhs_tridiagonal_matrix = get_tridiagonal(h_array)
    lhs_phi_vector = calculate_phi(Y, h_array)

    a_coefficients = Y
    c_coefficients = tridiagonal_matrix_solver(rhs_tridiagonal_matrix, lhs_phi_vector)
    c_coefficients = np.append(c_coefficients, 0)
    d_coefficients = generate_d_coefficients(c_coefficients, h_array)
    b_coefficients = generate_b_coefficients(Y, h_array, c_coefficients, d_coefficients)

    def spline_template(value_array) -> Iterable:
        result = []
        value_array = np.array(value_array)
        for x in value_array:
            index = np.argwhere(X >= x)
            if len(index) == 0:
                continue
            index = index[0, 0]
            if index == 0:
                temp = Y[index]
            else:
                temp = a_coefficients[index] +\
                       b_coefficients[index - 1] * (x - X[index]) +\
                       c_coefficients[index - 1] * (x - X[index])**2/2 +\
                       d_coefficients[index - 1] * (x - X[index])**3/6
            result.append(temp)
        return np.array(result)

    return spline_template


def func(x):
    return x ** (x * np.cos(x))


if __name__ == '__main__':
    a = 0
    b = 3
    res = []

    xx = np.arange(a, b, 0.01)
    for n in [5, 10, 20]:
        x = np.linspace(a, b, n)
        y = func(x)
        spline = generate_spline(x, y)
        res.append(spline(xx))

    plt.plot(xx, func(xx), 'b--', xx, res[0], 'r', xx, res[1], 'g', xx, res[2], 'y')
    plt.show()
