import numpy as np
import matplotlib.pyplot as plt

from NumMethods.NumMethodsTask4 import thomas_solver, build_tridiagonal, tridiagonal_matrix_solver


def solve_tridiagonal(size: int, h_array: np.array, y_vector: np.array):
    side_diagonal = np.array([h for h in h_array[1:-1]])
    mid_diagonal = np.array([2 * (h_array[i - 1] + h_array[i]) for i in range(1, size - 1)])

    lhs_matrix = build_tridiagonal(side_diagonal, mid_diagonal, side_diagonal)

    rhs_vector = np.array(
        [6 * ((y_vector[i] - y_vector[i - 1]) / h_array[i - 1] - (y_vector[i - 2] - y_vector[i - 1]) / h_array[i - 1])
         for i in range(2, size)])

    return tridiagonal_matrix_solver(lhs_matrix, rhs_vector)


def build_spline_coefficients(x_vector: np.array, y_vector: np.array):
    h_array = np.array([x_vector[i] - x_vector[i - 1] for i in range(1, len(x_vector))])  # from 1
    c_solve = solve_tridiagonal(len(x_vector), h_array, y_vector)

    d_solve = [c_solve[0] / h_array[0]]
    b_solve = []
    for i in range(0, len(x_vector) - 2):
        d_solve.append(c_solve[i] - c_solve[i - 1] / h_array[i])
    for i in range(0, len(x_vector) - 2):
        b_solve.append((y_vector[i] - y_vector[i - 1]) / h_array[i]
                       + h_array[i] * c_solve[i] / 2
                       + h_array[i] ** 2 * d_solve[i] / 6)

    return y_vector, np.array(b_solve), c_solve, np.array(d_solve)


def spline(a, b, c, d, x, y, x_i):
    return a + b * (x - x_i) + c / 2 * (x - x_i) ** 2 + d / 6 * (x - x_i) ** 3


def generate_spline(x, y):
    a, b, c, d = build_spline_coefficients(x, y)

    res = []

    for i in range(len(x) - 2):
        res.append(spline(a[i], b[i], c[i], d[i], x, y, x[i]))

    return np.array(res)


f = lambda x: x ** (x * np.cos(x))

a = 0
b = 3
res = []

x = np.linspace(a, b, 5)
y = f(x)
spline_val = generate_spline(x, y)
for i in range(3):
    plt.plot(x, spline_val[i])
plt.plot(x, f(x), '--')
plt.show()

