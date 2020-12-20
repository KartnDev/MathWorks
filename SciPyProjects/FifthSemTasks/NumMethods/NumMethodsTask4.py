from typing import Iterable
import numpy as np


def build_tridiagonal(lower: Iterable, mid: Iterable, upper: Iterable):
    return np.diag(lower, -1) + np.diag(mid, 0) + np.diag(upper, 1)


def is_tridiagonal(matrix):
    under = np.diagonal(matrix, -1)
    mid = np.diagonal(matrix, 0)
    upper = np.diagonal(matrix, 1)

    estimated_zeros = build_tridiagonal(under, mid, upper) - matrix

    return np.all(np.isclose(estimated_zeros, np.zeros(matrix.shape)))


def thomas_solver(under_main_diag: Iterable, main_diag: Iterable, upper_main_diag: Iterable, vector: Iterable):
    equation_count = len(vector)

    ac, bc, cc, dc = map(lambda item: np.array(item, dtype=np.float64), (under_main_diag,
                                                                         main_diag,
                                                                         upper_main_diag,
                                                                         vector))
    for i in range(1, equation_count):
        mc = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - mc * cc[i - 1]
        dc[i] = dc[i] - mc * dc[i - 1]

    x_temp = bc
    x_temp[-1] = dc[-1] / bc[-1]

    for il in range(equation_count - 2, -1, -1):
        x_temp[il] = (dc[il] - cc[il] * x_temp[il + 1]) / bc[il]

    return x_temp


def tridiagonal_matrix_solver(lsh_matrix: np.ndarray, rsh_vector: Iterable):
    if is_tridiagonal(lsh_matrix):
        return thomas_solver(np.diagonal(lsh_matrix, -1),
                             np.diagonal(lsh_matrix, 0),
                             np.diagonal(lsh_matrix, 1),
                             rsh_vector)
    else:
        raise ValueError("Argument lsh_matrix wasn't tridiagonal")


A = np.array([
    [1, 4, 0, 0, 0],
    [1, 2, 3, 0, 0],
    [0, 1, 2, 3, 0],
    [0, 0, 1, 2, 3],
    [0, 0, 0, 5, 2]
])

B = np.array([1, 2, 3, 4, 5])
res_x = tridiagonal_matrix_solver(A, B)

print('my solver,    X =', res_x)
print('Numpy solver, X =', np.linalg.solve(A, B))

print(all(np.isclose(res_x, np.linalg.solve(A, B))))
print(all(np.isclose(A.dot(res_x), B)))
