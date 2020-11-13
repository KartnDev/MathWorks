from NumMethods.NumericMethodsTask1 import euler_recount
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def full_compute(first_bound: float, second_bound: float, x_zero: int, y_zero: int, h_loss: float):
    n_steps = int((second_bound - first_bound) / h_loss)

    x_set = np.linspace(first_bound, second_bound, num=n_steps)
    x_set[0] = x_zero

    y_set = x_set.copy()
    y_set[0] = y_zero

    y_dif_set = euler_recount(x_set, y_set, n_steps, h_loss)
    x_set = np.linspace(first_bound, second_bound, num=n_steps)

    return y_dif_set, x_set


if __name__ == '__main__':
    a = 0
    b = 1
    A = 0
    B = 0
