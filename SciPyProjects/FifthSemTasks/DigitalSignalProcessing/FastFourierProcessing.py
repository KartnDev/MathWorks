import time
from typing import Iterable
from numba import jit, prange
import numpy as np
from matplotlib import pyplot as plt


@jit(nopython=True)
def discrete_fourier_transform(x_real: Iterable):
    n_count = len(x_real)

    result_complex_array = np.zeros(n_count, dtype=np.float64) + 0j
    for k in range(0, n_count):
        for n in range(0, n_count):
            oscillate_coefficient = 2 * np.pi / n_count * k * n

            result_complex_array[k] += x_real[n] * (np.cos(oscillate_coefficient) - np.sin(oscillate_coefficient) * 1j)

    return result_complex_array


def midpoint_discrete_value(x_real):
    n_count = len(x_real)

    arranged = np.arange(n_count)
    k_count = arranged.reshape((n_count, 1))

    M = np.exp(-2j * np.pi * k_count * arranged / n_count)
    res = np.dot(M, x_real)
    return res


def fast_fourier_transform(x_real):
    x_real = np.asarray(x_real, dtype=float)
    n_count = len(x_real)

    if n_count % 2 > 0:
        raise ValueError("must be a power of 2")
    elif n_count <= 2:
        return midpoint_discrete_value(x_real)
    else:
        x_even = fast_fourier_transform(x_real[::2])
        x_odd = fast_fourier_transform(x_real[1::2])
        terms = np.exp(-2j * np.pi * np.arange(n_count) / n_count)
        return np.concatenate([x_even + terms[:int(n_count / 2)] * x_odd, x_even + terms[int(n_count / 2):] * x_odd])


def rand_time_and_exec(callback, x_sequence: Iterable):
    start = time.time()

    result = callback(x_sequence)
    time_exec = time.time() - start

    return time_exec, result


if __name__ == '__main__':
    x_set = np.random.random(1024 * 2)

    discrete_time, discrete = rand_time_and_exec(discrete_fourier_transform, x_set)
    fast_time, fast = rand_time_and_exec(fast_fourier_transform, x_set)
    numpy_time, numpy_val = rand_time_and_exec(np.fft.fft, x_set)

    print(f"Discrete Time: {discrete_time} IsSame: {np.allclose(numpy_val, discrete)}")
    print(f"Fast Time: {fast_time} IsSame: {np.allclose(numpy_val, fast)}")
    print(f"Numpy Time: {numpy_time} IsSame: {np.allclose(numpy_val, numpy_val)}")
