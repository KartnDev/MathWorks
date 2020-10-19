import time
from typing import Iterable
from numba import jit, prange
import numpy as np
from matplotlib import pyplot as plt


def discrete_fourier_transform(x_real: Iterable):
    n_count = len(x_real)

    result_complex_array = np.zeros(n_count, dtype=np.float64) + 0j
    for k in range(0, n_count):
        for n in range(0, n_count):
            oscillate_coefficient = 2 * np.pi / n_count * k * n

            result_complex_array[k] += x_real[n] * (np.cos(oscillate_coefficient) - np.sin(oscillate_coefficient) * 1j)

    return result_complex_array


def hlaf_dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return hlaf_dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N / 2)] * X_odd, X_even + terms[int(N / 2):] * X_odd])


def rand_time_and_exec(callback, x_sequence: Iterable):
    start = time.time()

    result = callback(x_sequence)
    time_exec = time.time() - start

    return time_exec, result


if __name__ == '__main__':
    x_set = np.random.random(512)

    discrete_time, discrete = rand_time_and_exec(discrete_fourier_transform, x_set)
    fast_time, fast = rand_time_and_exec(fft, x_set)
    numpy_time, numpy_val = rand_time_and_exec(np.fft.fft, x_set)

    print(f"Discrete Time: {discrete_time} IsSame: {np.allclose(numpy_val, discrete)}")
    print(f"Fast Time: {fast_time} IsSame: {np.allclose(numpy_val, fast)}")
    print(f"Numpy Time: {numpy_time} IsSame: {np.allclose(numpy_val, numpy_val)}")
