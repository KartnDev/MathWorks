from typing import Iterable
import numpy as np


def discrete_fourier_transform(x_real: Iterable):
    n_count = len(x_real)

    result_complex_array = np.zeros(n_count, dtype=np.float64) + 0j
    for k in range(0, n_count):
        for n in range(0, n_count):
            oscillate_coefficient = 2 * np.pi / n_count * k * n

            result_complex_array[k] += x_real[n] * (np.cos(oscillate_coefficient) - np.sin(oscillate_coefficient) * 1j)

    return result_complex_array


def inverse_discrete_fourier_transform(x_imagine: Iterable):
    n_count = len(x_imagine)
    result_real_array = np.zeros(n_count, dtype=np.float64) + 0j
    for n in range(0, n_count):
        for k in range(0, n_count):
            result_real_array[n] += x_imagine[k] * np.exp(2j * np.pi * k * n / n_count)

        result_real_array[n] /= n_count  # Normalize

    return result_real_array


if __name__ == '__main__':
    rand_x = 100 * np.random.random_sample(33)
    assert_accuracy = 8

    print("\nDiscrete Fourier Transformation")
    print("===============================\n")

    my_transform = discrete_fourier_transform(rand_x)
    numpy_transform = np.fft.fft(rand_x)

    for my_value, numpy_value in zip(my_transform, numpy_transform):
        assert round(my_value, assert_accuracy) == round(numpy_value, assert_accuracy), "IS NOT EQUALS"
        print(f"My: {round(my_value, 2)} | Numpy: {round(numpy_value, 2)}")

    print("\nDiscrete Fourier Transformation")
    print("===============================\n")

    my_back_real = inverse_discrete_fourier_transform(my_transform)
    numpy_back_real = np.fft.ifft(numpy_transform)

    for my_value, numpy_value in zip(my_back_real, numpy_back_real):
        assert round(my_value, assert_accuracy) == round(numpy_value, assert_accuracy), "IS NOT EQUALS"
        print(f"My Inverse: {round(my_value, 2)} | Numpy inverse: {round(numpy_value, 2)}")