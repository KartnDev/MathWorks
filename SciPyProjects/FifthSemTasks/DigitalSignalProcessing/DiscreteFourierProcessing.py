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


def discrete_convolution(f_series: Iterable, g_series: Iterable):
    assert len(f_series) == len(g_series), "Different sizes!"
    n_count = len(f_series)
    result_conv = np.zeros(n_count, dtype=np.float64) + 0j

    for k in range(0, n_count):
        for n in range(0, n_count):
            result_conv[k] += f_series[n] * g_series[k - n]

    return result_conv


def task_part_one():
    rand_x = 100 * np.random.random_sample(33)
    assert_accuracy = 9

    print("\nDiscrete Fourier Transformation")
    print("===============================\n")

    my_transform = discrete_fourier_transform(rand_x)
    numpy_transform = np.fft.fft(rand_x)

    for my_value, numpy_value in zip(my_transform, numpy_transform):
        assert np.round(my_value, assert_accuracy) == np.round(numpy_value, assert_accuracy), "IS NOT EQUALS"
        print(f"My: {np.round(my_value, 2)} | Numpy: {np.round(numpy_value, 2)}")

    print("\nInverse Discrete Fourier Transformation")
    print("===============================\n")

    my_back_real = inverse_discrete_fourier_transform(my_transform)
    numpy_back_real = np.fft.ifft(numpy_transform)

    for my_value, numpy_value in zip(my_back_real, numpy_back_real):
        assert np.round(my_value, assert_accuracy) == np.round(numpy_value, assert_accuracy), "IS NOT EQUALS"
        print(f"My Inverse: {np.round(my_value, 2)} | Numpy inverse: {np.round(numpy_value, 2)}")


def task_part_two():
    n_const = 10
    rand_x = np.random.random_sample(n_const)
    my_transform = discrete_fourier_transform(rand_x)

    real_sum = np.sum(np.absolute(rand_x) ** 2)
    complex_sum = (1 / n_const) * np.sum(np.absolute(my_transform) ** 2)

    print(f"SumReal: {real_sum} | ComplexSum: {complex_sum}")

    rand_f = np.random.random_sample(n_const)
    rand_g = np.random.random_sample(n_const)

    rand_conv = discrete_convolution(rand_f, rand_g)

    transform_conv = discrete_fourier_transform(rand_conv)
    transform_f = discrete_fourier_transform(rand_f)
    transform_g = discrete_fourier_transform(rand_g)

    for ff, gg in zip(transform_conv, transform_f * transform_g):
        print(f"Conv: {np.round(ff, 2)} | Val: {np.round(gg, 2)}")

    print(f"ConvSum: {np.round(np.sum(np.absolute(transform_conv)), 2)} "
          f"| ValSum: {np.round(np.sum(np.absolute(transform_f * transform_g)), 2)}")


if __name__ == '__main__':
    task_part_two()
