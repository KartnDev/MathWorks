import numpy as np
from matplotlib import pyplot as plt



def fourier_first(x: float, steps_n: int = 10):
    sums = 0

    for i in range(1, steps_n):
        sums += ((-1) ** (i + 1)) * np.sin(i*x) / i

    return 2 - sums


def fourier_second(x: float, steps_n: int = 10):
    sums = 0

    for i in range(1, steps_n):
        sums += ((-1) ** i) * np.cos(i*x) / (i ** 2)

    return (np.pi ** 2)/3 + (4 * sums)


def fourier_third(x: float, steps_n: int = 10):
    sums = 0

    for i in range(1, steps_n):
        raw = 2*i - 1
        sums += np.sin(raw * x) / raw

    return 4 / np.pi * sums


def render_graphing_fourier(foo, interval_start, interval_end):
    x = np.linspace(interval_start, interval_end)

    y = foo(x, steps_n=3)
    plt.plot(x, y, color='r', label='steps=3')

    y = foo(x, steps_n=10)
    plt.plot(x, y, color='b', label='steps=10')

    y = foo(x, steps_n=50)
    plt.plot(x, y, color='g', label='steps=50')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    render_graphing_fourier(fourier_first, -np.pi, np.pi)
    render_graphing_fourier(fourier_second, -np.pi, np.pi)
    render_graphing_fourier(fourier_third, -np.pi, np.pi)
