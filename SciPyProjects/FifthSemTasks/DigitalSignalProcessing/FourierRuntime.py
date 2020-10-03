import numpy as np
from scipy.integrate import quad as integrate
from matplotlib import pyplot as plt


def __calculate_a_coefficient(foo, interval_start, interval_end, n):
    return integrate(lambda x: foo(x) * np.cos(n*x), interval_start, interval_end)[0]


def __calculate_b_coefficient(foo, interval_start, interval_end, n):
    return integrate(lambda x: foo(x) * np.sin(n*x), interval_start, interval_end)[0]


def fourier(x, foo, interval_start, interval_end, n_steps: int = 10):
    a_zero = 1/(2 * np.pi) * integrate(foo, interval_start, interval_end)[0]

    sum_val = 0

    for n in range(1, n_steps):
        a_at_iter = __calculate_a_coefficient(foo, interval_start, interval_end, n)
        b_at_iter = __calculate_b_coefficient(foo, interval_start, interval_end, n)

        sum_val += a_at_iter * np.cos(n*x) + b_at_iter * np.sin(n*x)

    return a_zero/2 + 1/np.pi * sum_val


def render_graphing_fourier_within(fourier_foo, inner_foo, interval_start, interval_end):
    x = np.linspace(interval_start, interval_end)

    y = fourier_foo(x, inner_foo, interval_start, interval_end, n_steps=3)
    plt.plot(x, y, color='r', label='steps=3')

    y = fourier_foo(x, inner_foo, interval_start, interval_end, n_steps=10)
    plt.plot(x, y, color='b', label='steps=10')

    y = fourier_foo(x, inner_foo, interval_start, interval_end, n_steps=50)
    plt.plot(x, y, color='g', label='steps=50')

    y = inner_foo(x)
    plt.plot(x, y, color='black', label='func')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    render_graphing_fourier_within(fourier, lambda x: x, -np.pi, np.pi)
    render_graphing_fourier_within(fourier, lambda x: x*x, -np.pi, np.pi)
    render_graphing_fourier_within(fourier, lambda x: np.sign(x), -np.pi, np.pi)