import numpy as np
import matplotlib.pyplot as plt

mean_func = lambda x, y: np.exp(x) * np.cos(2 * y ** 2)


def runge_kutta_core(x0, y0, n_steps: int, h_loss: float):
    y = y0
    for i in range(1, n_steps + 1):
        k1 = h_loss * mean_func(x0, y)
        k2 = h_loss * mean_func(x0 + 0.5 * h_loss, y + 0.5 * k1)
        k3 = h_loss * mean_func(x0 + 0.5 * h_loss, y + 0.5 * k2)
        k4 = h_loss * mean_func(x0 + h_loss, y + k3)

        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x0 = x0 + h_loss
    return y


def adams_core(x0, y0, n_steps: int, h_loss: float):
    x = [i for i in range(n_steps)]
    y = [i for i in range(n_steps)]
    f = [i for i in range(n_steps)]

    x[0] = x0
    y[0] = y0

    f[0] = mean_func(x[0], y[0])

    x[1] = x[0] + h_loss
    y[1] = y[0] + h_loss * f[0]
    f[1] = mean_func(x[1], y[1])

    x[2] = x[1] + h_loss
    y[2] = y[1] + h_loss * (3 / 2 * f[1] - 1 / 2 * f[0])
    f[2] = mean_func(x[2], y[2])

    x[3] = x[2] + h_loss
    y[3] = y[2] + h_loss / 12 * (23 * f[2] - 16 * f[1] + 5 * f[0])
    f[3] = mean_func(x[3], y[3])

    for i in range(3, n_steps - 1):
        y[i + 1] = y[i] + h_loss / 24 * (55 * f[i] - 59 * f[i - 1] + 37 * f[i - 2] - 9 * f[i - 3])
        x[i + 1] = x[i] + h_loss
        f[i + 1] = mean_func(x[i + 1], y[i + 1])

    return y[n_steps - 1]


def adams(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps):
        y[i] = adams_core(x[i], y[i], n_steps, h_loss)

    return y


def runge_kutta(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps):
        y[i] = runge_kutta_core(x[i], y[i], n_steps, h_loss)

    return y


def init_render(foo, first_bound: float, second_bound: float, x_zero: int, y_zero: int, h_losses: [float], title: str):
    for h_loss in h_losses:
        n_steps = int((second_bound - first_bound) / h_loss)

        x_set = np.linspace(first_bound, second_bound, num=n_steps)
        x_set[0] = x_zero

        y_set = x_set
        y_set[0] = y_zero

        y_dif_set = foo(x_set, y_set, n_steps, h_loss)
        x_set = np.linspace(first_bound, second_bound, num=n_steps)

        plt.plot(x_set, y_dif_set, label=f'Accuracy: {h_loss}')
        plt.legend()
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    # bounds of computing
    a = 0.0
    b = 1.0

    # entry conditions
    x_0 = 0
    y_0 = 1

    # maximum losses of accuracy
    h_list = [0.015, 0.010, 0.005]

    init_render(adams, a, b, x_0, y_0, h_list, 'Adams Diff Solver')
    init_render(runge_kutta, a, b, x_0, y_0, h_list, 'Runge Kutta Diff Solver')
