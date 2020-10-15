import numpy as np
import matplotlib.pyplot as plt

mean_func = lambda x, y: np.exp(x) * np.cos(2 * y ** 2)


def modified_euler(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps - 1):
        pre_counted = y[i] + h_loss / 2 * mean_func(x[i], y[i])
        y[i + 1] = y[i] + h_loss * mean_func(x[i] + h_loss / 2, pre_counted)
    return y


def euler_recount(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps - 1):
        pre_counted = y[i] + h_loss * mean_func(x[i], y[i])
        y[i + 1] = y[i] + h_loss / 2 * (mean_func(x[i], y[i]) + mean_func(x[i] + h_loss, pre_counted))
    return y


def euler_right_diff(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps - 1):
        y[i + 1] = y[i] + h_loss * mean_func(x[i], y[i])
    return y


def euler_left_diff(x, y, n_steps: int, h_loss: float):
    for i in range(n_steps):
        y[i] = y[i - 1] + h_loss * mean_func(x[i-1], y[i-1])
    return y


def euler_central_diff(x, y, n_steps: int, h_loss: float):
    x_0 = x[0]
    y_0 = y[0]
    y[1] = y[0] + h_loss * mean_func(x[0], y[0])
    for i in range(1, n_steps - 1):
        y[i + 1] = y[i - 1] + 2 * h_loss * mean_func(x[i-1], y[i-1])
    return y



def init_render(foo, first_bound: float, second_bound: float, x_zero: int, y_zero: int, h_losses: [float], title: str):
    for h_loss in h_losses:
        n_steps = int((second_bound - first_bound) / h_loss)

        x_set = np.linspace(first_bound, second_bound, num=n_steps)
        x_set[0] = x_zero

        y_set = x_set.copy()
        y_set[0] = y_zero

        y_dif_set = foo(x_set, y_set, n_steps, h_loss)
        x_set = np.linspace(first_bound, second_bound, num=n_steps)

        plt.plot(x_set, y_dif_set, label=f'Accuracy: {h_loss}')
        plt.legend()
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    a = 0.0
    b = 1.0
    x_0 = 0
    y_0 = 1
    h_list = [0.015, 0.010, 0.005]

    init_render(modified_euler, a, b, x_0, y_0, h_list, 'modified_euler')
    init_render(euler_recount, a, b, x_0, y_0, h_list, 'euler_recount')
    init_render(euler_central_diff, a, b, x_0, y_0, h_list, 'euler_central_diff')
    init_render(euler_left_diff, a, b, x_0, y_0, h_list, 'euler_left_diff')
    init_render(euler_right_diff, a, b, x_0, y_0, h_list, 'euler_right_diff')
