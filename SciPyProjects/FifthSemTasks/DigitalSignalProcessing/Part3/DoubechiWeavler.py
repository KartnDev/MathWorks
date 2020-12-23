import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
import math


def get_h(value, h_array):
    if value < -10:
        return 0
    elif value > 19:
        return 0
    else:
        return h_array[value + 10]


def generate_s_d(waveform, j_length, m_size, h_array):
    s_array = []
    d_array = []
    s_array.append(waveform)

    for j in range(j_length):
        end_value = m_size / np.power(2, j + 1)
        stored_prev = []
        stored_next = []

        for i in range(int(end_value)):
            summary_ = 0
            summary_with_d = 0

            for m in range(len(s_array[j])):
                summary_ += get_h(m - 2 * i, h_array) * s_array[j][m]
                summary_with_d += np.power(-1, m) * get_h(1 - m + 2 * i, h_array) * s_array[j][m]

            stored_next.append(summary_with_d)
            stored_prev.append(summary_)

        s_array.append(stored_prev)
        d_array.append(stored_next)

    return s_array, d_array


def update_sum(waveform, j_length, m_size, s_array, d_array):
    sum_first = 0
    sum_second = 0
    for j in range(j_length + 1):
        end_value = m_size / np.power(2, j + 1)
        end_value = int(end_value)
        for m in range(end_value):
            sum_first += math.pow(d_array[j][m], 2)
    for m in range(m_size):
        sum_second += math.pow(waveform[m], 2)
    print(sum_first + np.power(s_array[j_length][0], 2))
    print(sum_second)

    return sum_first, sum_second


def get_final_sum(h_array, j_length, m_size, s_array, d_array):
    s_result_value = [s_array[5]]
    for j in range(j_length - 1, -1, -1):
        end_value = int(m_size / np.power(2, j))
        stored_full_value = []
        for m in range(end_value):
            summary = 0
            for i in range(len(s_array[j + 1])):
                summary += s_array[j + 1][i] * get_h(m - 2*i, h_array) \
                           + np.power(-1, m) * d_array[j][i] * get_h(1 - m + 2*i, h_array)
            stored_full_value.append(summary)
        s_result_value.append(stored_full_value)
    return s_result_value


if __name__ == '__main__':
    n_const = 15
    n_value = 2 * n_const
    h_array = np.array([-0.0001499638, 0.0002535612, 0.0015402457, -0.0029411108, -0.0071637819, 0.0165520664, 0.0199178043,
                        -0.0649972628, -0.0368000736, 0.2980923235, 0.5475054294, 0.3097068490, -0.0438660508, -0.0746522389,
                        0.0291958795, 0.0231107770, -0.0139738879, -0.0064800900, 0.0047830014, 0.0017206547, -0.0011758222,
                        -0.0004512270, 0.0002137298, 0.00009937776, -0.0000292321, -0.0000150720, 0.0000026408, 0.0000014593,
                        -0.0000001184, -0.0000000673])

    h_array = h_array * np.sqrt(2)
    j_length = 5
    m_size = pow(2, j_length)
    range_array = np.arange(m_size)
    waveform_smooth = signal.sawtooth(2 * np.pi * 5 * range_array)

    s_array, d_array = generate_s_d(waveform_smooth, j_length, m_size, h_array)

    print(len(s_array[0]), len(s_array[1]), len(s_array[2]), len(s_array[3]), len(s_array[4]))
    sum_first, sum_second = update_sum(waveform_smooth, j_length, m_size, s_array, d_array)

    result_spiked = get_final_sum(h_array, j_length, m_size, s_array, d_array)
    print(result_spiked[5])

    plt.plot(range_array, waveform_smooth)
    plt.plot(range_array, result_spiked[5])
    plt.show()
