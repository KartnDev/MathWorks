import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numba import jit, prange


# noinspection PyShadowingNames
class Spike(object):
    def __init__(self, waveform: np.ndarray, j_length: int, m_size: int, logging: bool = False):
        self.__logging = logging

        self._sum_first = 0
        self._sum_second = 0

        self._d_array = []
        self._s_array = []

        self._s_result_array = []

        self._m_size = m_size
        self._j_length = j_length
        self._waveform = waveform

        self._build_matrix()
        self._update_summaries()

    def _build_matrix(self):
        self._s_array.append(self._waveform)
        for j in range(self._j_length):

            end_value = self._m_size / np.power(2, j + 1)
            end_value = int(end_value)

            stored_values_prev = []
            stored_values_next = []

            for m in range(end_value):
                value_prev = self._s_array[j][2 * m] + self._s_array[j][2 * m + 1]
                value_next = self._s_array[j][2 * m] - self._s_array[j][2 * m + 1]
                stored_values_prev.append(value_prev / np.sqrt(2))
                stored_values_next.append(value_next / np.sqrt(2))

            self._s_array.append(stored_values_prev)
            self._d_array.append(stored_values_next)

    def _update_summaries(self):
        for j in range(self._j_length + 1):
            end_value = self._m_size / np.power(2, j + 1)
            end_value = int(end_value)

            for m in range(end_value):
                self._sum_second += np.power(self._d_array[j][m], 2)

        for m in range(self._m_size):
            self._sum_first += np.power(self._waveform[m], 2)

        if self.__logging:
            print(self._sum_second + np.power(self._s_array[j_length][0], 2))
            print(self._sum_first)

    @property
    def get_spike_array(self):

        for j in range(self._j_length - 1, -1, -1):
            end_value = self._m_size / np.power(2, j)
            end_value = int(end_value)
            stored_values = []
            for m in range(end_value):
                value = self._s_array[j + 1][int(m / 2)] + np.power(-1, m) * self._d_array[j][int(m / 2)]
                stored_values.append(value / np.sqrt(2))
            self._s_result_array = [stored_values] + self._s_result_array
        return self._s_result_array[0]


if __name__ == '__main__':
    j_length = 6
    m_size = pow(2, j_length)
    range_array = np.arange(m_size)
    wave_smooth = signal.sawtooth(2 * np.pi * 5 * range_array)

    spike = Spike(wave_smooth, j_length, m_size, True)
    spike_result = spike.get_spike_array

    plt.plot(range_array, wave_smooth)
    plt.show()
    plt.plot(range_array, spike_result)
    plt.show()
