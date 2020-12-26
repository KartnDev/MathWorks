import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt


# noinspection DuplicatedCode
class DoubechiWeavler(object):
    def __init__(self, h_array, waveform, j_length, m_size, logging: bool = False):
        self.__logging = logging
        self._m_size = m_size
        self._j_length = j_length
        self._waveform = waveform
        self._h_array = h_array

        self._s_array = []
        self._d_array = []

        self._sum_first = 0
        self._sum_second = 0

        self._generate_arrays()
        self._update_sums()

    def get_h(self, value):
        if value < -10:
            return 0
        elif value > 19:
            return 0
        else:
            return self._h_array[value + 10]

    def _generate_arrays(self):

        self._s_array.append(self._waveform)

        for j in range(self._j_length):
            end_value = self._m_size / np.power(2, j + 1)
            stored_prev = []
            stored_next = []

            for i in range(int(end_value)):
                summary_ = 0
                summary_with_d = 0

                for m in range(len(self._s_array[j])):
                    summary_ += self.get_h(m - 2 * i) * self._s_array[j][m]
                    summary_with_d += np.power(-1, m) * self.get_h(1 - m + 2 * i) * self._s_array[j][m]

                stored_next.append(summary_with_d)
                stored_prev.append(summary_)

            self._s_array.append(stored_prev)
            self._d_array.append(stored_next)

        if self.__logging:
            print("Lengths:",
                  len(self._s_array[0]),
                  len(self._s_array[1]),
                  len(self._s_array[2]),
                  len(self._s_array[3]),
                  len(self._s_array[4]))

    def _update_sums(self):
        for j in range(self._j_length + 1):
            end_value = self._m_size / np.power(2, j + 1)
            end_value = int(end_value)
            for m in range(end_value):
                self._sum_first += np.power(self._d_array[j][m], 2)
        for m in range(self._m_size):
            self._sum_second += np.power(self._waveform[m], 2)
        if self.__logging:
            print("First sum:", self._sum_first + np.power(self._s_array[self._j_length][0], 2))
            print("Second sum:", self._sum_second)

    @property
    def get_final_sum(self):
        s_result_value = [self._s_array[5]]
        for j in range(self._j_length - 1, -1, -1):
            end_value = int(self._m_size / np.power(2, j))
            stored_full_value = []
            for m in range(end_value):
                summary = 0
                for i in range(len(self._s_array[j + 1])):
                    summary += self._s_array[j + 1][i] * self.get_h(m - 2 * i) \
                               + np.power(-1, m) * self._d_array[j][i] * self.get_h(1 - m + 2 * i)
                stored_full_value.append(summary)
            s_result_value.append(stored_full_value)
        print(np.array(s_result_value[5]))
        return np.array(s_result_value[5])


if __name__ == '__main__':
    n_const = 15
    n_value = 2 * n_const

    # noinspection DuplicatedCode
    h_vector_const = np.array(
        [-0.0001499638, 0.0002535612, 0.0015402457, -0.0029411108, -0.0071637819, 0.0165520664, 0.0199178043,
         -0.0649972628, -0.0368000736, 0.2980923235, 0.5475054294, 0.3097068490, -0.0438660508, -0.0746522389,
         0.0291958795, 0.0231107770, -0.0139738879, -0.0064800900, 0.0047830014, 0.0017206547, -0.0011758222,
         -0.0004512270, 0.0002137298, 0.00009937776, -0.0000292321, -0.0000150720, 0.0000026408, 0.0000014593,
         -0.0000001184, -0.0000000673]) * np.sqrt(2)

    j_len = 5
    m_sized = np.power(2, j_len)
    range_array = np.arange(m_sized)
    waveform_smooth = signal.sawtooth(2 * np.pi * 5 * range_array)

    wave = DoubechiWeavler(h_vector_const, waveform_smooth, j_len, m_sized, True)

    result_spiked = wave.get_final_sum


    plt.plot(range_array, waveform_smooth)
    plt.plot(range_array, result_spiked)
    plt.show()
