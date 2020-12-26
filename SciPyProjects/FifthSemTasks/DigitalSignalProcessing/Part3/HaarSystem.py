import os
import math
from typing import Iterable

import matplotlib.pyplot as plt
from numba import jit, prange


def read_file(path_file, number_signal=1, J=4):
    signal = []
    with open(path_file) as file:
        for elem in file.read().split('\n'):
            values_list = elem.split(' ')
            if len(values_list) == 3:
                signal.append(int(values_list[number_signal - 1]))
    if 2 ** J > len(signal):
        return False
    return signal[:2 ** J]


class HaarSystem(object):

    def __init__(self, M, signal: Iterable = None):
        self.M = M
        self.signal = signal
        self.J = int(math.ceil(math.log2(self.M)))
        self.M = 2 ** self.J

        self.accuracy = 'Not calculated'

        self.S = [[0 for _ in range(self.M)] for _ in range(self.J + 1)]
        self.d = [[0 for _ in range(self.M)] for _ in range(self.J + 1)]

    def from_file_signal(self, path_file, num):
        self.signal = read_file(path_file, num, self.J)
        return self

    def build_haar(self):
        if self.signal:
            self.S[0] = self.signal[:]
            for j in range(self.J):
                for m in range(int(self.M / (2 ** (j + 1)))):
                    self.S[j + 1][m] = (self.S[j][2 * m] + self.S[j][2 * m + 1]) / math.sqrt(2)
                    self.d[j + 1][m] = (self.S[j][2 * m] - self.S[j][2 * m + 1]) / math.sqrt(2)
            for j in range(self.J - 1, -1, -1):
                for m in range(int(self.M / (2 ** j))):
                    self.S[j][m] = (self.S[j + 1][int(m / 2)] + ((-1) ** m) * self.d[j + 1][int(m / 2)]) / math.sqrt(2)

            return self

    def with_check(self):
        check_right = 0
        for i in range(self.M):
            check_right += (self.S[0][i] ** 2)
        check_left = 0
        for j in range(self.J + 1):
            for m in range(self.M):
                check_left += (self.d[j][m] ** 2)
        self.accuracy = check_right - (check_left + (self.S[self.J][0] ** 2))
        return self

    def with_plot(self):
        x = [i for i in range(self.M)]
        plt.figure()
        plt.plot(x, self.signal, color='red')
        plt.plot(x, self.S[0], '--')
        plt.legend(['Signal', 'S[0] - interpolated'])
        plt.title(f'Difference: {self.accuracy}')
        plt.show()
        return self


if __name__ == '__main__':
    HaarSystem(128) \
        .from_file_signal("..\\Resources\\ClosedEyes.asc", 1) \
        .build_haar() \
        .with_check() \
        .with_plot() \

