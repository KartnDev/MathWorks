import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numba import jit, prange



def build_matrix(waveform, j_length, m_size):
    S = []
    D = []
    S.append(waveform)
    for j in range(j_length):
        end = m_size / pow(2, j + 1)
        end = int(end)
        temp = []
        temp2 = []
        for m in range(end):
            val = S[j][2 * m] + S[j][2 * m + 1]
            val2 = S[j][2 * m] - S[j][2 * m + 1]
            temp.append(val / np.sqrt(2))
            temp2.append(val2 / np.sqrt(2))
        S.append(temp)
        D.append(temp2)

    return S, D


def get_summaries(waveform, D_array, j_length, m_size):
    sum1 = 0
    sum2 = 0

    for j in range(j_length + 1):
        end = m_size / pow(2, j + 1)
        end = int(end)
        for m in range(end):
            sum1 += np.power(D_array[j][m], 2)
    for m in range(m_size):
        sum2 += np.power(waveform[m], 2)

    return sum1, sum2


def final_sum_array(S_array, D_array, j_length, m_size):
    Snew = np.array([])
    for j in range(j_length - 1, -1, -1):
        end = m_size / np.power(2, j)
        end = int(end)
        temp = []
        for m in range(end):
            val = S_array[j + 1][int(m / 2)] + np.power(-1, m) * D_array[j][int(m / 2)]
            temp.append(val / np.sqrt(2))
        Snew = [temp] + Snew
    return Snew


if __name__ == '__main__':
    J = 6
    M = pow(2, J)
    t = np.arange(M)
    Fm = signal.sawtooth(2 * np.pi * 5 * t)

    S, D = build_matrix(Fm, J, M)

    sum1, sum2 = get_summaries(Fm, D, J, M)

    print(sum1 + np.power(S[J][0], 2))
    print(sum2)

    Snew = final_sum_array(S, D, J, M)

    plt.plot(t, Fm)
    plt.show()
    plt.plot(t, Snew[0])
    plt.show()
