import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
import math


def getH(a):
    if a < -10:
        return 0
    elif a > 19:
        return 0
    else:
        return h[a + 10]


if __name__ == '__main__':
    N = 15
    n = 2 * N;
    h = np.array([-0.0001499638, 0.0002535612, 0.0015402457, -0.0029411108, -0.0071637819, 0.0165520664, 0.0199178043
                     , -0.0649972628, -0.0368000736, 0.2980923235, 0.5475054294, 0.3097068490, -0.0438660508,
                  -0.0746522389, 0.0291958795, 0.0231107770, -0.0139738879, -0.0064800900, 0.0047830014,
                  0.0017206547, -0.0011758222, -0.0004512270, 0.0002137298, 0.00009937776, -0.0000292321,
                  -0.0000150720, 0.0000026408, 0.0000014593, -0.0000001184, -0.0000000673])

    h = h * np.sqrt(2)
    J = 5
    M = pow(2,J)
    t = np.arange(M)
    Fm = signal.sawtooth(2 * np.pi * 5 * t)
    S = []
    D = []
    S.append(Fm)
    for j in range(J):
        end = M / pow(2, j + 1)
        temp = []
        tempD = []
        for l in range(int(end)):
            Sum = 0;
            SumD = 0;
            for m in range(len(S[j])):
                Sum += getH(m - 2 * l) * S[j][m]
                SumD += pow(-1, m) * getH(1 - m + 2 * l) * S[j][m]
            tempD.append(SumD)
            temp.append(Sum)
        S.append(temp)
        D.append(tempD)
    print(len(S[0]), len(S[1]), len(S[2]), len(S[3]), len(S[4]))
    sum1 = 0
    sum2 = 0
    for j in range(J + 1):
        end = M / pow(2, j + 1)
        end = int(end)
        for m in range(end):
            sum1 += math.pow(D[j][m], 2)
    for m in range(M):
        sum2 += math.pow(Fm[m], 2)
    print(sum1 + pow(S[J][0], 2))
    print(sum2)

    Snew = [S[5]]
    for j in range(J - 1, -1, -1):
        end = int(M / pow(2, j))
        temp = []
        for m in range(end):
            Sum = 0;
            for l in range(len(S[j + 1])):
                Sum += S[j + 1][l] * getH(m - 2 * l) + pow(-1, m) * D[j][l] * getH(1 - m + 2 * l)
            temp.append(Sum)
        Snew.append(temp)
    print(Snew[5])

    plt.plot(t, Fm)
    plt.plot(t, Snew[5])
    plt.show()