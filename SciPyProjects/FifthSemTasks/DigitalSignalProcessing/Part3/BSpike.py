import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
import math
J = 6
M = pow(2,J)
t = np.arange(M)
Fm = signal.sawtooth(2*np.pi*5*t)
S = []
D = []
S.append(Fm)
for j in range(J):
    end = M/pow(2, j+1)
    end = int(end)
    temp = []
    temp2 = []
    for m in range(end):
        val = S[j][2*m]+ S[j][2*m+1]
        val2 = S[j][2*m] - S[j][2*m+1]
        temp.append(val/math.sqrt(2))
        temp2.append(val2/math.sqrt(2))
    S.append(temp)
    D.append(temp2)
sum1 = 0
sum2 = 0
for j in range(J+1):
    end = M/pow(2,j+1)
    end = int(end)
    for m in range(end):
        sum1 += math.pow(D[j][m],2)
for m in range(M):
    sum2 += math.pow(Fm[m],2)
print(sum1 + pow(S[J][0],2))
print(sum2)
Snew = [];
for j in range(J-1,-1,-1):
    end = M/pow(2,j)
    end = int(end)
    temp = []
    for m in range(end):
        val = S[j+1][int(m/2)] + pow(-1,m)*D[j][int(m/2)]
        temp.append(val/math.sqrt(2))
    Snew = [temp] + Snew
plt.plot(t,Fm)
plt.show()
plt.plot(t,Snew[0])
plt.show()
