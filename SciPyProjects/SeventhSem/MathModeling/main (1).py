import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

""" CONST """
ERR = 0.3

data = pandas.read_csv('resources/market-price_hit.csv')
data_list = list(data['price'])
# Difference method
deg = 0
diff_k = [1 for _ in range(0, len(data_list))]
diff = [0 for _ in range(0, len(data_list))]
for i in range(2, len(data_list)):
    diff[i] = data_list[i] - data_list[i - 1]
diff_k[1] = sum(diff)
diff.clear()
diff = [0 for _ in range(0, len(data_list))]
for j in range(2, len(data_list)):
    for i in range(j + 1, len(data_list)):
        diff[i] = data_list[i] - data_list[i - 1]
    diff_k[j] = sum(diff) - diff_k[j - 1]
    diff.clear()
    diff = [0 for _ in range(0, len(data_list))]
for i in range(0, len(diff_k)):
    if abs(diff_k[i]) < ERR:
        deg = i + 1
        break
print('Degree =', deg)
time = np.arange(0, len(data_list))
# Data for polynom
A = time[:, np.newaxis] ** [0, 1, 2, 3]
# Data for trigonometry
table = []
for i in range(len(data_list)):
    table.append([1, np.sin(i), np.cos(i) * i, i])
# Data for spline
t = [i * 5 for i in range(1, 50)]
# Search coef of spline and their points
spl = splrep(time, data_list, task=-1, t=t, k=3)
mnk_spl = splev(time, spl)
# Search coef of polynom
a1 = np.linalg.lstsq(table, data_list, rcond=None)[0]
# Search coef of trigonometry
a2 = np.linalg.lstsq(A, data_list, rcond=None)[0]
newDataPolynom = []
newDataTrigonometry = []
for i in range(len(data_list)):
    # Point for polynom
    newDataPolynom.append(a1[0] + a1[1] * np.sin(i) + a1[2] * i * np.cos(i) + a1[3] * i)
    # Points for trigonometry
    newDataTrigonometry.append(a2[0] + a2[1] * i + a2[2] * i ** 2 + a2[3] * i ** 3)
x = np.arange(0, len(data_list))
plt.title('Модели кривых роста')
plt.plot(x, data_list, label="Изначально")
plt.plot(x, newDataPolynom, label="Тригонометрический")
plt.plot(x, newDataTrigonometry, label="Полиномиальный")
plt.plot(x, mnk_spl, label="Сплайн")
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()
