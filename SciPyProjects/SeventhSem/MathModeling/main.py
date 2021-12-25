import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import splrep, splev


def read_data():
    data = pandas.read_csv('E:\\Projects\\mat-model\\5 lab\\OGZPY.csv')
    return list(data['Close'])


def get_models(data_list):
    time = np.arange(0, len(data_list))
    A = time[:, np.newaxis] ** [0, 1, 2, 3]
    table = []
    for i in range(len(data_list)):
        table.append([1, np.sin(i), np.cos(i) * i, i])
    t = [i * 5 for i in range(1, 50)]
    spl = splrep(time, data_list, task=-1, t=t, k=3)
    # Splain Model
    mnk_spl = splev(time, spl)
    a1 = np.linalg.lstsq(table, data_list, rcond=None)[0]
    a2 = np.linalg.lstsq(A, data_list, rcond=None)[0]
    newDataPolynom = []
    newDataTrigonometry = []
    for i in range(len(data_list)):
        # Polynom Model
        newDataPolynom.append(a1[0] + a1[1] * np.sin(i) + a1[2] * i * np.cos(i) + a1[3] * i)
        # Trigonometric Model
        newDataTrigonometry.append(a2[0] + a2[1] * i + a2[2] * i ** 2 + a2[3] * i ** 3)
    x = np.arange(0, len(data_list))
    return mnk_spl, newDataPolynom, newDataTrigonometry

def plot(data, mnk, polynom, trig):
    x = np.arange(0, len(data))
    plt.title('Модели кривых роста')
    plt.plot(x, data_list, label="Изначально")
    plt.plot(x, polynom, label="Тригонометрический")
    plt.plot(x, trig, label="Полиномиальный")
    plt.plot(x, mnk, label="Сплайн")
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

def calc_d(data, model):
    e = data - model
    d_numerator = 0
    d_denominator = 0
    for i in range(0, len(e)):
        if i != 0:
            d_numerator += (e[i] - e[i - 1]) ** 2
        d_denominator += e[i] ** 2
    return d_numerator/d_denominator


def get_d(n):
    # For alpha = 0.01
    if n == 50:
        return 1.324, 1.403
    if n == 150:
        return 1.611, 1.637
    if n == 100:
        return 1.522, 1.562
    if n == 200:
        return 1.664, 1.684
    else:
        return 0, 0


def calc_accuracy(data, model):
    sum_abs_error = 0
    for i in range(0, len(data)):
        sum_abs_error += ((np.abs(model[i] - data[i])) / data[i])
    return (sum_abs_error / len(data)) * 100


def calc_adeq(data, model, title):
    e = data - model
    n = len(e)
    A_num = 0
    A_denum = 0
    E_num = 0
    E_denum = 0
    for i in range(0, n):
        A_num += e[i] ** 3
        A_denum += e[i] ** 2
        E_num += e[i] ** 4
        E_denum += e[i] ** 2
    A = (A_num * 1 / n) / np.sqrt(((1 / n * A_denum) ** 3))
    E = ((E_num * 1 / n) / ((1 / n * E_denum) ** 2)) - 3 + (6/(n+1))
    ineq_A = 1.5 * np.sqrt((6 * (n - 2))/((n + 1) * (n + 3))),\
             2 * np.sqrt((6 * (n - 2))/((n + 1) * (n + 3)))
    ineq_E = 1.5 * np.sqrt((24 * n * (n - 2) * (n - 3)) / (((n + 1) ** 2) * (n + 3) * (n + 5))),\
             2 * np.sqrt((24 * n * (n - 2) * (n - 3)) / (((n + 1) ** 2) * (n + 3) * (n + 5)))
    print(f'A {np.abs(A)} < {ineq_A[0]} | {np.abs(A)} > {ineq_A[1]}')
    print(f'Э {np.abs(E)} < {ineq_E[0]} | {np.abs(E)} > {ineq_E[1]}')
    plt.title(title)
    sns.kdeplot(e, shade=True)
    plt.show()


def get_r(data, model):
    e = 0
    e_f = 0
    for i in range(0, len(data)):
        e += (model[i] - data[i] ** 2)
        e_f += (model[i] - (sum(data) / len(data)) ** 2)
    r = 1 - (e / e_f)
    print(f'R^2 = {np.abs(r)}')


data_list = np.array(read_data())
mnk, polynom, trig = get_models(data_list)
N = 150
print('MNK Spline')
calc_adeq(data_list, mnk, 'MNK')
print('Polynom')
calc_adeq(data_list, polynom, 'Polynom')
print('Trigonometry')
calc_adeq(data_list, trig, 'Trigonometry')
d_mnk = calc_d(data_list[0:N], mnk[0:N])
d_polynom = calc_d(data_list[0:N], polynom[0:N])
d_trig = calc_d(data_list[0:N], trig[0:N])
print('\n')
print(f'MNK Spline critical = {d_mnk} | (4-d) = {4-d_mnk}')
print(f'Polynom critical = {d_polynom} | (4-d) = {4-d_polynom}')
print(f'Trigonometry critical = {d_polynom} | (4-d) = {4-d_trig}')
d_1, d_2 = get_d(N)
print(f'd_1 = {d_1} | d_2 = {d_2}\n')
e_mnk = np.round(calc_accuracy(data_list, mnk), 5)
e_polynom = np.round(calc_accuracy(data_list, polynom), 5)
e_trig = np.round(calc_accuracy(data_list, trig), 5)
print(f'MNK Spline Error = {e_mnk}%')
print(f'Polynom Error = {e_polynom}%')
print(f'Trigonometry Error = {e_trig}%')
print('')
print('MNK Spline')
get_r(data_list, mnk)
print('Polynom')
get_r(data_list, polynom)
print('Trigonometry')
get_r(data_list, trig)
