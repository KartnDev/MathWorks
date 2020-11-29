import matplotlib.pyplot as plt


def plotData(data):
    """
        Функция позволяет выполнить визуализацию данных в декартовой 
        системе координат с подписанным осями (численность населения 
        и прибыль)
    """
    plt.figure(dpi=150)
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'rx', label='Тренировочные данные', markersize=5)
    plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Численность населения в 10.000')
    plt.ylabel('Прибыль в 10.000$')
    plt.grid()