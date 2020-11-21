import numpy as np
from PyAndMl.MLTask5.computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """

    J_history = []
    m = y.shape[0]

    for i in range(num_iters):
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию

        for k in range(0, len(theta)):
            sum_pred = np.sum(np.dot(X[0], theta) - y[0])
            for j in range(1, m):
                sum_pred += np.sum((np.dot(X[j], theta) - y[j]) * X[j])

            theta[k] = theta[k] - alpha / m * sum_pred

        # ============================================================

        J_history.append(computeCost(X, y, theta))  # сохранение значений стоимостной функции
        # на каждой итерации

    return theta, J_history
