import numpy as np

# For my fork in git
try:
    from PyAndMl.MLLec5.computeCost import computeCost
except:
    # for standalone invoke
    from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """

    J_history = []
    m = y.shape[0]
    n = theta.shape[0]
    temp = np.zeros((n, 1))

    for i in range(num_iters):
        temp[0] = theta[0][0] - (alpha * (1 / m) * sum(X.dot(theta) - y))
        theta[0][0] = temp[0][0]
        for j in range(1, n):
            temp[j][0] = theta[j][0] - (alpha * (1 / m) * sum(X.T.dot(X.dot(theta) - y)))
            J_history.append(computeCost(X, y, theta))  # сохранение значений стоимостной функции
            # на каждой итерации
            theta[j][0] = temp[j][0]
    return theta, J_history
