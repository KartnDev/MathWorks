from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import int64
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import normalize, StandardScaler


def row_to_intenum(row: [str]) -> [int]:
    row_unique_values = list(set(row))
    mapping = {key: value for value, key in enumerate(row_unique_values)}

    return list(map(lambda item: mapping[item], row))


def optimize_csv(csv: DataFrame):
    for column_alias in csv:
        current_column = csv[column_alias]

        if current_column.dtypes is np.dtype(object):
            csv[column_alias] = normalize(np.array(row_to_intenum(current_column))[:, np.newaxis], axis=0).ravel()

        elif current_column.dtypes is np.dtype(int64) or current_column.dtypes is np.dtype(np.float64):
            csv[column_alias] = normalize(np.array(current_column)[:, np.newaxis], axis=0).ravel()


data = pd.read_csv('res/airfoil_self_noise.dat', sep='\t')
optimize_csv(data)

data.columns = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'pressure']
plt.scatter(data['thickness'], data['pressure'], s=0.3)
plt.show()

y = np.array(data['pressure'])
del data['pressure']

model = LinearRegression()
model.fit(data, y)
predicted = model.predict(data)

plt.scatter(data['thickness'], y, s=0.3)
plt.scatter(data['thickness'], predicted, s=0.3)
plt.show()
