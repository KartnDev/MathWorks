from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

data = pd.read_csv('res/RSAFS.csv')
plt.scatter(data['DATE'], data['RSAFS'], s=0.3)
plt.xticks(data['DATE'][::100])
plt.show()

x = [i for i, _ in enumerate(map(lambda s: datetime.strptime(s, '%Y-%m-%d'), data['DATE']))]

normalized_x = preprocessing.normalize(np.array(x)[:, np.newaxis], axis=0).ravel()
normalized_y = preprocessing.normalize(data['RSAFS'][:, np.newaxis], axis=0).ravel()

model = LinearRegression()
model.fit(normalized_x[:, np.newaxis], normalized_y)

half_year_offset = 183
step = np.abs(normalized_x[0] - normalized_x[1])

predict_x = np.array([normalized_x[-1] + step * i for i in range(1, half_year_offset)])
predict_y = model.predict(predict_x.reshape(-1, 1))

plt.scatter(normalized_x, normalized_y, s=0.3)
plt.xticks(normalized_x[::100])

plt.scatter(predict_x, predict_y, s=0.3)
plt.xticks(predict_x[::100])
plt.show()