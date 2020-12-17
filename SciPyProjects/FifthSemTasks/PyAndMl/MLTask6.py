from sklearn import linear_model
from sklearn import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X, y)

predictions = lm.predict(X)
print(predictions)

print(lm.score(X, y))

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = - 5 * x - 100 + rng.randn(50)
plt.scatter(x, y)
plt.show()

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
print(model.score(x[:, np.newaxis], y))


rng = np.random.RandomState(1)
x = 1 * rng.rand(50)
y = - 5 * x + 100 + rng.randn(50)
plt.scatter(x, y)
plt.show()

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 1, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
print(model.score(x[:, np.newaxis], y))