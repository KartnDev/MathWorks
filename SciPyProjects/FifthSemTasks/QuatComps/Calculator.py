import numpy as  np

n = 60000

res = np.linalg.solve(np.random.random((n, n)), np.random.random(n))

print(res.shape)