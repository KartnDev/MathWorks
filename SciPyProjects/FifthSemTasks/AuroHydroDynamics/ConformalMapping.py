import numpy as np
import matplotlib.pyplot as plt
import api
import scipy

grid = api.init_grid()
rotation = lambda z: z * np.exp(1j * np.pi / 6)
api.plot_map(grid, rotation)[0].plot()

