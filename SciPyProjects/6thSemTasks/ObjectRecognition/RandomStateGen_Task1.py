import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import erlang

if __name__ == '__main__':
    k = 10000

    scale_param_b = 2
    form_param_c = 5
    predefined_mean = scale_param_b * form_param_c

    a = erlang.rvs(a=form_param_c, scale=scale_param_b, size=k)
    means_per_realise = np.array([np.mean(a[0: i]) for i in range(k)])

    plt.plot(means_per_realise, linewidth=1)
    plt.plot([scale_param_b * form_param_c for _ in range(k)], linewidth=1)
    plt.show()