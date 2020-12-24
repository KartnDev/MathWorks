import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def get_sum_at(vector, index):
    sum_val = 0
    for i in prange(n_length - 2 * index):
        sum_val += vector[i] * vector[i + 2 * index]
    return sum_val


if __name__ == '__main__':
    h_vector = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0001499638, 0.0002535612, 0.0015402457, -0.0029411108, -0.0071637819,
         0.0165520664, 0.0199178043, -0.0649972628, -0.0368000736, 0.2980923235, 0.5475054294, 0.3097068490,
         -0.0438660508, -0.0746522389, 0.0291958795, 0.0231107770, -0.0139738879, -0.0064800900, 0.0047830014,
         0.0017206547, -0.0011758222, -0.0004512270, 0.0002137298, 0.00009937776, -0.0000292321, -0.0000150720,
         0.0000026408, 0.0000014593, -0.0000001184, -0.0000000674]) * np.sqrt(2)

    n_length = len(h_vector)

    print("Summary: ", h_vector.sum(), " | ", np.sqrt(2))
    print("Length", n_length)

    for current_index in range(int(n_length / 2)):
        print(f"{current_index}) ", get_sum_at(h_vector, current_index))
