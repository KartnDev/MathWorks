import numpy as np


# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def tdma_solver(a, b, c, d):

    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


if __name__ == '__main__':
    A = np.array([[10, 2, 0, 0],
                  [3, 10, 4, 0],
                  [0, 1, 7, 5],
                  [0, 0, 3, 4]], dtype=np.float64)

    a = np.array([3., 1, 3])
    b = np.array([10., 10., 7., 4.])
    c = np.array([2., 4., 5.])
    d = np.array([3, 4, 5, 6.])

    print(tdma_solver(a, b, c, d))

    # compare against numpy linear algebra library
    print(np.linalg.solve(A, d))
