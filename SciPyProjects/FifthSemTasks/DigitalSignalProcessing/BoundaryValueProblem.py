import numpy as np
import matplotlib.pyplot as plt

from DigitalSignalProcessing.TridiagonalMatrixAlgorithm import thomas_solver


def finite_difference(u_init, v_init, w_init, times_stepping, a_left_boundary: float, b_right_boundary: float):
    """Implements the shooting method to solve linear second order BVPs

    Compute finite difference solution to the BVP

        x'' = u(t) + v(t) x + w(t) x'
        x(t[0]) = a, x(t[n-1]) = b

    t should be passed in as an n element array.   u, v, and w should be
    either n element arrays corresponding to u(t), v(t) and w(t) or
    scalars, in which case an n element array with the given value is
    generated for each of them.

    USAGE:
        x = fd(u, v, w, t, a, b)

    INPUT:
        u_init, v_init, w-init - arrays containing u(t), v(t), and w(t) values.  May be
                specified as Python lists, NumPy arrays, or scalars.  In
                each case they are converted to NumPy arrays.
        times_stepping - array of n time values to determine x at
        a - solution value at the left boundary: a = x(t[0])
        b - solution value at the right boundary: b = x(t[n-1])

    OUTPUT:
        x     - array of solution function values corresponding to the
                values in the supplied array t.
    """

    # Get the dimension of t and make sure that t is an n-element vector

    if type(times_stepping) != np.ndarray:
        if type(times_stepping) == list:
            times_stepping = np.array(times_stepping)
        else:
            times_stepping = np.array([float(times_stepping)])

    n = len(times_stepping)

    # Make sure that u, v, and w are either scalars or n-element vectors.
    # If they are scalars then we create vectors with the scalar value in
    # each position.

    if type(u_init) == int or type(u_init) == float:
        u_init = np.array([float(u_init)] * n)

    if type(v_init) == int or type(v_init) == float:
        v_init = np.array([float(v_init)] * n)

    if type(w_init) == int or type(w_init) == float:
        w_init = np.array([float(w_init)] * n)

    # Compute the stepsize.  It is assumed that all elements in t are
    # equally spaced.

    h = times_stepping[1] - times_stepping[0];

    # Construct tridiagonal system; boundary conditions appear as first and
    # last equations in system.

    under_diag = -(1.0 + w_init[1:n] * h / 2.0)
    under_diag[-1] = 0.0

    upper_diag = -(1.0 - w_init[0:n - 1] * h / 2.0)
    upper_diag[0] = 0.0

    vector = 2.0 + h * h * v_init
    vector[0] = vector[n - 1] = 1.0

    main_diag = - h * h * u_init
    main_diag[0] = a_left_boundary
    main_diag[n - 1] = b_right_boundary

    # Solve tri diagonal system

    x = thomas_solver(under_diag, vector, upper_diag, main_diag)

    return x


def boundary_problem_solve(x: any):
    return (2 * x + 1) * np.exp(x)


if __name__ == "__main__":
    # Solves y'' = y + 4exp(x), y(0)=1, y(1/2) = 2exp(1/2) using both the
    # finite difference method

    # Set up interval.  We will solve the problem for both n=64 and n=128.

    a = 0.0
    b = 1.0
    n1 = 64
    n2 = 128
    t1 = np.linspace(a, b, n1)
    t2 = np.linspace(a, b, n2)

    # Compute finite difference solutions

    xfd1 = finite_difference(2 * t1, 1, 0, t1, 0, -1)
    xfd2 = finite_difference(2 * t2, 1, 0, t2, 0, -1)

    func = boundary_problem_solve(t1)

    plt.plot(t1, xfd1)
    plt.plot(t1, xfd2[::2])
    #plt.plot(t1, func)
    plt.show()
