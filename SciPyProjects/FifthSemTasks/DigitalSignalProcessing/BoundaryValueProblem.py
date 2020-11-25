import numpy
import matplotlib.pyplot as plt


def finite_difference(u_init, v_init, w_init, t_times, a_diagonal, solutions_b):
    """Implements the shooting method to solve linear second order BVPs

    Compute finite difference solution to the BVP

        x'' = u(t) + v(t) x + w(t) x'
        x(t[0]) = a, x(t[n-1]) = b

    t should be passed in as an n element array.   u, v, and w should be
    either n element arrays corresponding to u(t), v(t) and w(t) or
    scalars, in which case an n element array with the given value is
    generated for each of them.

    USAGE:
        x = finite_difference(u, v, w, t, a, b)

    INPUT:
        u_init, v_init, w_init - arrays containing u(t), v(t), and w(t) values.  May be
                specified as Python lists, NumPy arrays, or scalars.  In
                each case they are converted to NumPy arrays.
        t_times - array of n time values to determine x at
        a_diagonal - solution value at the left boundary: a = x(t[0])
        solutions_b - solution value at the right boundary: b = x(t[n-1])

    OUTPUT:
        x     - array of solution function values corresponding to the
                values in the supplied array t.
    """

    # Get the dimension of t and make sure that t is an n-element vector

    if type(t_times) != numpy.ndarray:
        if type(t_times) == list:
            t_times = numpy.array(t_times)
        else:
            t_times = numpy.array([float(t_times)])

    n = len(t_times)

    # Make sure that u, v, and w are either scalars or n-element vectors.
    # If they are scalars then we create vectors with the scalar value in
    # each position.

    if type(u_init) == int or type(u_init) == float:
        u_init = numpy.array([float(u_init)] * n)

    if type(v_init) == int or type(v_init) == float:
        v_init = numpy.array([float(v_init)] * n)

    if type(w_init) == int or type(w_init) == float:
        w_init = numpy.array([float(w_init)] * n)

    # Compute the step size.  It is assumed that all elements in t are
    # equally spaced.

    h_steps = t_times[1] - t_times[0];

    # Construct tri diagonal system; boundary conditions appear as first and
    # last equations in system.

    a_diagonal = -(1.0 + w_init[1:n] * h_steps / 2.0)
    a_diagonal[-1] = 0.0

    c_diagonal = -(1.0 - w_init[0:n - 1] * h_steps / 2.0)
    c_diagonal[0] = 0.0

    D = 2.0 + h_steps * h_steps * v_init
    D[0] = D[n - 1] = 1.0

    B = - h_steps * h_steps * u_init
    B[0] = a_diagonal
    B[n - 1] = solutions_b

    # Solve tri diagonal system Aka "Thomas method" / "Tri diagonal alg"

    for i in range(1, n):
        x_multiplied = a_diagonal[i - 1] / D[i - 1]
        D[i] = D[i] - x_multiplied * c_diagonal[i - 1]
        B[i] = B[i] - x_multiplied * B[i - 1]

    x_result = numpy.zeros(n)
    x_result[n - 1] = B[n - 1] / D[n - 1]

    for i in range(n - 2, -1, -1):
        x_result[i] = (B[i] - c_diagonal[i] * x_result[i + 1]) / D[i]

    return x_result


def boundary_problem_solve(x: any):
    return (2 * x + 1) * numpy.exp(x)


def exact(t):
    return numpy.exp(t) * (1 + 2 * t)


if __name__ == "__main__":
    # Solves y'' = y + 4exp(x), y(0)=1, y(1/2) = 2exp(1/2) using both the
    # finite difference method

    # Set up interval.  We will solve the problem for both n=64 and n=128.

    a = 0.0
    b = 0.5
    n1 = 64
    n2 = 128
    t1 = numpy.linspace(a, b, n1)
    t2 = numpy.linspace(a, b, n2)

    x1 = exact(t1)
    x2 = exact(t2)

    # Compute finite difference solutions

    xfd1 = finite_difference(4 * numpy.exp(t1), 1, 0, t1, 1, 2 * numpy.exp(0.5))
    xfd2 = finite_difference(4 * numpy.exp(t2), 1, 0, t2, 1, 2 * numpy.exp(0.5))

    func = boundary_problem_solve(t1)

    plt.plot(t1, xfd1)
    plt.plot(t1, xfd2[::2])
    plt.plot(t1, func)
    plt.show()
