"""This modules contains quadrature methods."""
from grid import Grid
from legendre_functions import legendre_rec
import numpy an np


def quad_glob(fs, a, b, n):
    """Gauss Lobatto quadrature I(f(x)) = ∑ w_i(ξ_i) f_i(ξ_i).

    The weights are indicated by w_i (source worlfram).
    The nodal point ξ_i are the nodal points of gauss lobatto "grid".
    Error term ∝ f^{2n-2}
    Exact for p ∈ P_{2n-1}.

    Args:
        fs (list) = list of functions (if more than one is present they are multiplied pointwise).
        a (float) = start domain.
        b (float) = end domain.
        n (int) = degree of quadrature.

    Return:
        value (float) = value of the integral over the interval [a,b].
    """
    # Exact up to polinomial in P_(2n-1)
    # init grid
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
    # init value of the integral
    value = 0
    for i in range(n + 1):
        if i == 0 or i == n:
            # weight for starting and ending point
            w_i = 2 / (n * (n + 1))
            value_partial = w_i
            for f in fs:
                value_partial *= f(grid.nodal_pts[i])
            value += value_partial
        else:
            # weights for the nodes
            w_i = 2 / (n * (n + 1) * legendre_rec(grid.nodal_pts[i], n) ** 2)
            value_partial = w_i
            for f in fs:
                value_partial *= f(grid.nodal_pts[i])
            value += value_partial
    # TODO(scaling to arbitrary interval)
    return value


def quad_trap(a, b, n, f, exact_int=0):
    """Trapezoid quadrature method.

        Args:
            a (float) = integration interval start
            b (float) = integration interval end
            n (int) = number of subintervals of integration
            f (obj func) = function to be integrated.
            exact_int (float : optional) = exact integral value

        Returns:
            integral_value (float) = value of the integral_value.
            error (float) = error of integration if available
    """
    # subintervals
    x = np.linspace(a, b, n + 1)
    # spacing
    h = (b - a) / n
    # trap quadrature
    num_int = h / 2 * (f(x) + f(x + h))
    # sum value of subintervals
    integral_value = np.sum(num_int[:-1])
    return integral_value, integral_value - exact_int


def quad_gauss(a, b, n, f, exact_int=0):
    """Gauss quadrature.
        Args:
            a (float) = integration interval start
            b (float) = integration interval end
            n (int) = number of subintervals of integration
            f (obj func) = function to be integrated.
            exact_int (float : optional) = exact integral value

        Returns:
            integral_value (float) = value of the integral_value.
            error (float) = error of integration if available
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    num_int = h / 2 * (f(x + h * ((-1 / 3 ** 0.5 + 1) / 2)) +
                       f(x + h * ((1 / 3 ** 0.5 + 1) / 2)))
    integral_value = np.sum(num_int[:-1])
    return integral_value, integral_value - exact_int


def quad_simpson(a, b, n, f, exact_int=0):
    """Simpson quadrature.

        Args:
            a (float) = integration interval start
            b (float) = integration interval end
            n (int) = number of subintervals of integration
            f (obj func) = function to be integrated.
            exact_int (float : optional) = exact integral value

        Returns:
            integral_value (float) = value of the integral_value.
            error (float) = error of integration if available
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    num_int = h / 2 * (1 / 3 * f(x) + 4 / 3 * f(x + h / 2) + 1 / 3 * f(x + h))
    integral_value = np.sum(num_int[:-1])
    return integral_value, integral_value - exact_int
