"""This modules contains quadrature methods."""
from grid import Grid
from legendre_functions import legendre_rec


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
