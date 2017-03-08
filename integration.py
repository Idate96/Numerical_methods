from scipy import integrate
from grid import Grid
from legendre_functions import legendre_rec
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import math


def quad_glob(fs, a, b, n):
    """
    Gauss Lobatto quadrature I(f(x)) = ∑ w_i(ξ_i) f_i(ξ_i).
    weights w_i, ξ_i nodal point of gauss lobatto "grid".
    Error term ∝ f^{2n-2}
    Exact for p ∈ P_{2n-1}
     """
    # Exact up to polinomial in P_(2n-1)
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
    value = 0

    for i in range(n + 1):
        if i == 0 or i == n:
            # print(i)
            # print("value nodal pt ", grid.nodal_pts[i])
            w_i = 2 / (n * (n + 1))
            value_partial = w_i
            for f in fs:
                value_partial *= f(grid.nodal_pts[i])
            value += value_partial
        else:
            w_i = 2 / (n * (n + 1) * legendre_rec(grid.nodal_pts[i], n) ** 2)
            value_partial = w_i
            for f in fs:
                value_partial *= f(grid.nodal_pts[i])
            value += value_partial
    # TODO scaling to arbitrary interval
    return value


def f1(x):
    return x
#
#


def f2(x):
    return x ** 3 + x - 1 + x ** 2 + 1 / 2 * x ** 9 + 1 / 4 * x ** 5 + 2 * x ** 8
    #
value1 = quad_glob([f2], -1, 1, 4)
# value2 = quad_glob([f2], -1, 1, 3)
value_sci = integrate.quad(f2, -1, 1)
# # fnew = partial(product_e0, [f1, f1])
# # value1 = integrate.quad(fnew, -1, 1)
print(value1, value_sci)
