import numpy as np
import integration
from functools import partial
from scipy import integrate


def product_basis(x, fs, indexes, g=1):
    value = 1
    for i, f in enumerate(fs):
        print('func at nodal pts: ', f(x, indexes[i]))
        print('index ', i)
        value *= f(x, indexes[i])
    return value * g


def inner_product(basis, degree):
    if degree == 1:
        fs = [basis.edge, basis.edge]
    elif degree == 0:
        fs = [basis.lagrange, basis.lagrange]
    M = np.zeros((basis.n, basis.n))
    for i in range(basis.n):
        for j in range(i + 1):
            # TODO use gauss lobatto from integrate.py
            prod = partial(product_basis, fs=fs, indexes=[i, j])
            # M[i, j] = integrate.quad(prod, -1, 1)[0]
            M[i, j] = integration.quad_glob([prod], -1, 1, basis.n)
            M[j, i] = M[i, j]
    return M
