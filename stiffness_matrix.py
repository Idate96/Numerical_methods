"""Allows to calculate the inner product of basis functions."""
from functools import partial
import integration
import math
import numpy as np
from scipy import integrate


def product_basis(x, fs, indexes, g=1, a=-1, b=1):
    """Multiply two functions pointwise.

    Useful function when the product of more than one function needs to be passed to a quadrature.

    Args:
        x   (float, np.array) = domain of evaluation
        fs  (list of function obj) = list of functions to be multiplied
        indexes (list of ints) = the indexes reppresent the indexes of the basis
        functions as reppresented in the Basis. [i,j] represent the ith and jth basis function
        g = metric tensor
    Returns:
        float,np.array = value of the product of funcs
    """
    # TODO(implement metric tensor g)
    value = 1
    # pointwise multiply the basis functions
    for i, f in enumerate(fs):
        value *= f((b - a) / 2 * (x + 1) + a, indexes[i])
    return value * g


def inner_product(basis, degree, a=-1, b=1):
    """Calculate the inner product of two basis functios.

    Args:
        basis (Basis obj) = Basis object for your space of functions
        degree (int) = Degree of the basis function. 0 corresponds to
        simple lagrange poli, 1 to edge functions...

    Returns:
        (n-degree) × (n-degree) array containing the values of the inner product over the element
    """
    # TODO(merge the cases using the int degree, generalize the interval from -1,1 to a,b)
    # degree_quad = math.ceil((basis.n ** 2 + 1) / 2)
    if degree == 1:
        fs = [basis.edge, basis.edge]
        M = np.zeros((basis.n - 1, basis.n - 1))
        for i in range(basis.n - 1):
            for j in range(i + 1):
                prod = partial(product_basis, fs=fs, indexes=[i + 1, j + 1], a=a, b=b)
                # M[i, j] = integration.quad_glob([prod], -1, 1, degree_quad)
                linear_scaling = 2 / (b - a)
                # linear_scaling = (b - a) / 2
                M[i, j] = linear_scaling * integrate.quad(prod, -1, 1)[0]
                M[j, i] = M[i, j]
    elif degree == 0:
        fs = [basis.lagrange, basis.lagrange]
        M = np.zeros((basis.n, basis.n))
        for i in range(basis.n):
            for j in range(i + 1):
                prod = partial(product_basis, fs=fs, indexes=[i, j], a=a, b=b)
                linear_scaling = 2 / (b - a)
                M[i, j] = linear_scaling * integrate.quad(prod, -1, 1)[0]
                # M[i, j] = integration.quad_glob([prod], -1, 1, degree_quad)
                M[j, i] = M[i, j]
    return M
