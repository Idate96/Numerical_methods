"""This module contain function to calculate Legendre polynomials and its derivatives."""
import numpy as np
from scipy.special import legendre
import scipy as sc


def legendre_rec(x, n):
    """Return Legendre polynomial of order n."""
    if n == 1:
        return x
    if n == 0:
        return 1
    leg = (2 * (n - 1) + 1) * x * legendre_rec(x, n - 1) - (n - 1) * legendre_rec(x, n - 2)
    return leg / n


def legendre_prime(x, n):
    """Calculate first derivative of the nth Legendre Polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = n * legendre(n - 1)(x) - n * x * legendre(n)(x)
    return legendre_p * (1 - x ** 2)


def legendre_double_prime_recursive(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)


def legendre_double_prime_num(x, n):
    # The result is highly inaccurate and causes newton rapson to diverge
    legendre_p = partial(legendre_prime, n=n)
    legendre_pp = np.zeros(np.size(x))
    if isinstance(x, np.ndarray):
        for i, x_i in enumerate(x):
            legendre_pp[i] = sc.misc.derivative(legendre_p, x_i)
    else:
        legendre_pp = sc.misc.derivative(legendre_p, x)
    return legendre_pp
