"""Setup of 1D Poisson problem"""
from grid import Grid
from Interpolation import PolyBasis
from stiffness_matrix import inner_product
from index_matrices import incidence_m1d
import numpy as np


def original_f(x):
    # solution to current poisson problem
    return np.sin(2 * np.pi * x)


def rhs_sin(x):
    # 2nd derivative of orginal_f
    f_i = -4 * np.pi ** 2 * np.sin(2 * np.pi * x)
    return f_i


def poisson_homo(func, a, b, n, m):
    """1D poisson PDE with Dirichlet BC.

    Calculate the FEM solution to the poisson problem with sinusoidal forcing.
    Args:
        func (obj function) = forcing function
        a (float) = start domain
        b (float) = end domain
        n (int) = number of nodal point - 1
        m (int) = resolution interval

    Return:
        phi(np.array) = FEM solution
    """
    # init underlying structures
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
    x = np.linspace(a, b, m)
    basis = PolyBasis(x, grid)

    # lhs 1-edge inner product (nxn)
    M_1 = inner_product(basis, degree=1)
    # incidence matrix E_01 (n+1xn)
    E_10 = np.transpose(incidence_m1d(n))
    # stiffness_matrix
    N = np.transpose(E_10) @ M_1 @ E_10
    print('Matrix M_1 : \n', M_1)
    print('Matrix N :\n', N)

    # rhs
    M_0 = inner_product(basis, degree=0)
    print('Matrix M_0 : \n', M_0)
    f = func(grid.nodal_pts)
    print('Function at nodal pts :\n', f)
    rhs = np.transpose(f) @ M_0
    print('RHS : ', rhs)

    # solve linear system
    c =  rhs @ np.linalg.inv(N)
    print('The coefficient matrix is : \n', c)

    # phi = np.transpose(c) @ basis.lagrange(x)
    lag_base = basis.lagrange(x)
    print('Lagrange basis : \n', lag_base)  # the basis seems fine
    phi = 0
    for i in range(n):
        phi += c[i] * basis.basis[i]

    #alternative to previous loop
    # c = c.reshape((n+1, 1))
    # phi = np.sum(c * lag_base, axis=0)
    return phi
