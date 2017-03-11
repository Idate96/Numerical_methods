"""Setup of 1D Poisson problem"""
from grid import Grid
import math
from poly_interpolation import PolyBasis
from stiffness_matrix import inner_product
from functools import partial
import integration
from scipy import integrate
from incidence_matrices import incidence_m1d
import matplotlib.pyplot as plt
import numpy as np


def original_f(x):
    # solution to current poisson problem
    return np.sin(2 * np.pi * x)


def rhs_sin(x):
    # 2nd derivative of orginal_f
    f_i = -4 * np.pi ** 2 * np.sin(2 * np.pi * x)
    return f_i


def poisson_homo(func, a, b, n, m, f_exact=None):
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
    print('NEW CASE : N = ' + str(n))
    # init underlying structures
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
    x = np.linspace(a, b, m)
    basis = PolyBasis(x, grid)

    # lhs 1-edge inner product (nxn)
    M_1 = -inner_product(basis, degree=1)
    # print('Eigenvalues M_1: ', np.linalg.eig(M_1)[0])

    # incidence matrix E_01 (n+1xn)
    E_10 = np.transpose(incidence_m1d(n))
    # print('Incidence matrix E_10 : \n', E_10)
    # stiffness_matrix
    N = np.transpose(E_10) @ M_1 @ E_10
    # enforce boundary conditions
    N[0, 0] = N[-1, -1] = 1
    N[0, 1:] = N[-1, :-1] = 0
    # print('Matrix M_1 : \n', M_1)
    # print('Matrix N :\n', N)

    # rhs
    M_0 = inner_product(basis, degree=0)
    # print('Matrix M_0 : \n', M_0)
    f = func(grid.nodal_pts)
    # print('Function at nodal pts :\n', f)
    rhs = np.transpose(f) @ M_0
    # boudary condtions
    rhs[0] = rhs[-1] = 0
    # print('RHS : ', rhs)

    # print('Determinant N: ', np.linalg.det(N))
    # print('Eigenvalues N: ', np.linalg.eig(N)[0])
    # solve linear system
    c = np.linalg.inv(N) @ rhs

    if f_exact is not None:
        error_norm = l2_error_norm(c, basis.lagrange, f_exact, x)
    else:
        error_norm = 0
    # print('The coefficient matrix is : \n', c)

    phi = np.transpose(c) @ basis.lagrange(x)
    # lag_base = basis.lagrange(x)
    # print('Lagrange basis : \n', lag_base)  # the basis seems fine
    # phi = 0
    # for i in range(n):
    #     phi += c[i] * basis.basis[i]

    # alternative to previous loop
    # c = c.reshape((n+1, 1))
    # phi = np.sum(c * lag_base, axis=0)
    # phi = 0
    return phi, c, error_norm


def plot_solution(phi, f_exact, x, *args, save=False, show=True):
    """Plot exact and FEM solutions

    Args:
        f_exact (obj func)= exact solution function.
        phi (np.array) = FEM solution.
        x (np.array) = domain.
    Returns:
        error_max(float) = max error."""
    plt.figure(1)
    error = abs(f_exact(x) - phi)
    plt.plot(x, f_exact(x), '-', label='Original function')
    plt.plot(x, phi, '-', label='FEM solution')
    plt.plot(x, error, '-', label='error')
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend()
    if args:
        plt.title(args[0] + '_N_' + args[1])
    error_max = np.max(error)
    if save:
        plt.savefig('images/' + args[0] + '_solution/' + args[0] +
                    '_N_' + args[1] + '.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()
    return error_max


def add_square(x, c, basis_func, f_exact):
    # print("x give to add ", x)
    # print("Type basis function :", type(basis_func(x)))
    # print("Type coeff")
    # # sumi = np.sum(c * basis_func(x).reshape(1, np.size(c)))
    # print("Basis functions at " + str(x) + " : ", basis_func(x))
    # print("Shape basis func : ", np.shape(basis_func(x)))
    # print("f exact", f_exact(x))
    # print("Exact function at " + str(x) + " : " +  str(f_exact(x)))
    # # print("Sum of basis functions : ", sumi)
    # print("Exis add square func")
    value_basis = basis_func(x)
    # print("VALUE BASIS ", value_basis)
    error_sq = (c @ value_basis - f_exact(x)) ** 2
    # print("error sq at " + str(x) + " : ", error_sq)
    return error_sq[0]


def l2_error_norm(c, basis_func, f_exact, x):
    coeff = c.reshape(1, np.size(c))
    square_error = partial(add_square, c=coeff, basis_func=basis_func, f_exact=f_exact)
    # error_func = square_error(0.5)
    # print("Local error thrugh partial: ", error_func)

    # n_glob = math.ceil((np.size(c) ** 2 + 1) / 2)
    # integral = integration.quad_glob([square_error], x[0], x[-1], n_glob)

    integral = integrate.quad(square_error, x[0], x[-1])[0]
    return integral ** (0.5)


def plot_convergence(error_norm, n_0):
    """Plot convergence rate
    Args:
        error_norm (np.array) = array containing error norm for incresing n
        n_0 (int) = num dof + 1 for the first norm
    """
    n = np.arange(n_0, np.size(error_norm) + n_0)
    linear_conv = np.ones((np.size(error_norm)))
    for i, el in enumerate(linear_conv):
        linear_conv[i] = el / 10 ** i
    print(n.shape)
    print(error_norm.shape)
    plt.plot(n, np.log(linear_conv))
    plt.plot(n, np.log(error_norm))
    plt.show()
