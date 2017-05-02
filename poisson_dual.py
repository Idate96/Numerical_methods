from grid import Grid
import poisson
import math
import itertools
from poly_interpolation import PolyBasis
from stiffness_matrix import inner_product
from functools import partial
import integration
from scipy import integrate
from incidence_matrices import incidence_m1d
import matplotlib.pyplot as plt
import numpy as np
import pdb


def hodge_01(primal_basis, dual_mesh):
    hodge = primal_basis.edge(dual_mesh[1:-1])[1:]
    return hodge


def dual_poisson_element(p, a, b, m, f_rhs):

    dual_grid = Grid(a, b, p)
    primal_grid = Grid(a, b, p)
    dual_mesh = dual_grid.dual_central()
    primal_mesh = primal_grid.gauss_lobatto()

    x = np.linspace(a, b, m)

    primal_basis = PolyBasis(x, primal_grid)
    dual_basis = PolyBasis(x, dual_grid)

    # inner product dual 0 forms
    M_0_dual = inner_product(dual_basis, degree=0)[1:-1, 1:-1]
    M_1_dual = inner_product(dual_basis, degree=1)
    # print(M_0_dual[1:-1, 1:-1])

    E_10 = np.transpose(incidence_m1d(p))
    E_10_dual = incidence_m1d(p)

    H_01 = hodge_01(primal_basis, dual_mesh)

    # upper diagonal
    U_d = -M_0_dual @ H_01 @ E_10
    print("Ud :", U_d)

    # lower diagonal
    L_d = M_1_dual @ E_10_dual
    print("E_01 ", E_10)
    print("E_01_dual ", E_10_dual)
    # L_d[0, 0] = L_d[-1, -1] = 1
    # L_d[0, 1:] = L_d[-1, :-1] = 0

    A = np.zeros((np.shape(M_0_dual)[0] + np.shape(L_d)[0],
                  np.shape(M_0_dual)[1] + np.shape(U_d)[1]))

    A[:np.shape(M_0_dual)[0], :np.shape(M_0_dual)[1]] = M_0_dual
    A[:np.shape(U_d)[0], np.shape(M_0_dual)[1]:] = U_d
    A[np.shape(M_0_dual)[0]:, :np.shape(L_d)[1]] = L_d

    f_projected = poisson.projection_1form(dual_mesh, f_rhs)
    b = f_projected @ M_1_dual
    rhs = np.zeros((np.shape(A)[0]))
    rhs[np.shape(rhs)[0] - np.shape(b)[0]:] = b
    poisson.plot_matrix(M_1_dual)
    print("det A :", np.linalg.det(A))
    print(A)
    print(rhs)
    print(np.shape(A))


def direct_dual_poisson_hp(h, p, a, b, m, f_rhs, f_exact=None):
    index_matrix = poisson.populate_index_matrix(h, p)
    # list of start points for the elements
    a_list = [(a + (b - a) / h * n) for n in range(h)]
    # list of the end pts for the elements
    b_list = a_list[1:]
    b_list.append(b)
    print("a list", a_list)
    print("b_list ", b_list)
    laplace_list = []
    f_list = []
    primal_basis_function_list = []
    primal_basis_list = []
    dual_meshes = []

    errors = np.zeros((h))

    for i in range(h):
        laplace, f, primal_basis_func, primal_basis, dual_mesh = direct_dual_poisson_element(
            p, a_list[i], b_list[i], m, f_rhs, f_exact)
        laplace_list.append(laplace)
        f_list.append(f)
        primal_basis_function_list.append(primal_basis_func)
        primal_basis_list.append(primal_basis)
        dual_meshes.append(dual_mesh)

    A = poisson.assemble(laplace_list, index_matrix)

    # add value of the first element to the list
    f_gen = f_list[0]

    # continue for the elements 1 onword
    for i in range(1, len(f_list)):
        # if I put *1 it resolves the equation better
        f_gen[-1] += f_list[i][0]
        f_gen = np.hstack((f_gen, f_list[i][1:]))
    # BCs Homogenous
    A[0, 0] = A[-1, 1] = 1
    A[0, 1:] = A[-1, :-1] = 0

    if f_exact != None:
        f_gen[0] = f_exact(a_list[0])
        f_gen[-1] = f_exact(b_list[-1])
    else:
        f_gen[0] = f_gen[-1] = 0

    c = np.linalg.solve(A, f_gen)

    function = np.ones((np.size(primal_basis_list[0][p, :]) - 1) * h + 1)

    for el in range(h):
        errors[el] = poisson.l2_error_norm(c[p * el:p * (el + 1) + 1],
                                           primal_basis_function_list[el].lagrange, f_exact, (a_list[el], b_list[el]))
        partial = c[p * el: p * (el + 1) + 1] @ primal_basis_list[el]

        if el == h - 1:
            function[el * (m - 1):] = partial
        else:
            function[el * (m - 1): (m - 1) * (el + 1)] = partial[:-1]

    x = np.linspace(a, b, np.size(function))

    return function, x, np.sum(errors)


def direct_dual_poisson_element(p, a, b, m, f_rhs, f_exact=None, uniform=True):
    dual_grid = Grid(a, b, p)
    primal_grid = Grid(a, b, p)
    dual_mesh = dual_grid.dual_central()
    primal_mesh = primal_grid.gauss_lobatto()

    x = np.linspace(a, b, m)

    primal_basis = PolyBasis(x, primal_grid)
    dual_basis = PolyBasis(x, dual_grid)

    H_01 = hodge_01(primal_basis, dual_mesh)
    E_10 = np.transpose(incidence_m1d(p))
    E_10_dual = incidence_m1d(p)

    A = - E_10_dual @ H_01 @ E_10

    f = poisson.projection_1form(dual_mesh, f_rhs)

    x = np.linspace(a, b, m)

    return A, f, primal_basis, primal_basis.lagrange(x), dual_mesh


def direct_dual_poisson(p, a, b, m, f_rhs):
    dual_grid = Grid(a, b, p)
    primal_grid = Grid(a, b, p)
    dual_mesh = dual_grid.dual_central()
    primal_mesh = primal_grid.gauss_lobatto()

    x = np.linspace(a, b, m)

    primal_basis = PolyBasis(x, primal_grid)
    dual_basis = PolyBasis(x, dual_grid)

    H_01 = hodge_01(primal_basis, dual_mesh)

    E_10 = np.transpose(incidence_m1d(p))
    E_10_dual = incidence_m1d(p)

    f = poisson.projection_1form(dual_mesh, f_rhs)
    f[0] = f[-1] = 0
    print(f)
    A = - E_10_dual @ H_01 @ E_10
    A[0, 0] = A[-1, -1] = 1
    A[0, 1:] = A[-1, : -1] = 0
    print(np.linalg.det(A))
    c = np.linalg.solve(A, f)

    function = c @ primal_basis.lagrange(x)

    return function, x

if __name__ == '__main__':
    # dual_grid = Grid(-1, 1, 4)
    # primal_grid = Grid(-1, 1, 4)
    # x = np.linspace(-1, 1, 151)
    # dual_mesh = dual_grid.dual_central()
    # primal_mesh = primal_grid.gauss_lobatto()
    # primal_basis = PolyBasis(x, primal_grid)
    # e_ij = primal_basis.edge(dual_mesh[1:-1])[1:]
    # print(e_ij)
    # dual_poisson_element(2, -1, 1, 50, poisson.rhs_sin2)
    solution, x, error = direct_dual_poisson_hp(6,
                                                8, 0, 0.5, 150, poisson.rhs_sin2, poisson.original_f_sin2)
    print(error)
    poisson.plot_solution(solution, poisson.original_f_sin2, x)
