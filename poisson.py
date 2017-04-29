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
import pdb


def original_f_poly4(x):
    # solution to current poisson problem
    # return np.sin(2 * np.pi * x)
    return x ** 2 - x ** 4


def rhs_poly4(x):
    # 2nd derivative of orginal_f
    return -(12 * x ** 2 - 2)


def original_f_sin2(x):
    return np.sin(2 * np.pi * x)


def original_f_sin2_shifted(x):
    return np.sin(2 * np.pi * x + np.pi / 2)


def rhs_sin2_shifted(x):
    f_i = -4 * np.pi ** 2 * np.sin(2 * np.pi * x + np.pi / 2)
    return f_i


def rhs_sin2(x):
    f_i = -4 * np.pi ** 2 * np.sin(2 * np.pi * x)
    return f_i


def populate_index_matrix(h, p):
    index_matrix = np.zeros((p + 1, h))
    n = 1
    for j in range(h):
        n -= 1
        for i in range(p + 1):
            index_matrix[i, j] = int(n)
            n += 1
    print('index matrix former ', index_matrix)
    return index_matrix.astype(int)


def populate_mixed_form(M_0, M_1, f, p):
    A = np.zeros((2 * (p + 1) - 1, 2 * (p + 1) - 1))
    # Boundary operator
    E_01 = incidence_m1d(p)
    N_1 = E_01 @ M_1
    assert np.shape(N_1) == (p + 1, p), \
        "wrong dim for N : {0}" .format(np.shape(N_1))
    A[:p + 1, :p + 1] = M_0
    A[:p + 1, p + 1:] = N_1
    A[p + 1:, :p + 1] = np.transpose(N_1)
    b = np.zeros((2 * (p + 1) - 1))
    b[p + 1:] = f @ M_1
    return A, b


def mixed_hp_poisson(h, p, a, b, m, f_rhs):


def mixed_poisson_element(p, a, b, m, f_rhs):
    grid = Grid(a, b, p)
    grid.gauss_lobatto()
    basis = PolyBasis(grid.nodal_pts, grid)

    M_1 = inner_product(basis, degree=1, a=a, b=b)

    M_0 = inner_product(basis, degree=0, a=a, b=b)

    E_01 = incidence_m1d(p)
    N_1 = E_01 @ M_1

    f = np.zeros((p))
    print(grid.nodal_pts)
    for i in range(p):
        f[i] = integrate.quad(f_rhs, grid.nodal_pts[i], grid.nodal_pts[i + 1])[0]

    return M_0, N_1, f, basis


def mixed_poisson(p, a, b, m, f_rhs):
    grid = Grid(a, b, p)
    grid.gauss_lobatto()
    basis = PolyBasis(grid.nodal_pts, grid)

    M_1 = inner_product(basis, degree=1, a=a, b=b)

    M_0 = inner_product(basis, degree=0, a=a, b=b)

    f = np.zeros((p))
    print(grid.nodal_pts)
    for i in range(p):
        f[i] = integrate.quad(f_rhs, grid.nodal_pts[i], grid.nodal_pts[i + 1])[0]

    A, rhs = populate_mixed_form(M_0, M_1, f, p)
    c = np.linalg.solve(A, rhs)
    sigma = c[:p + 1]
    phi = c[p + 1:]
    print("a {0} \n b {1} \n m {2}" .format(a, b, m))

    x = np.linspace(a, b, m)
    print("shape lagrange {0} \n shape edge {1}" .format(np.shape(basis.lagrange(x)),
                                                         np.shape(basis.edge(x))))
    print('basis edge ', basis.edge())
    phi_func = phi @ basis.edge(x)[1:, :]
    sigma_func = sigma @ basis.lagrange(x)
    print(rhs)
    return phi_func, sigma_func, x


def reverse_engineer(h, p, m, func, a, b, f_exact):
    index_matrix = populate_index_matrix(h, p)
    a_list = [a + (b - a) / h * n for n in range(h)]
    b_list = a_list[1:]
    b_list.append(b)

    N_list = []
    M_list = []
    f_list = []
    basis_functions = []
    basis_list = []
    nodal_pts = []
    errors = np.zeros((h))
    for i in range(h):
        N_el, M_0_el, f_el, basis = poisson_element(func, a_list[i], b_list[i], p, m)
        basis_functions.append(basis)
        N_list.append(N_el)
        M_list.append(M_0_el)
        f_list.append(f_el)
        basis_list.append(basis.basis)
        nodal_pts.append(basis.nodal_pts)

    N_gen = assemble(N_list, index_matrix)
    M_gen = assemble(M_list, index_matrix)

    nodal_pts_gen = nodal_pts[0]
    f_gen = f_list[0]
    for i in range(1, len(f_list)):
        f_gen = np.hstack((f_gen, f_list[i][1:]))
        nodal_pts_gen = np.hstack((nodal_pts_gen, nodal_pts[i][1:]))
    print('Nodal pts general ', nodal_pts_gen)
    print('f_gen Shape ', np.shape(f_gen))
    print('Exact rhs ', rhs_sin2(np.asarray(a_list)))
    print('f_gen 2 ', f_gen)
    rhs = M_gen @ f_gen

    # Neumann BCs
    N_gen[0, 0] = N_gen[-1, -1] = 1
    N_gen[0, 1:] = N_gen[-1, :-1] = 0
    print('det M gen : ', np.linalg.det(M_gen))
    print('det N gen : ', np.linalg.det(N_gen))
    print('M_Gen ---------------\n', M_gen)
    print('N_Gen ---------------\n', N_gen)
    print('rhs --------------\n', rhs)
    # rhs[0] = rhs[-1] = 0
    rhs[0] = f_exact[a_list[0]]
    rhs[1] = f_exact[b_list[-1]]
    print('rhs \n', rhs)
    difference = N_gen @ f_exact(nodal_pts_gen) - rhs
    print("DIFFERENCE LHS - RHS\n", difference)
    c = np.linalg.solve(N_gen, rhs)
    print("DIFFERENCE C - EXACT SOL", c - f_exact(nodal_pts_gen))
    return difference


def hp_poisson(h, p, m, func, a, b, f_exact=None):
    index_matrix = populate_index_matrix(h, p)
    # list of start points for the elements
    a_list = [(a + (b - a) / h * n) for n in range(h)]
    # list of the end pts for the elements
    b_list = a_list[1:]
    b_list.append(b)

    N_list = []
    M_list = []
    f_list = []
    basis_functions = []
    basis_list = []
    errors = np.zeros((h))

    # generate all matrices on the elements
    for i in range(h):
        N_el, M_0_el, f_el, basis = poisson_element(func, a_list[i], b_list[i], p, m)
        basis_functions.append(basis)
        N_list.append(N_el)
        M_list.append(M_0_el)
        f_list.append(f_el)
        basis_list.append(basis.basis)

    N_gen = assemble(N_list, index_matrix)
    print("HELLO")
    print('shape N-Gen : ', np.shape(N_gen))
    M_gen = assemble(M_list, index_matrix)
    print('shape M-Gen : ', np.shape(M_gen))

    # add value of the first element to the list
    f_gen = f_list[0]
    print("f_gen ", f_gen)

    # continue for the elements 1 onword
    for i in range(1, len(f_list)):
        # if I put *1 it resolves the equation better
        f_gen = np.hstack((f_gen, f_list[i][1:]))
    print('f_gen Shape ', np.shape(f_gen))
    print('Exact rhs ', rhs_sin2(np.asarray(a_list)))
    print('f_gen 2 ', f_gen)
    rhs = M_gen @ f_gen

    # Neumann BCs
    N_gen[0, 0] = N_gen[-1, -1] = 1
    N_gen[0, 1:] = N_gen[-1, :-1] = 0
    print('det M gen : ', np.linalg.det(M_gen))
    print('det N gen : ', np.linalg.det(N_gen))
    print('M_Gen ---------------\n', M_gen)
    print('N_Gen ---------------\n', N_gen)
    print('rhs --------------\n', rhs)
    if f_exact != None:
        rhs[0] = f_exact(a_list[0])
        rhs[-1] = f_exact(b_list[-1])
    else:
        rhs[0] = rhs[0 - 1] = 0
    print('rhs \n', rhs)

    function = np.ones((np.size(basis_list[0][p, :]) - 1) * h + 1)
    print('function size : ', np.shape(function))
    # Coefficients
    c = np.linalg.solve(N_gen, rhs)
    n = 1
    j = 1

    for el in range(h):
        print("-------el  = ", el)
        print("DIM basis_list[el] ", np.shape(basis_list[el]))
        print("DIM c[0:p*el-1]", np.shape(c[p * el: p * (el + 1) + 1]))
        print("DIM func[el*m + 1:(el+1)*m]", np.shape(function[el * m: (el + 1) * m - j]))
        errors[el] = l2_error_norm(c[p * el:p * (el + 1) + 1],
                                   basis_functions[el].lagrange, f_exact, (a_list[el], b_list[el]))
        partial = c[p * el: p * (el + 1) + 1] @ basis_list[el]
        print("DIM partial ", partial)
        if el == h - 1:
            function[el * (m - 1):] = partial
        else:
            function[el * (m - 1): (m - 1) * (el + 1)] = partial[:-1]
    #
    #
    # for i in range(h):
    #     n -= 1
    #     x = np.linspace(a_list[i], b_list[i], m)
    #     print('ELEMENT ', i)
    #     print('general c ', np.shape(c))
    #     print('shape c', np.shape(c[((p + 1) * i - j):((p + 1) * (i + 1) - j)]))
    #     print('c[{0}:{1}]' .format(p * i, (p + 1) * i))
    #     errors[i] = l2_error_norm(c[p * i:p * (i + 1) + 1],
    #                               basis_functions[i].lagrange, f_exact, x)
    #     for j in range(p + 1):
    #         # print('list lenght ', len(basis_list))
    #         print('coefficients c[{0}] : {1}\n' .format(n, c[n]))
    #         print('basis \n', basis_list[i][j, :])
    #         # print('shape c :', np.size(c))
    #         # print('CN4 :', c[4])
    #         # print('basis 4-0 :', basis_list[0][4, :])
    #         # print('basis 4-1 :', basis_list[1][0, :])
    #         function[m * i: m * (i + 1)] += c[n] * basis_list[i][j, :]
    #         n += 1
    #     j += 1
    x = np.linspace(a, b, np.size(function))
    return function, x, np.sum(errors)


def assemble(matrix_list, index_matrix):
    print('matrix list 0', matrix_list[0])
    p = np.shape(matrix_list[0])[0]
    dim = index_matrix[-1, -1] + 1
    print('dim ', dim)
    matrix_gen = np.zeros((dim, dim))
    for i in range(np.shape(index_matrix)[1]):
        print('shape matrixgen', np.shape(matrix_gen[index_matrix[:, i], index_matrix[:, i]]))

        print('shape matrix list', np.shape(matrix_list[i]))
        i_0, i_f = index_matrix[0, i], index_matrix[-1, i] + 1
        print('index matrix [:,i] \n', index_matrix[:, i])

        print('matrixgen \n', matrix_gen[i_0: i_f, i_0: i_f])

        matrix_gen[i_0: i_f, i_0: i_f] += matrix_list[i]
    return matrix_gen


def poisson_element(func, a, b, n, m, f_exact=None):
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
    print("nodal values : ", grid.nodal_pts)
    x = np.linspace(a, b, m)
    basis = PolyBasis(grid.nodal_pts, grid)
    M_1 = -inner_product(basis, degree=1, a=a, b=b)
    E_10 = np.transpose(incidence_m1d(n))
    N = np.transpose(E_10) @ M_1 @ E_10
    M_0 = inner_product(basis, degree=0, a=a, b=b)
    f = func(grid.nodal_pts)
    basis.lagrange(x)
    return N, M_0, f, basis


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
    print("nodal values : ", grid.nodal_pts)
    x = np.linspace(a, b, m)
    basis = PolyBasis(x, grid)

    # lhs 1-edge inner product (nxn)
    M_1 = -inner_product(basis, degree=1, a=a, b=b)
    # print('Eigenvalues M_1: ', np.linalg.eig(M_1)[0])

    # incidence matrix E_01 (n+1xn)
    E_10 = np.transpose(incidence_m1d(n))
    # print('Incidence matrix E_10 : \n', E_10)
    # stiffness_matrix
    N = np.transpose(E_10) @ M_1 @ E_10
    # enforce boundary conditions
    N[0, 0] = N[-1, -1] = 1
    N[0, 1:] = N[-1, : -1] = 0
    # print('Matrix M_1 : \n', M_1)
    # print('Matrix N :\n', N)

    # rhs
    M_0 = inner_product(basis, degree=0, a=a, b=b)
    print('M_0 ------- \n', M_0)
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
    print('N ------------ \n', N)

    if f_exact is not None:
        error_norm = l2_error_norm(c, basis.lagrange, f_exact, x)
    else:
        error_norm = 0
    # print('The coefficient matrix is : \n', c)
    # c = np.ones((len(grid.nodal_pts)))
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
        plt.title(args[0] + '_p_' + args[1])
    error_max = np.max(error)
    if save:
        plt.savefig('images/' + args[0] + '_solution/' + 'h' + args[1] +
                    '-p' + args[2] + '.png', bbox_inches='tight')
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
    print("VALUE BASIS ", value_basis)
    print("C Value ", c)
    error_sq = (c @ value_basis - f_exact(x)) ** 2
    # print("error sq at " + str(x) + " : ", error_sq)
    return error_sq[0]


def l2_error_norm(c, basis_func, f_exact, x=(-1, 1)):
    coeff = c.reshape(1, np.size(c))
    square_error = partial(add_square, c=coeff, basis_func=basis_func, f_exact=f_exact)
    # error_func = square_error(0.5)
    # print("Local error thrugh partial: ", error_func)

    # n_glob = math.ceil((np.size(c) ** 2 + 1) / 2)
    # integral = integration.quad_glob([square_error], x[0], x[-1], n_glob)

    integral = integrate.quad(square_error, x[0], x[-1])[0]
    return integral ** (0.5)


def plot_convergence(error, *args, save=False, show=True):
    """Plot convergence rate
    Args:
        error_norm (np.array) = array containing error norm for incresing n
        n_0 (int) = num dof + 1 for the first norm
    """
    fig = plt.figure
    x, error_norm = zip(*error)
    print('X ', x)
    x = [el ** (-1) for el in x]
    print('x ', x)
    print('error ', error_norm)
    # n = np.arange(n_0, np.size(error_norm) + n_0)
    # x = np.arange(10)

    # print(error_norm.shape)
    if args[0] == 'p':
        h = args[1]
        linear_conv = np.ones((np.size(x)))
        for i, xi in enumerate(x):
            linear_conv[i] = 1 / 10 ** i
        plt.plot(x, np.log10(linear_conv), label='Gradient 1')
        plt.plot(x, np.log10(error_norm), '-o', label='Error norm')
        plt.ylabel(r'log(ϵ)')
        plt.title('p error convergence rate, h = ' + str(args[1]))
    if args[0] == 'h':
        p = args[1]
        p_convergence = np.zeros((np.size(x)))
        for i, xi in enumerate(x):
            p_convergence[i] = xi ** (p + 1)
        # try:
        #     plt.plot(x, np.log10(error_norm), '-o', label='Error norm')
        # except ZeroDivisionError:
        #     plt.plot(x, error_norm, '-o', label='Error norm')
        print(np.any(error_norm) == 0)
        if np.any(error_norm) == 0:
            print("Some zeros in the error_norm")
            plt.plot(x, error_norm, '-o', label=r'Error_norm')
        else:
            plt.plot(np.log10(x), np.log10(error_norm), '-o', label=r'L2 error norm')
        plt.plot(np.log10(x), np.log10(p_convergence), label=r'$h^{(p+1)}$')
        plt.ylabel(r'log(ϵ)')
        plt.title('h error convergence rate, p = ' + str(args[1]))

    plt.xlabel(args[0])
    plt.legend()
    if save:
        plt.savefig('images/convergence' + args[0] + '.png', bbox_inches='tight')
    if show:
        plt.show()
