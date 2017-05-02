"""Setup of 1D Poisson problem"""
from grid import Grid
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


def d_original_f_sin2_dx(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)


def rhs_sin2(x):
    f_i = -4 * np.pi ** 2 * np.sin(2 * np.pi * x)
    return f_i


def projection_1form(nodal_pts, f_rhs):
    """Project a 0 form into a 1 form."""
    p = np.size(nodal_pts) - 1
    f = np.zeros((p))
    for i in range(p):
        f[i], error = integrate.quad(f_rhs, nodal_pts[i], nodal_pts[i + 1])
    print("Error quadrature :", error)
    return f


def plot_matrix(matrix):
    """Plot the entries of a matrix.

    Args:
    matrix (np.ndarray) = array to be plotted
    """
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()


def populate_index_matrix(h, p):
    """Build the index matrix for FEM.

    Args:
        h (int) = number of elements
        p (int) = number of subintervals in the element dictated by the nodes

    Returns:
        index_matrix (np.ndarray) = index_matrix[i,j] reppresent the node i of the element j, very useful to assemble the general matrices of the inner product
    """
    # initialize
    index_matrix = np.zeros((p + 1, h))
    n = 1
    # filling with the global indeces, a node shared by multiple elements have the same index
    for j in range(h):
        n -= 1
        for i in range(p + 1):
            index_matrix[i, j] = int(n)
            n += 1
    return index_matrix.astype(int)


def populate_mixed_form(M_0, M_1, f, p):
    """Populate the genearal matrix of the mixed formulation. Only for 1 element.

    Args:
        M_0 (np.array) = Inner product matrix of 0-basis forms
        M_1 (np.array) = Inner product matrix of 1-basis forms
        f (np.array) = f[i] corresponds to \int_[i,i+1] f dx where
        f is the forcing function of the system.
        p (int) = # dof for 1-forms

    Returns:
        A (np.array) = lhs of m.f.
        b (np.array) = rhs of m.f.
         """
    # Build mixed form matrix A = [[M, D], [D^T, 0]]
    A = np.zeros((2 * (p + 1) - 1, 2 * (p + 1) - 1))
    # Boundary operator
    E_01 = incidence_m1d(p)
    # p+1 x p matrix
    N_1 = E_01 @ M_1
    assert np.shape(N_1) == (p + 1, p), \
        "wrong dim for N : {0}" .format(np.shape(N_1))
    A[:p + 1, :p + 1] = M_0
    A[:p + 1, p + 1:] = N_1
    A[p + 1:, :p + 1] = np.transpose(N_1)
    # build rhs of the formulation rhs = [0, M@f]
    b = np.zeros((2 * (p + 1) - 1))
    b[p + 1:] = M_1 @ f
    return A, b


def assemble_mixed_N(N_list):
    """Assemble N matrix.

    Args:
        N_list (list) = list of N matrices N_list[i] is N matrix of the ith element
    """
    # Assemble N matrices for multiple elements
    p = np.shape(N_list[0])[1]
    h = len(N_list)
    # init general matrix (use plot matrix to see shape)
    N_gen = np.zeros((h * p + 1, h * p))
    for i, N in enumerate(N_list):
        # TODO: use index matrix to build it up
        N_gen[(p + 1) * i - i:(p + 1) * (i + 1) - i, p * i:p * (i + 1)] = N
    # plot_matrix(N_gen)
    return N_gen


def assemble_M_1(M_1_list):
    """Assembly of inner product between 1-forms

    Args:
        M_1_list (list) = list of M_1 matrices M_1_list[i] is M_1 matrix of the ith element

    Returns:
        M_1_gen (np.array) = general matrix of 1-forms inner products.
    """
    p = np.shape(M_1_list[0])[0]
    h = len(M_1_list)
    # init shape
    M_1_gen = np.zeros((h * p, h * p))

    for i, M in enumerate(M_1_list):
        M_1_gen[p * i:p * (i + 1), p * i:p * (i + 1)] = M
    # plot_matrix(M_1_gen)
    return M_1_gen


def mixed_hp_poisson(h, p, a, b, m, f_rhs, f_exact=None):
    """Mixed FEM formulation of the poisson problem in 1D.

    Args:
        h (int) = # of elements
        p (int) = # of edges in one elements
        a (float) = start pt of domain
        b (float) = end pt of domain
        m (int) = resolution of plotting
        f_rhs (obj.func) = forcing function
        f_exact (obj.func) = exact solution

    Returns:
        phi (np.array) = 1-form solution of poisson problem
        sigma (np.array) = dual-0-form of the solution (d^*phi)
        error_phi (float) = L2 error of FEM - exact solution
    """
    index_matrix = populate_index_matrix(h, p)
    # list of start points for the elements
    a_list = [(a + (b - a) / h * n) for n in range(h)]
    # list of the end pts for the elements
    b_list = a_list[1:]
    b_list.append(b)

    # init all necessary lists
    N_list = []
    M_0_list = []
    M_1_list = []
    f_list = []
    basis_functions = []
    basis_0_list = []
    basis_1_list = []
    errors_sigma = np.zeros((h))
    errors_phi = np.zeros((h))

    # generate all matrices on the elements
    for i in range(h):
        M_0_el, N_el, M_1_el, f_el, basis_func, basis = mixed_poisson_element(p, a_list[i], b_list[
                                                                              i], m, f_rhs)
        basis_functions.append(basis_func)
        N_list.append(N_el)
        M_0_list.append(M_0_el)
        M_1_list.append(M_1_el)
        f_list.append(f_el)
        basis_0_list.append(basis[0])
        basis_1_list.append(basis[1])
    # assemble 0-forms inner product
    M_0_gen = assemble(M_0_list, index_matrix)
    # plot_matrix(M_0_gen)

    # assemble 1-forms inner product @ E
    N_gen = assemble_mixed_N(N_list)
    # assemble 1-forms inner product
    M_1_gen = assemble_M_1(M_1_list)

    # assemble function in one array
    f = np.asarray(list(itertools.chain.from_iterable(f_list)))

    rhs_partial = f @ M_1_gen
    rhs = np.zeros((h * (2 * p) + 1))
    rhs[h * p + 1:] = rhs_partial

    # plot_matrix(M_1_gen)

    # assemble lsh of the formulation
    A = np.zeros((2 * h * p + 1, 2 * h * p + 1))
    # upper left
    A[:h * p + 1, :h * p + 1] = M_0_gen
    # upper right
    A[:h * p + 1, h * p + 1:] = N_gen
    # lower left
    A[h * p + 1:, :h * p + 1] = np.transpose(N_gen)

    # solve for the cochain coefficients
    c = np.linalg.solve(A, rhs)
    # coefficients relative to sigma
    sigma_c = c[:h * p + 1]
    # coefficients relative to phi
    phi_c = c[h * p + 1:]

    # init solutions
    sigma = np.ones((np.size(basis_0_list[0][p, :]) - 1) * h + 1)
    phi = np.ones((np.size(basis_1_list[0][p, :]) * h))

    # Interpolation
    for el in range(h):
        # if f_exact provided calculate L2 error
        if f_exact:

            phi_c_error = np.hstack((0, phi_c[p * el: p * (el + 1)]))
            errors_phi[el] = l2_error_norm(phi_c_error,
                                           basis_functions[el].edge, f_exact,  (a_list[el], b_list[el]))

        # to avoid storing twice the shared nodes
        partial_sigma = sigma_c[p * el: p * (el + 1) + 1] @ basis_0_list[el]
        # phi interpolation (note basis_1[0,:] is an array of zeros)
        phi[el * m:(el + 1) * m] = phi_c[p * el: p * (el + 1)] @ basis_1_list[el][1:]
        if el == h - 1:
            sigma[el * (m - 1):] = partial_sigma
        else:
            sigma[el * (m - 1): (m - 1) * (el + 1)] = partial_sigma[:-1]

    return phi, sigma, np.sum(errors_phi)


def mixed_poisson_element(p, a, b, m, f_rhs):
    """Evaluate relevant matrices in the element

    Args:
        p (int) = # of edges
        a (float) = start of the element
        b (float) = end of the element
        m (int) = resolution of the basis functions
        f_rhs (obj.func) = forcing function

    Returns:
        M_0 (np.array) = Inner product 0-forms
        M_1 (np.array) = Inner product 1-forms
        N_1 (np.array) = E_01 @ M_1
        f (np.array) = coefficients of projection of forcing function onto 1-forms
        basis (obj.Polybasis) = Basis for the element
        (basis_0, basis_1) (tuple of arrays) = explicit evaluation of basis function of the element
    """
    # init grid
    grid = Grid(a, b, p)
    # use Gauss lobatto nodes
    grid.gauss_lobatto()
    # init basis
    basis = PolyBasis(grid.nodal_pts, grid)
    x = np.linspace(a, b, m)

    M_1 = inner_product(basis, degree=1, a=a, b=b)

    M_0 = inner_product(basis, degree=0, a=a, b=b)

    E_01 = incidence_m1d(p)
    N_1 = E_01 @ M_1
    # projection of f
    # f = np.zeros((p))
    # for i in range(p):
    #     f[i] = integrate.quad(f_rhs, grid.nodal_pts[i], grid.nodal_pts[i + 1])[0]
    f = projection_1form(grid.nodal_pts, f_rhs)
    basis_0 = basis.lagrange(x)
    basis_1 = basis.edge(x)
    # plt.plot(x, np.transpose(basis_1))
    return M_0, N_1, M_1, f, basis, (basis_0, basis_1)


def mixed_poisson(p, a, b, m, f_rhs):
    """Mixed poisson, works only for 1 element."""
    grid = Grid(a, b, p)
    grid.gauss_lobatto()
    basis = PolyBasis(grid.nodal_pts, grid)

    M_1 = inner_product(basis, degree=1, a=a, b=b)

    M_0 = inner_product(basis, degree=0, a=a, b=b)

    f = np.zeros((p))
    for i in range(p):
        f[i] = integrate.quad(f_rhs, grid.nodal_pts[i], grid.nodal_pts[i + 1])[0]

    A, rhs = populate_mixed_form(M_0, M_1, f, p)
    c = np.linalg.solve(A, rhs)
    sigma = c[:p + 1]
    phi = c[p + 1:]

    x = np.linspace(a, b, m)

    phi_func = phi @ basis.edge(x)[1:, :]
    sigma_func = sigma @ basis.lagrange(x)
    return phi_func, sigma_func, x


def reverse_engineer(h, p, m, func, a, b, f_exact):
    """Project exact solution onto the finite space and use the coefficiets to verify results
    for the direct formulation"""
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
    """Direct formulation Poisson problem.

    Args:
        h (int) = # of elements
        p (int) = # of edges per elements
        func(obj.func) = forcing function
        a (float) = start of domain
        b (float) = end of domain
        f_exact (obj.func) = exact solution

    Returns:
        function (np.array) = FEM solution
        x (np.array) = computational domain
        error (float) = L_2 error
        """
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
    M_gen = assemble(M_list, index_matrix)

    # add value of the first element to the list
    f_gen = f_list[0]

    # continue for the elements 1 onword
    for i in range(1, len(f_list)):
        # if I put *1 it resolves the equation better
        f_gen = np.hstack((f_gen, f_list[i][1:]))

    rhs = M_gen @ f_gen

    # Neumann BCs
    N_gen[0, 0] = N_gen[-1, -1] = 1
    N_gen[0, 1:] = N_gen[-1, :-1] = 0

    if f_exact != None:
        rhs[0] = f_exact(a_list[0])
        rhs[-1] = f_exact(b_list[-1])
    else:
        rhs[0] = rhs[0 - 1] = 0

    function = np.ones((np.size(basis_list[0][p, :]) - 1) * h + 1)
    # Coefficients
    c = np.linalg.solve(N_gen, rhs)
    n = 1
    j = 1

    # interpolation and error calculation
    for el in range(h):
        errors[el] = l2_error_norm(c[p * el:p * (el + 1) + 1],
                                   basis_functions[el].lagrange, f_exact, (a_list[el], b_list[el]))
        partial = c[p * el: p * (el + 1) + 1] @ basis_list[el]

        if el == h - 1:
            function[el * (m - 1):] = partial
        else:
            function[el * (m - 1): (m - 1) * (el + 1)] = partial[:-1]

    x = np.linspace(a, b, np.size(function))
    return function, x, np.sum(errors)


def assemble(matrix_list, index_matrix):
    """Assemble inner product 0 forms where overlapping at nodes occurs

    Args:
        matrix_list (list) = list of matrices to assemble
        index_matrix (np.array) = Index matrix

    Returns:
        matrix_gen (np.array) = assembled matrix
    """
    p = np.shape(matrix_list[0])[0]
    # dimension of general matrix
    dim = index_matrix[-1, -1] + 1
    matrix_gen = np.zeros((dim, dim))

    for i in range(np.shape(index_matrix)[1]):
        i_0, i_f = index_matrix[0, i], index_matrix[-1, i] + 1
        matrix_gen[i_0: i_f, i_0: i_f] += matrix_list[i]

    return matrix_gen


def poisson_element(func, a, b, n, m, f_exact=None):
    """Direct formulation for 1 element. It calculates the relevant matrices for the hp version.

    Args:
        func (obj.func) = forcing function
        a (float) = start element
        b (float) = end element
        n (int) = p = # of edges
        m (int) = resolution of solution
        f_exact (obj.func) = exact solution

    Returns:
        N (np.array) = rhs matrix np.transpose(E_10) @ M_1 @ E_10
        M_0 (np.array) = inner product 0-forms basis_0
        f (np.array) = reduction of f onto 0-forms
        basis (obj.Polybasis) = shape function for the element
    """
    # init grid
    grid = Grid(a, b, n)
    # gauss lobatto nodes
    grid.gauss_lobatto()
    x = np.linspace(a, b, m)

    basis = PolyBasis(grid.nodal_pts, grid)
    # 1-forms basis inner product
    M_1 = -inner_product(basis, degree=1, a=a, b=b)
    # Coboudary operator
    E_10 = np.transpose(incidence_m1d(n))
    N = np.transpose(E_10) @ M_1 @ E_10
    # 0-forms basis inner product
    M_0 = inner_product(basis, degree=0, a=a, b=b)
    f = func(grid.nodal_pts)
    # explicit calculation of 0-form basis functions
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
    # init underlying structures
    grid = Grid(a, b, n)
    grid.gauss_lobatto()
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

    f = func(grid.nodal_pts)
    # print('Function at nodal pts :\n', f)
    rhs = np.transpose(f) @ M_0
    # boudary condtions
    rhs[0] = rhs[-1] = 0
    # solve linear system
    c = np.linalg.inv(N) @ rhs

    if f_exact is not None:
        error_norm = l2_error_norm(c, basis.lagrange, f_exact, x)
    else:
        error_norm = 0

    phi = np.transpose(c) @ basis.lagrange(x)

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
    """Calculate square of error. To be called by partial (functools)"""
    value_basis = basis_func(x)
    error_sq = (c @ value_basis - f_exact(x)) ** 2
    return error_sq[0]


def l2_error_norm(c, basis_func, f_exact, x=(-1, 1)):
    """Calculate the L_2 error norm.

    Args:
        c (np.array) = cochain coefficiets of solution
        basis_func (obj.Polybasis) = basis functons
        f_exact (obj.func) = exact solutions
    """
    # reshape coefficients for multiplication in add square
    coeff = c.reshape(1, np.size(c))
    # calculate the square error evaluating the difference squared at a pt
    square_error = partial(add_square, c=coeff, basis_func=basis_func, f_exact=f_exact)
    # TODO: Gauss lobatto quadrature
    integral = integrate.quad(square_error, x[0], x[-1])[0]
    return integral ** (0.5)


def plot_convergence(error, *args, save=False, show=True, type='Direct'):
    """Plot convergence rate
    Args:
        error_norm (np.array) = array containing error norm for incresing n
        args[0] = h or p refinement
        args[1] = number of the parameter not refined
        mixed (bool) = mixed formulation
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # unzip the h/p from the error
    x, error_norm = zip(*error)

    if args[0] == 'p':
        # set log scale along the axis
        ax.set_yscale('log')
        # cange frequency of label in x axis
        plt.xticks(np.arange(min(x), max(x) + 1, 2.0))
        # read value of h
        h = args[1]

        plt.plot(x, error_norm, '-o')
        # label of L2 error norm
        plt.ylabel(r'$ \Vert \phi^{(0)} - \phi_h^{(0)} \Vert_{L^2}$')
        # different title for mixed case
        plt.title(type + 'formulation convergence for p-refinment for h = ' + str(args[1]) + '\n'
                  r'$\phi^{(0)} = sin(2\pi x )$')

    if args[0] == 'h':
        # take the inverse of h
        x = [el ** (-1) for el in x]
        # read value of p
        p = args[1]

        p_convergence = np.zeros((np.size(x)))

        # set log scale on axis
        ax.set_yscale('log')
        ax.set_xscale('log')
        # mixed convergence show different convergence behaviour and title is different
        if type == 'Mixed':
            for i, xi in enumerate(x):
                p_convergence[i] = xi ** (p)
            plt.plot(x, p_convergence, label=r'$h^{(p)}$')
            plt.title('Mixed formulation convergence for h-refinment for p = ' +
                      str(args[1]) + '\n' r'$\phi^{(0)} = sin(2\pi x )$')
        if type == 'Direct':
            for i, xi in enumerate(x):
                p_convergence[i] = xi ** (p + 1)
            plt.plot(x, p_convergence, label=r'$h^{(p+1)}$')
            plt.title('Direct formulation convergence for h-refinment for p = ' +
                      str(args[1]) + '\n' r'$\phi^{(0)} = sin(2\pi x )$')
        if type == 'Dual direct':
            for i, xi in enumerate(x):
                p_convergence[i] = xi ** (p + 1)
            plt.plot(x, p_convergence, label=r'$h^{(p+1)}$')
            plt.title('Dual direct formulation convergence for h-refinment for p = ' +
                      str(args[1]) + '\n' r'$\phi^{(0)} = sin(2\pi x )$')

        plt.plot(x, error_norm, '-o', label=r'L2 error norm')
        plt.ylabel(r'$ \Vert \phi^{(0)} - \phi_h^{(0)} \Vert_{L^2}$')

    plt.xlabel(args[0])
    plt.legend()

    if save:
        plt.savefig('images/convergence' + args[0] + '.png', bbox_inches='tight')
    if show:
        plt.show()
