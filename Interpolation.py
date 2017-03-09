"""This modules contains functions and classes to generate a basis of interpolant.

Current supported interpolants: PolyBasis1d, Polybasis2d and Splines.
"""
from grid import Grid, Grid2d
import matplotlib.pyplot as plt
import numpy as np
import plotly
# plotly credential for plottin in jupyter
plotly.tools.set_credentials_file(username='lorenzoterenzi96', api_key='sXv7B4MQcJKMvaYp96Jq')

# modules used in examples
# import math
# import stiffness_matrix
# import integration
# from scipy import integrate
# from functools import partial


class PolyBasis(object):
    """1D polynomial basis set for interpolation.

    Calculate all the necessary basis function for 1d interpolation.

    Attributes:
        x (float, np.array) = domain
        grid (obj Grid) = computational grid
        nodal_pts (np.array) = nodal points used to impose the intepolation condition
        n (int) = number of nodal points - 1
        basis(np.array) = n+1 × len(x) array containing the value of the n+1 basis functions at each point of the domain
    """

    def __init__(self, x, grid):
        """Initialize basis.

        Args:
            x (float, np.array) = domain
            grid (obj Grid) = computational grid
        """
        self.x = x
        self.grid = grid
        self.nodal_pts = grid.nodal_pts
        # degree of the basis
        self.n = np.size(self.nodal_pts)
        # init basis
        self.init_basis()

    def init_basis(self):
        self.basis = np.ones((np.size(self.nodal_pts), np.size(self.x)))

    def monomial(self, x=None):
        """Monomial basis functions {1,x,x**2,..}.
        """
        if x:
            self.x = x
        for i in range(self.n):
            self.basis[i, :] = self.x ** i
        if x:
            return self.basis

    def newton(self, x=None):
        """Calculate newton polynomial basis.

        Basis function ϵ_i = ∏_j=0^i-1 (x-x_i)
        Returns (if domain is given):
            basis (np.array) = set of basis functions evaluated

        """
        # if domain is not given do not override pre-prescribed domain
        if x is not None:
            self.x = x

        self.init_basis()
        for i in range(self.n):
            # multiply through all the nodal pts
            for j in range(i):
                self.basis[i, :] *= (self.x - self.nodal_pts[j])
        if x is not None:
            return self.basis

    def lagrange(self, x=None, index_basis=None):
        """Calculate Lagrange basis.

        Lagrange basis is of the form ϕ_i = ∏_j!=i,j=0^n frac{x-x_j}{x_i-x_j}

        Args:
            x (np.array,int : optional) = domain
            index_basis (int : optional) = index of the basis to be returned

        Returns:
            basis[index_basis] = index_basis th basis in self.basis
        """
        if x is not None:
            self.x = x
        # reset self.basis
        self.init_basis()
        # iteration through all nodal points
        for j in range(self.n):
            for i in range(self.n):
                if i != j:
                    self.basis[j] *= (self.x - self.nodal_pts[i]) / \
                        (self.nodal_pts[j] - self.nodal_pts[i])
        if x is not None:
            if index_basis is not None:
                return self.basis[index_basis]
            else:
                return self.basis

    def edge(self, x=None, index_basis=None):
        """Calculate edge basis functions.

        Edge functions are used to interpolate 1-forms.
        Args:
            x (np.array,int : optional) = domain
            index_basis (int : optional) = index of the basis to be returned

        Returns:
            basis[index_basis] = index_basis th basis in self.basis
        """
        if x is not None:
            self.x = x
        self.init_basis()
        # fill basis with lagrange polynomilas
        self.lagrange()

        # calculate derivative lagrange poly
        np.seterr(divide='ignore', invalid='ignore')
        for j in range(self.n):
            self.basis[j] = 1 / (self.x - self.nodal_pts[j]) * (P_prime(self.x, self.nodal_pts) /
                                                                P_prime(self.nodal_pts[j], self.nodal_pts) - self.basis[j])
            # Since the pts in the interval coincide with nodal_pts[j]
            # we have to add the limit of the function
            limit_value = 0
            for l in range(self.n):
                if l != j:
                    limit_value += 1 / (self.nodal_pts[j] - self.nodal_pts[l])
            # check where the limit value has to be substitued
            self.basis[j, np.where(
                abs(self.x - self.nodal_pts[j]) < 1.e-10)] = limit_value

        # calculation of the edge polynomials
        # allocation memory
        edge = np.zeros((self.n, np.size(self.x)))
        for i in range(1, self.n):
            for k in range(0, i):
                edge[i] += self.basis[k]
        self.basis = -edge

        if x is not None:
            if index_basis is not None:
                return self.basis[index_basis]
            else:
                return self.basis

    def lagrange_1(self):
        """Different way of calculating lagrange poly.

        Deprecated.
        """
        for i in range(self.n):
            np.seterr(divide='ignore', invalid='ignore')
            self.basis[i] = P(self.x, self.nodal_pts) /\
                (P_prime(self.nodal_pts[i], self.nodal_pts)
                 * (self.x - self.nodal_pts[i]))
            self.basis[i, np.where(
                abs(self.x - self.nodal_pts[i]) < 1.e-10)] = 1

    def plot(self, *args, save=False):
        """Plot basis functions.

        Args:
            args(optinal) = list of strings used for title and axis labels
            save(bool : optional) = True save the figure
        """
        # find max and min value to be plotted
        y_min, y_max = 0, 0
        for base in self.basis:
            plt.plot(self.x, base)
            if y_min > np.min(base):
                y_min = np.min(base)
            if y_max < np.max(base):
                y_max = np.max(base)
        # set limits to the axis
        plt.ylim(y_min * 1.1, y_max * 1.1)
        # add labels and title
        if args:
            plt.title(args[0] + ' interpolation')
            plt.ylabel(args[1])
        plt.xlabel(r'$\xi$')
        # save
        if save:
            plt.savefig('images/' +
                        args[0] + '_N_' + str(self.n - 1) + '.png', bbox_inches='tight')
        self.grid.plot()


class Polybasis2d(object):

    """2D polynomial basis set for interpolation.

    Calculate all the necessary basis function for 2d interpolation.
    It is required a regualar grid. The 2d basis functions are computed through
    tensor product of 1d basis functions.

    Attributes:
        domain (np.array) = domain of computation.
        It's given by two arrays of n+1 pts in the x and y direction.
        n_x (float) = number of nodal_pts in the x direction - 1.
        n_y (float) = number of nodal_pts in the y direction - 1.
        grid (obj Grid2d) = 2D computational grid.
        basis_x (obj Polybasis) = 1d basis for the equivalent 1d grid in the x direction.
        basis_y (obj Polybasis) = 1d basis for the equivalent 1d grid in the y direction.
    """

    def __init__(self, domain, grid):
        assert(np.shape(domain)[0] == 2), "The domain should be 2 dimensional"
        # number of colums = # nodal points in x
        self.n_x = np.shape(grid.nodal_pts)[1]
        # number of rows = # nodal points in y
        self.n_y = np.shape(grid.nodal_pts)[0]
        # self.basis = np.empty((self.n_y, self.n_x, len(domain[1]),
        # len(domain[0])))
        self.domain = domain
        self.init_basis()
        self.grid = grid
        self.basis_x = PolyBasis(self.domain[0], self.grid.grid1d[0])
        self.basis_y = PolyBasis(self.domain[1], self.grid.grid1d[1])

    def init_basis(self):
    """Setup structure of basis attribute a 4 dimensional array.

        Example:
            basis[i,j,k,n] identifies the basis function for the nodal point [i,j] in the 2dgrid, k and n are the y and x position in the domain where the basis function is evaluated.
            If the domain is normalize, i.e x ∈ [-1,1] and y ∈ [-1,1], then basis[0,0,:,:] reppresent the basis function at nodal point = (-1,-1) (in the lower left corner).

        Note:
            The order of x and y indeces appear inverted but in this case using a right handed coordinate system moving along x corresponds in shifting in column and moving along y corresponds to shifting along rows.
        """
        self.basis = np.empty((self.n_y, self.n_x, np.size(
            self.domain[1]), np.size(self.domain[0])))

    def lagrange(self, domain=None):
        """Compute the 2d orthogonal Lagrange basis functions.

        The basis function are obtained through tensor product of corresponding 1D ones.

        Args:
            domain(np.array:optional) = 2D dimensional array reppresenting the domain.

        Returns(optional):
            basis(4d np.array) = all the basis functions evaluated in the domain.
        """

        # probably there is a bug in the case the domain consists of two pts
        if domain is not None:
            assert(np.shape(domain)[0] ==
                   2), "The domain should be 2 dimensional"
            self.domain = domain
            # updating the 1s basis for new domain
            self.basis_x = PolyBasis(self.domain[0], self.grid.grid1d[0])
            self.basis_y = PolyBasis(self.domain[1], self.grid.grid1d[1])
        # populating 1d basis func with Lagrange polynomials
        self.basis_x.lagrange()
        self.basis_y.lagrange()
        # calculation of ϕ_ij(x,y) = ϕ_i(x) ⊗ ϕ_j(y)
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.basis[j, i] = np.tensordot(
                    self.basis_y.basis[j], self.basis_x.basis[i],  axes=0)
        if domain is not None:
            return self.basis

    def plot(self, i, j):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        x, y = np.meshgrid(x, y)
        trace1 = go.Surface(x=x, y=y, z=self.basis[i, j], colorscale='Viridis')
        py.iplot([trace1])


def find_interpolation_coeff(type, grid, f):
    """Find interpolation coefficients.

    Calculate the coefficents used to take recontruct the function through a linear
    combination of the basis functions.

    Args:
        type (str) =name of the basis functions you want to use
        grid (obj Grid) = computational grid
        f (obj func) = fuction to interpolate

    Returns:
        coeff (np.array) = basis functions coefficiets

    """
    # value of the basis functions at nodal points
    my_basis = PolyBasis(grid.nodal_pts, grid)
    # Vandermonde matrix
    A = np.zeros((np.size(grid.nodal_pts), np.size(grid.nodal_pts)))
    # value of the function at nodal pts
    b = f(grid.nodal_pts)
    if type == 'lagrange':
        my_basis.lagrange()
    elif type == 'newton':
        my_basis.newton()
    elif type == 'monomial':
        my_basis.monomial()
    # populated Vandermonde
    for i, base in enumerate(my_basis.basis):
        A[:, i] = base
    # solve linear system
    coeff = np.linalg.solve(A, b)
    return coeff, A


def reconstruct(coeff, basis):
    """Reconstruct polynomial from basis.

    Returns:
        interpolant (np.array) = intepolant polynomial evaluated in the domain.
    """
    return coeff.dot(basis.basis)


def plot_interpolation(f, interpolated_f, x):
    """Plot intepolant and original function side by side.

    Args:
        f (obj func)= original functions.
        interpolated_f (np.array) = interpolant.
        x (np.array) = domain.
    Returns:
        error_max(float) = max error in the interpolation.
    """
    error = abs(f(x) - interpolated_f)
    plt.plot(x, f(x), '-', label='Original function')
    plt.plot(x, interpolated_f, '-', label='Interpolated function')
    plt.plot(x, error, '-', label='error')
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend(), plt.show()
    error_max = np.max(error)
    return error_max


def plot_funcs(funcs):
    """Plot multiple at the same time."""
    a, b = -5, 5
    x_1 = np.linspace(a, b, 101)

    grid_1 = Grid(a, b, 20)
    grid_1.gauss_lobatto()

    basis_1 = PolyBasis(x_1, grid_1)
    basis_1.lagrange()

    for func in funcs:
        coeff, A = find_interpolation_coeff('lagrange', grid_1, func)
        interpolated_f = reconstruct(coeff, basis_1)
        error = plot_interpolation(func, interpolated_f, x_1)
        print('Error max: ', error)


def P(x, nodal_pts):
    """Calculate monic polynomial with roots at nodal points."""
    ans = 1
    for i, nodal_pt in enumerate(nodal_pts):
        ans *= (x - nodal_pt)
    return ans


def P_prime(x, nodal_pts):
    """Calculate derivative of monic polynomial."""
    ans = 0
    for n in range(len(nodal_pts)):
        prod = 1
        for l in range(len(nodal_pts)):
            if l != n:
                prod *= (x - nodal_pts[l])
        ans += prod
    return ans


class Splines(object):
    """Piecewise interpolation of function on subintervals, total points n.

    The degree of the interpolant determines the number of points in the interval needed.
    Current options: -cubic.

    Attributes:
        nodal_pts (np.array) = nodal points for interpolation
        n (int) = number of nodal pts - 1
        x (np.array, float) = domain
        h (np.array) = spacing between domain pts
        h_i (np.array) = spacing between nodal pts
        coeffs (np.array) = coeffcients of cubic polynomial (4)

    """

    def __init__(self, x, Grid):
        self.nodal_pts = Grid.nodal_pts
        self.n = np.size(self.nodal_pts)
        self.x = x
        self.h = self.x[1:] - self.x[:-1]
        self.hi = self.nodal_pts[1:] - self.nodal_pts[:-1]
        self.coeffs = np.empty(1)
        self.f = None
        self.fi = None
        self.interpolant = None

    def cubic_function(self, mask, index):
        """Cubic interpolant polynomial.

            Args:
                mask (np.array)  = arrays of 0s and 1s to select just part of the domain.
                index (int) = indicate the nodal point (ith) used to construct the cubic polinomial.

            Returns:
                S (np.array) = index th polynomial evaluated in the ith interval of the domain.
        """
        S = self.coeffs[index, 0] + \
            self.coeffs[index, 1] * (self.x[mask] - self.nodal_pts[index]) + \
            self.coeffs[index, 2] * (self.x[mask] - self.nodal_pts[index]) ** 2 + \
            self.coeffs[index, 3] * (self.x[mask] - self.nodal_pts[index]) ** 3
        return S

    def cubic(self, f_i, natural=[0, 0]):
        """Generete cubic spline.

        Calculates each spline per interval in the domain and populates
        the attribute self.interpolant.

        Args:
            f_i (obj func or np.array) = func or set of values used for interpolation conditions.
            natural (list : optional) = values of second derivatives of cubic splines at
            the extremes of the domain.
        """
        # check if f_i is a function
        if hasattr(f_i, '__call__'):
            self.fi = f_i(self.nodal_pts)
        # in case f_i is a array of points
        else:
            self.fi = f_i
        self.interpolant = np.empty(np.size(self.x))
        # Setup of necessary coeffs : n x 4 array
        self.coeffs = np.zeros((self.n - 1, 4))
        # First solve linear system to find Mi where coeffs[:,2] = Mi/2
        A = np.zeros((self.n, self.n))
        rhs = np.zeros(self.n)
        for i in range(1, self.n - 1):
            A[i, i - 1] = self.hi[i - 1]
            A[i, i] = 2 * (self.hi[i] + self.hi[i - 1])
            A[i, i + 1] = self.hi[i]
            rhs[i] = 3 * ((self.fi[i + 1] - self.fi[i]) / self.hi[i] -
                          (self.fi[i] - self.fi[i - 1]) / self.hi[i - 1])
        # Impose end points that are used later to impose BCs
        A[0, 0] = A[self.n - 1, self.n - 1] = 1
        rhs[0] = natural[0]
        rhs[self.n - 1] = natural[-1]
        # solve for coeffs
        M = np.linalg.solve(A, rhs)
        # print('Matrix M: ', M)
        # coeff[:,3 and 2] should be half of what they are now,
        # coeff[:,1] should be divided by 6 and not by 3
        # this is due to M = 2 b = 2 * coeff[:,1]
        self.coeffs[:, 2] = M[:-1]
        self.coeffs[:, 0] = self.fi[:-1]
        self.coeffs[:, 3] = (M[1:] - M[:-1]) / (3 * self.hi)
        self.coeffs[:, 1] = (self.fi[1:] - self.fi[:-1]) / self.hi - \
            self.hi / 3 * (2 * M[:-1] + M[1:])

        # lets find in which interval j is x
        index_location = np.digitize(self.x, self.nodal_pts, right=True) - 1
        # remove indeces lower than zero due x mathching fist value of the
        # nodal_pts
        index_location[np.where(index_location < 0)] = 0
        # calculate values of the cubic polynomials
        for i in range(self.n - 1):
            # set up a mask for the interval i
            mask = np.where(index_location == i)
            # evaluate cubic polinomial for a single interval
            self.interpolant[mask] = self.cubic_function(mask, i)
# examples

    # # testing inner product
    # a, b = -1, 1
    # n = 2
    # n_glob_int = math.ceil((n ** 2 + 1) / 2)
    # x = np.linspace(a, b, 101)
    # #
    # grid = Grid(a, b, n)
    # grid.gauss_lobatto()
    # # grid.plot()
    # #
    # basis = PolyBasis(x, grid)
    # # basis.lagrange()
    # # basis.edge()
    # M = stiffness_matrix.inner_product(basis, degree=1)
    # basis_0 = partial(basis.edge, index_basis=0)
    # # basis_1
    # print('basis at pt :', basis_0(x))
    # print('shape baiss.edge(x,0) ', np.shape(basis_0(x)))
    # # print('shape basis.lagrange(x,0) ', np.shape(basis.lagrange(x, 0)))
    # # print('shape basis.lagrange ', np.shape(basis.lagrange(x)))
    # plt.plot(x, basis_0(x))
    # plt.show()
    # # basis_00**2 inner product works
    # # int = integration.quad_glob([basis_0, basis_0], -1, 1, n_glob_int)
    # # print(int)
    # # print('integral lagrange ', int)
    # print(M)

    # for i in range(n):
    #     for j in range(n):
    #         prod = partial(stiffness_matrix.product_basis, fs=[
    #                        basis.edge, basis.edge], indexes=[i, j])
    #         print('inner product {0},{1} : {2}' .format(i, j, integrate.quad(prod, -1, 1)[0]))
    # prod = partial(stiffness_matrix.product_basis, fs=[basis.edge, basis.edge], indexes=[1, 0])
    # print('Product of basis ', prod(x))
    # l = basis.edge(0.5)
    # prod = basis.basis[3]
    # print('Basis edge \n', l)
    # print(np.size(prod))
    # plt.plot(x, basis.basis[2])
    # plt.show()
    # 2d lagrange testing
    # grid2d = Grid2d((-1, -1), (1, 1), (4, 4))
    # grid2d.uniform()
    # # grid2d.plot()
    # # print(grid2d.nodal_pts[0, 0], grid2d.nodal_pts[0, -1])
    # x = np.linspace(-1, 1, 10)
    # y = np.linspace(-1, 1, 10)
    # basis = Polybasis2d((x, y), grid2d)
    # foo = np.array((0, 1))
    # basis.lagrange(foo)
    # print(basis.basis)
    # # basis.plot(0, 0)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # np.savetxt('test.csv', basis.basis[1, 1], delimiter=',')
    # print(basis.basis[1, 1])
    # surf = ax.plot_surface(x, y, basis.basis[1, 1])
    # plt.show()

    # # spline testing
    # a, b = -1, 1
    # x = np.linspace(a, b, 101)
    # grid = Grid(a, b, 4)
    # grid.uniform()
    # spline = Splines(x, grid)
    # spline.cubic(f_runge)
    # print(spline.coeffs)
    # plt.plot(x, f_runge(x))
    # plt.plot(x, spline.interpolant)
    # plt.show()

    # a, b = -1, 1
    # x = np.linspace(a, b, 101)
    # #
    # grid = Grid(a, b, 4)
    # grid.gauss_lobatto()
    # #
    # basis = PolyBasis(x, grid)
    # x = 0.5
    # e_0 = basis.lagrange(x)
    # print('basis 0 at x : \n', e_0)
    # basis.lagrange()
    # basis.plot('GL nodal', r'$l_i(\xi)$', save=True)
    #
    # basis_1 = PolyBasis(x, grid)
    # e_1 = basis_1.edge(x)
    # print('basis 1 at x : \n', e_1)
    # basis_1.edge()

    # basis_1.plot('GL edge', r'$e_i(\xi)$', save=True)
    #
    # # coeff, A = find_interpolation_coeff('lagrange',grid_2,f)
    # phi = basis_lagrange(x,grid_2)
    # interpolated_f = reconstruct(coeff,phi)
    # error = plot_interpolation(f,interpolated_f,x)

    # # error tends to decrease with the Chebischev nodes
    # print('Error max: ', error)
    # funcs = [f,f_poly5,f_abs,f_abs3,f_gauss,f_runge,f_step]
    # plot_funcs(funcs)
