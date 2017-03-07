from mpl_toolkits.mplot3d import Axes3D
from grid import Grid, Grid2d
import matplotlib.pyplot as plt
import numpy as np
import plotly
plotly.tools.set_credentials_file(username='lorenzoterenzi96', api_key='sXv7B4MQcJKMvaYp96Jq')


def f(x):
    return np.sin(np.pi * x / 2.)


def f_poly5(x):
    return 0.001 * x ** 5 + 0.02 * x ** 3 - x  # Polynomial


def f_step(x):
    return np.sign(x - 1)                 # Discontinuous at x=2


# Discontinuous derivative at x=2
def f_abs(x):
    return np.abs(x - 2)


# Discontinuous 3rd derivative at x=2
def f_abs3(x):
    return np.abs((x - 2) ** 3)

# Infinitely differentiable (everywhere)


def f_runge(x):
    return 1. / (1 + (4. * x) ** 2)


def f_gauss(x):
    return np.exp(-(x - 2) ** 2 / 2.)


class PolyBasis(object):

    def __init__(self, x, grid):
        self.x = x
        self.grid = grid
        self.nodal_pts = grid.nodal_pts
        # degree of the basis
        self.n = np.size(self.nodal_pts)
        self.init_basis()

    def init_basis(self):
        self.basis = np.ones((np.size(self.nodal_pts), np.size(self.x)))
        # if isinstance(self.x, (int, float)):
        #     self.basis = np.ones((len(self.nodal_pts), 1))
        # else:
        #     self.basis = np.ones((len(self.nodal_pts), len(x)))

    def monomial(self, x=None):
        if x:
            self.x = x
        for i in range(self.n):
            self.basis[i, :] = self.x ** i
        if x:
            return self.basis

    def newton(self, x=None):
        """Calculate newton polynomial basis."""
        if x != None:
            self.x = x

        self.init_basis()
        for i in range(self.n):
            for j in range(i):
                self.basis[i, :] *= (self.x - self.nodal_pts[j])
        if x != None:
            return self.basis

    def lagrange(self, x=None):
        if x != None:
            self.x = x
        self.init_basis()
        for j in range(self.n):
            for i in range(self.n):
                if i != j:
                    self.basis[j] *= (self.x - self.nodal_pts[i]) / \
                        (self.nodal_pts[j] - self.nodal_pts[i])
        if x != None:
            return self.basis

    def edge(self, x=None):
        if x != None:
            self.x = x
        self.init_basis()
        # lagrange poly
        self.lagrange()

        # derivative lagrange poly
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
            self.basis[j, np.where(
                abs(self.x - self.nodal_pts[j]) < 1.e-10)] = limit_value

        # calculation of the edge poly

        edge = np.zeros((self.n, np.size(self.x)))
        for i in range(1, self.n):
            for k in range(0, i):
                edge[i] += self.basis[k]
        self.basis = -edge
        if x != None:
            return self.basis

    def lagrange_1(self):
        for i in range(self.n):
            np.seterr(divide='ignore', invalid='ignore')
            self.basis[i] = P(self.x, self.nodal_pts) /\
                (P_prime(self.nodal_pts[i], self.nodal_pts)
                 * (self.x - self.nodal_pts[i]))
            self.basis[i, np.where(
                abs(self.x - self.nodal_pts[i]) < 1.e-10)] = 1

    def plot(self, *args, save=False):
        y_min, y_max = 0, 0
        for base in self.basis:
            plt.plot(self.x, base)
            if y_min > np.min(base):
                y_min = np.min(base)
            if y_max < np.max(base):
                y_max = np.max(base)
        plt.ylim(y_min * 1.1, y_max * 1.1)
        if args:
            plt.title(args[0] + ' interpolation')
            plt.ylabel(args[1])
        plt.xlabel(r'$\xi$')
        if save:
            plt.savefig('images/' +
                        args[0] + '_N_' + str(self.n - 1) + '.png', bbox_inches='tight')
        self.grid.plot()


def find_interpolation_coeff(type, grid, f):
    my_basis = PolyBasis(grid.nodal_pts, grid)
    # Vandermonde matrix
    A = np.zeros((np.size(grid.nodal_pts), np.size(grid.nodal_pts)))
    b = f(grid.nodal_pts)
    if type == 'lagrange':
        my_basis.lagrange()
    elif type == 'newton':
        my_basis.newton()
    elif type == 'monomial':
        my_basis.monomial()
    for i, base in enumerate(my_basis.basis):
        A[:, i] = base
    coeff = np.linalg.solve(A, b)
    return coeff, A


def reconstruct(coeff, basis):
    """Reconstruct polynomial from basis."""
    return coeff.dot(basis.basis)


def plot_interpolation(f, interpolated_f, x):
    error = abs(f(x) - interpolated_f)
    plt.plot(x, f(x), '-', label='Original function')
    plt.plot(x, interpolated_f, '-', label='Interpolated function')
    plt.plot(x, error, '-', label='error')
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend(), plt.show()
    error_max = np.max(error)
    return error_max


def plot_funcs(funcs):
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
    ans = 1
    for i, nodal_pt in enumerate(nodal_pts):
        ans *= (x - nodal_pt)
    return ans


def P_prime(x, nodal_pts):
    ans = 0
    for n in range(len(nodal_pts)):
        prod = 1
        for l in range(len(nodal_pts)):
            if l != n:
                prod *= (x - nodal_pts[l])
        ans += prod
    return ans


class Splines(object):
    """
    Piecewise interpolation of function on subintervals, total points n.
    The degree of the interpolant determines the number of points in the interval needed.
    Current options: -cubic.
    """

    def __init__(self, x, Grid):
        self.nodal_pts = Grid.nodal_pts
        self.n = len(self.nodal_pts)
        self.x = x
        self.h = self.x[1:] - self.x[:-1]
        self.hi = self.nodal_pts[1:] - self.nodal_pts[:-1]
        self.coeffs = np.empty(1)
        self.f = None
        self.fi = None
        self.interpolant = None

    def cubic_function(self, mask, index):
        """Cubic interpolant polynomial."""
        S = self.coeffs[index, 0] + \
            self.coeffs[index, 1] * (self.x[mask] - self.nodal_pts[index]) + \
            self.coeffs[index, 2] * (self.x[mask] - self.nodal_pts[index]) ** 2 + \
            self.coeffs[index, 3] * (self.x[mask] - self.nodal_pts[index]) ** 3
        return S

    def cubic(self, f, natural=[0, 0]):
        # laod function
        self.f = f
        self.fi = f(self.nodal_pts)
        self.interpolant = np.empty(len(x))
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
        print('rhs : \n', rhs)
        print('fi : ', self.fi, self.nodal_pts)
        # plt.imshow(A, interpolation='none')
        # BSc for s''(x) = Mi
        rhs[0] = natural[0]
        rhs[self.n - 1] = natural[-1]
        # set coefficients
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
            mask = np.where(index_location == i)
            self.interpolant[mask] = self.cubic_function(mask, i)
            # print('Self interpolant \n', self.interpolant)


class Polybasis2d(object):

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
        self.basis = np.empty((self.n_y, self.n_x, np.size(
            self.domain[1]), np.size(self.domain[0])))

    def lagrange(self, domain=None):
        # probably there is a bug in the case the domain consists of two pts
        if domain != None:
            assert(np.shape(domain)[0] ==
                   2), "The domain should be 2 dimensional"
            self.domain = domain
            self.basis_x = PolyBasis(self.domain[0], self.grid.grid1d[0])
            self.basis_y = PolyBasis(self.domain[1], self.grid.grid1d[1])
        self.basis_x.lagrange()
        self.basis_y.lagrange()
        # print('x basis \n', self.basis_x.basis)
        # print('y basis \n', self.basis_y.basis)
        # print('squared \n', self.basis_x.basis ** 2)
        # print('nx : ', self.n_x)
        # print('ny : ', self.n_y)
        test1 = np.tensordot(self.basis_x.basis[0], self.basis_y.basis[0], axes=0)
        # print('tensor product: ', np.shape(test1), ' \n', test1)
        print(self.n_x, self.n_y)
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.basis[j, i] = np.tensordot(
                    self.basis_y.basis[j], self.basis_x.basis[i],  axes=0)
        if domain != None:
            return self.basis

    def plot(self, i, j):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        x, y = np.meshgrid(x, y)
        trace1 = go.Surface(x=x, y=y, z=self.basis[i, j], colorscale='Viridis')
        py.iplot([trace1])

# if __name__ == '__main__':
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
    # # x = 0.5
    # # e_0 = basis.lagrange(x)
    # # print('basis 0 at x : \n', e_0)
    # basis.lagrange()
    # basis.plot('GL nodal', r'$l_i(\xi)$', save=True)
    # #
    # basis_1 = PolyBasis(x, grid)
    # # e_1 = basis_1.edge(x)
    # # print('basis 1 at x : \n', e_1)
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
