import matplotlib.pyplot as plt
from legendre_functions import *
from grid import Grid

def f(x): return np.sin(np.pi*x/2.)
def f_poly5(x): return 0.001*x**5 + 0.02*x**3 - x  # Polynomial
def f_step(x): return np.sign(x-1)                 # Discontinuous at x=2
def f_abs(x): return np.abs(x-2)                   # Discontinuous derivative at x=2
def f_abs3(x): return np.abs((x-2)**3)             # Discontinuous 3rd derivative at x=2
def f_runge(x): return 1./(1+(x-2)**2)             # Infinitely differentiable (everywhere)
def f_gauss(x): return np.exp(-(x-2)**2/2.)        # Very similar curve to above


class PolyBasis:

    def __init__(self, x, grid):
        self.x = x
        self.grid = grid
        self.nodal_pts = grid.nodal_pts
        # degree of the basis
        self.n = len(self.nodal_pts)
        self.basis = np.ones((len(self.nodal_pts), len(x)))

    def monomial(self):
        self.basis = np.zeros((self.n, len(self.x)))
        for i in range(self.n):
            self.basis[i,:] = self.x**i

    def newton(self):
        self.basis = np.ones((self.n, len(self.x)))
        for i in range(self.n):
            for j in range(i):
                self.basis[i,:] *= (self.x-self.nodal_pts[j])

    def lagrange(self):
        self.basis = np.ones((self.n, len(self.x)))
        for j in range(self.n):
            for i in range(self.n):
                if i != j:
                    self.basis[j] *= (self.x-self.nodal_pts[i])/(self.nodal_pts[j]-self.nodal_pts[i])

    def plot(self):
        for base in self.basis:
            plt.plot(self.x,base)
        self.grid.plot()


def find_interpolation_coeff(type, grid, f):
    my_basis = PolyBasis(grid.nodal_pts,grid)
    # Vandermonde matrix
    A = np.zeros((np.size(grid.nodal_pts),np.size(grid.nodal_pts)))
    b = f(grid.nodal_pts)
    if type == 'lagrange':
        my_basis.lagrange()
    elif type == 'newton':
        my_basis.newton()
    elif type == 'monomial':
        my_basis.monomial()
    for i,base in enumerate(my_basis.basis):
        A[:,i] = base
    coeff = np.linalg.solve(A,b)
    return coeff, A


def reconstruct(coeff, basis):
    '''
    Reconstruct polynomial from basis
    '''
    return coeff.dot(basis.basis)


def plot_interpolation(f, interpolated_f, x):
    error = abs(f(x) - interpolated_f)
    plt.plot(x,f(x),'-', label='Original function')
    plt.plot(x,interpolated_f, '-', label='Interpolated function')
    plt.plot(x,error,'-', label='error')
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

if __name__ == '__main__':
    a,b = -1,1
    x = np.linspace(a,b, 1001)

    grid = Grid(a,b,3)
    grid.gauss_lobatto()

    basis = PolyBasis(x,grid)
    basis.lagrange()
    # basis.plot()

    # grid_0.plot()

    # grid = grid_chebychev(-1,1,3)
    # grid_2 = grid_uniform(-2,2,3)
    # print('Chebishev grid ', grid)
    # x = np.linspace(-1,1,1001)
    # plot_grid(grid,show=False)

    # testing legendre derivative and closeness of chebichev
    # legendre_p = legendre_prime(x,2)
    # legendre_pp = legendre_double_prime(x,5)
    # legendre_pp2 = legendre_double_prime_num(x,3)
    # plt.plot(x,legendre_p),\
    # plt.plot(x,legendre_pp),\
    # plt.plot(x, legendre_pp2),

    # # testing gauss lobatta
    # grid_lobatto = grid_gauss_lobatto(-1,1,2)
    # #
    # print(grid_lobatto)
    # plot_grid(grid_lobatto)

    # phi_mon = basis_monomial(x,grid_2)
    # phi_new = basis_newton(x,grid_2)
    # phi_lag = basis_lagrange(x,grid_2)
    # # plot_basis(x,phi_mon)
    # # plot_basis(x,phi_new)
    # plot_basis(x,phi_lag, grid_2)
    # print(basis_lagrange([grid_2[1]],grid_2))
    # plot_grid([grid, grid_2])

    # coeff, A = find_interpolation_coeff('lagrange',grid_2,f)
    # phi = basis_lagrange(x,grid_2)
    # interpolated_f = reconstruct(coeff,phi)
    # error = plot_interpolation(f,interpolated_f,x)

    # # error tends to decrease with the Chebischev nodes
    # print('Error max: ', error)
    funcs = [f,f_poly5,f_abs,f_abs3,f_gauss,f_runge,f_step]
    plot_funcs(funcs)