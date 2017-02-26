import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from Root_finding import newton_method
from functools import partial
import scipy as sc
from Root_finding import recursive_bisection

def f(x):return np.sin(np.pi*x/2.)
def f_poly5(x): return 0.001*x**5 + 0.02*x**3 - x  # Polynomial
def f_step(x): return np.sign(x-1)                 # Discontinuous at x=2
def f_abs(x): return np.abs(x-2)                   # Discontinuous derivative at x=2
def f_abs3(x): return np.abs((x-2)**3)             # Discontinuous 3rd derivative at x=2
def f_runge(x): return 1./(1+(x-2)**2)             # Infinitely differentiable (everywhere)
def f_gauss(x): return np.exp(-(x-2)**2/2.)        # Very similar curve to above

# Grid


def grid_uniform(a,b,n):
    # n is the number of elements in the grid
    return np.linspace(a,b,n)


def grid_chebychev(a,b,n):
    # chebychev interpolation nodes impose the minimal upper bound on the monic polinomial
    # there exist a convient formula to find the root of the Ch. polynomial
    i = np.arange(0,n+1)
    # nodal point in [-1,1]
    nodal_pts = np.cos((2*i+1)/(2*(n+1))*np.pi)
    return (-nodal_pts+1)*(b-a)/2 + a


def legendre_prime(x,n):
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x,(int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    # legendre_p = (2*(n-1)+1)*legendre(n-1)(x) + legendre_prime(x, n-2)
    legendre_p = n*legendre(n-1)(x)-n*x*legendre(n)(x)
    return legendre_p*(1-x**2)

# def legendre_prime_adjusted(x,n):
#     return legendre_prime(x,n)*(1-x**2)


def legendre_double_prime_num(x,n):
    # The result is highly inaccurate and causes newton rapson to diverge
    legendre_p = partial(legendre_prime, n=n)
    legendre_pp = np.zeros(np.size(x))
    if isinstance(x, np.ndarray):
        for i, x_i in enumerate(x):
            legendre_pp[i] = sc.misc.derivative(legendre_p,x_i)
    else:
        legendre_pp = sc.misc.derivative(legendre_p,x)
    return legendre_pp

def legendre_double_prime_recursive(x,n):
    legendre_pp = 2 * x * legendre_prime(x,n) - n*(n+1)*legendre(n)(x)
    return legendre_pp * (1-x**2)


def grid_gauss_lobatto(a, b, n):
    # roots of chebichev polynomials
    x_0 = np.cos(np.arange(1,n)/n*np.pi)
    # print('second derivatives recursive : ', legendre_double_prime_recursive(x_0,n))
    # print('second derivatives numerical : ', legendre_double_prime_num(x_0,n))
    nodal_pts = np.zeros(len(x_0)+2)
    for i, ch_pt in enumerate(x_0):
        leg_p = partial(legendre_prime, n=n)
        leg_pp = partial(legendre_double_prime_recursive, n=n)
        nodal_pts[i+1] = newton_method(leg_p, leg_pp, ch_pt, 40)[0]
    nodal_pts[0] = -1
    nodal_pts[-1] = 1
    return (-nodal_pts+1)*(b-a)/2 + a


def plot_grid(grids,show=True):
    if np.ndim(grids)>1:
        for i, grid in enumerate(grids):
            plt.plot(grid, np.ones(np.size(grid))*i, '-o')
    else:
        plt.plot(grids, np.zeros(len(grids)), '-o')
    plt.ylim(-1, 2)

    if show:
        plt.show()

# Basis


def basis_monomial(x, grid):
    phi = np.zeros((len(grid), len(x)))
    for i in range(len(grid)):
        phi[i,:] = x**i
    return phi


def basis_newton(x, grid):
    phi = np.ones((len(grid), len(x)))
    for i in range(len(grid)):
        for j in range(i):
            phi[i,:] *= (x-grid[j])
    return phi


def basis_lagrange(x,grid):
    phi = np.ones((len(grid), len(x)))
    for j in range(len(grid)):
        for i in range(len(grid)):
            if i != j:
                phi[j] *= (x-grid[i])/(grid[j]-grid[i])
    return phi


def basis_edge(x, grid):
    phi = np.ones((len(grid), len(x)))
    n = len(grid)-1
    for i in range(len(grid)):
        phi[i] = (n*(n+1)*legendre(n)(x)*(x-grid[i])+(1-x**2)*legendre_prime(x,n))/(
            n*(n+1)*legendre(n)(grid[i])*(x-grid[i])**2)
        phi[i,np.where(abs(x-grid[i]) < (b-a)/(len(x)*2))] = 0
        phi[0] = -n*(n+1)/4
        phi[-1] = -phi[0]
    return phi



def plot_basis(x, phi, grid_2):
    for base in phi:
        plt.plot(x,base)
    plot_grid(grid_2,show=False)
    plt.ylim(-2,50)
    plt.show()

# Solving interpolation condition A x = f


def find_interpolation_coeff(type, grid, f):
    A = np.zeros((np.size(grid),np.size(grid)))
    b = f(grid)
    if type == 'lagrange':
        phi = basis_lagrange(grid,grid)
    elif type == 'newton':
        phi = basis_newton(grid,grid)
    elif type == 'monomial':
        phi = basis_monomial(grid,grid)
    for i,base in enumerate(phi):
        A[:,i] = base
    coeff = np.linalg.solve(A,b)
    return coeff, A

# reconstruction

def reconstruct(coeff, phi):
    return coeff.dot(phi)


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
    a = -10
    b = 10
    grid = grid_chebychev(a,b,50)
    x = np.linspace(a,b,101)
    for func in funcs:
        coeff, A = find_interpolation_coeff('lagrange', grid, func)
        phi = basis_lagrange(x, grid)
        interpolated_f = reconstruct(coeff, phi)
        error = plot_interpolation(func, interpolated_f, x)
        print('Error max: ', error)

if __name__ == '__main__':
    a,b = -1,1
    x = np.linspace(-1,1,101)
    grid = grid_gauss_lobatto(-1,1,4)
    basis_l = basis_lagrange(x,grid)
    plot_basis(x,basis_l, grid)

    basis_e = basis_edge(x, grid)
    plot_basis(x,basis_e, grid)











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
    # funcs = [f,f_poly5,f_abs,f_abs3,f_gauss,f_runge,f_step]
    # plot_funcs(funcs)