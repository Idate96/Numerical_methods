import numpy as np
import matplotlib.pyplot as plt


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
    return np.linspace(a,b,n+1)


def grid_chebychev(a,b,n):
    # chebychev interpolation nodes impose the minimal upper bound on the monic polinomial
    # there exist a convient formula to find the root of the Ch. polynomial
    i = np.arange(0,n+1)
    # nodal point in [-1,1]
    nodal_pts = np.cos((2*i + 1)/(2*(n+1))*np.pi)
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


def plot_basis(x, phi, grid_2):
    for base in phi:
        plt.plot(x,base)
    plot_grid(grid_2,show=False)
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
    grid = grid_chebychev(-2,2,2)
    grid_2 = grid_uniform(-2,2,3)
    x = np.linspace(-2,2,101)
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