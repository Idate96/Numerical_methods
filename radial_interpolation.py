"""This module contains methods to perform radial interpolation in 1D and 2D."""
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid, Grid2d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import pdb

# radial functions


def phi_invquad(r, l):
    return 1. / (1 + (l * r) ** 2)


def phi_invmultiquad(r, l):
    return 1. / np.sqrt(1 + (l * r) ** 2)


def phi_multiquad(r, l):
    return np.sqrt(1 + (l * r) ** 2)


def phi_linear(r, l):
    return r


def test_func(x):
    return np.sin(x)


def test_func2d(x, y):
    return y / (2. + x ** 2)


def metric_1d(x_1, x_2):
    """Euclidean metric function on 1d."""
    return np.abs(x_1 - x_2)


def metric_2d(x_2, y_2, x_1, y_1):
    """Euclidean metric function in 2d.

    Example:
        x = np.linspace(-5, 5, 11)
        y = np.linspace(-5, 5, 11)
        xx, yy = np.meshgrid(x, y)
        dist = metric_2d(xx, yy, 0, 0)
        # returns distances of the domain pts from (0,0) in a 11x11 array
        # ex distance of domain[i,j] wrt to (0,0) is dist[i,j] cool!.
    """
    return np.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2)


def construct_basis(radial_func, l, x_i, x):
    """Construct the basis functions over the domain.

    Args:
        radial_func (obj func) = radial function of your choice.
        l (float) = parameter for radial function the higher, the more spicky.
        x_i (np.array) = degrees of freedom.
        f_exact = func to be interpolated.
    """
    x = np.array(np.size(x_i) * [x])
    x_i = x_i.reshape(np.size(x_i), 1)
    basis = radial_func(metric_1d(x, x_i), l)
    return basis


def construct_basis2d(radial_func, l, nodal_pts, domain):
    """Construct the 2d radial basis on the domain.

    Args:
        radial_func (obj func) = radial function
        l (float) = parameter determining the sharpness of radial function
        nodal_pts (np.array : shape (# dof, 2)) =  dof with coordinates stored as column vector, i.e. ith dof : nodal_pts[i] = [x,y].
        domain (tuple of np.array) = (xx,yy) generated with meshgrid.

    Returns:
        basis (np.array: shape = (# dof x, # dof y, # pts domain x, # pts domain y)) =
        set of basis function evaluated over the domain.

    """
    # stucture basis array as (# dof x, # dof y, # pts domain x, # pts domain y)
    # *np.shape(nodal_pts)[:-1] select of the shape (# dof x, # dof y)
    # *np.shape(domain)[1:]) select the shape (# pts domain x, # pts domain y)
    basis = np.zeros((*np.shape(nodal_pts)[:-1], *np.shape(domain)[1:]))
    for i in range(np.shape(nodal_pts)[0]):
        for j in range(np.shape(nodal_pts)[1]):
            coord = nodal_pts[i, j]
            # fancy evaluation of distances
            basis[i, j] = radial_func(metric_2d(domain[0], domain[1], coord[0], coord[1]), l)
    return basis


def construct_vandermonde2d(radial_func, l, nodal_pts):
    """Construct the Vandermonde matrix.

    Args:
        radial_func (obj func) = radial function
        l (float) = parameter determining the sharpness of radial function
        nodal_pts (np.array : shape(# dof x, # dof y, 2) = dof with coordinates stored, i.e. nodal_pts[i,j] = [x,y].

    Return:
        V (np.array : shape(# dof, # dof)) = radial_func(x_i - x_j), Vandermonde matrix
        nodal_pts (np.array : shape(# dof, 2) = reshaped dof.
    """
    n = np.shape(nodal_pts)[0] * np.shape(nodal_pts)[1]
    # flatten out the nodal pts
    # nodal pts new shape = (# dof, 2) with coordinates stored as column vector
    nodal_pts = nodal_pts.reshape(np.shape(grid.nodal_pts)[0] * np.shape(grid.nodal_pts)[1], 2)
    # allocate array for Vandermonde V
    V = np.zeros((n, n))
    # order of distances allocations in V, for an element in the unit square:
    # [dist(-1,-1), dist(1,-1), dist(1,-1), dist(1,1)]
    for i in range(n):
        # fancy evaluations of distances
        V[i, :] = radial_func(metric_2d(nodal_pts[:, 0], nodal_pts[
                              :, 1], nodal_pts[i, 0], nodal_pts[i, 1]), 1)
    return V, nodal_pts


def interpolation_coeff2d(radial_func, l, f_exact, nodal_pts):
    """Calculate the interpolation coefficients in 2d.

    Args:
        radial_func (obj function) = radial function of choice
        l (float) = parameter determining the sharpness of radial function
        f_exact (obj func | np.array : shape (# dof)) = function to be interpolated or
        points from dataset
        nodal_pts (np.array : shape(# dof x, # dof y, 2) = dof with coordinates stored, i.e. nodal_pts[i,j] = [x,y].

    Returns:
        coeffs (np.array) = interpolation coefficients.

    """
    # calculate the Vandermonde matrix and reshape nodal pts
    # nodal pts new shape = (# dof, 2) with coordinates stored as column vector
    V, nodal_pts = construct_vandermonde2d(radial_func, l, nodal_pts)   # nodal_pts[i] = [x_i,y_i]
    if callable(f_exact):   # check if is a function
        f_i = f_exact(nodal_pts[:, 0], nodal_pts[:, 1])
    else:
        assert isinstance(f_exact, np.ndarray), "An array should be provided"
        f_i = f_exact
    coeffs = np.linalg.solve(V, f_i)    # solve linear system
    return coeffs


def reconstruct2d(coeffs, basis):
    """Reconstruc the interpolant from the coefficients and basis.

    Args:
        coeffs (np.array : size = # of dof) = coefficients for linear combination of basis functions
        basis (np.array: shape = (# dof x, # dof y, # pts domain x, # pts domain y)) =
        set of basis function evaluated over the domain.

    Returns:
        interpolant (np.array : shape (# pts domain x, # pts domain y)) = interpolant evaluated over the domain.

    """
    # use the dimension of the domain to structure the interpolant
    interpolant = np.zeros(np.shape(basis)[2:])
    # restructure coeffs with shape (# dof in x, # dof in y)
    # to be able to multiply it with the basis
    coeffs = coeffs.reshape(np.shape(basis)[:2])
    # multiply coeffs and respective basis
    for i in range(np.shape(coeffs)[0]):
        for j in range(np.shape(coeffs)[1]):
            interpolant += coeffs[i, j] * basis[i, j]
    return interpolant


def interpolation_coeff(radial_func, l, x_i, f_exact):
    """Find interpolation coefficients.

    Args:
        radial_func (obj func) = radial function of your choice.
        l (float) = parameter for radial function the higher, the more spicky.
        x_i (np.array) = degrees of freedom.
        f_exact (obj func, np.array)= func to be interpolated or dataset

    Returns:
        coeffs (np.array) = coefficient to be multiplied with the basis.

    """
    # Vandermonde
    V = construct_basis(radial_func, l, x_i, x_i)
    # rhs
    # check if it is a function
    if callable(f_exact):
        f_i = f_exact(x_i)
    else:
        assert isinstance(f_exact, np.darray)
        f_i = f_exact
    # coefficient interpolation_coeff
    coeffs = np.linalg.solve(V, f_i)
    return coeffs


def plot_function2d(func_pts, xx, yy):
    """Plot interpolant function.

    Args:
        xx (np.array) = array of x coordinated created through meshgrid
        yy (np.array) = array of y coordinated created through meshgrid
        interpolant (np.array) = values of the interpolant in the domain.

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, func_pts, rstride=1, cstride=1, linewidth=0, cmap=cm.coolwarm)
    plt.show()


def plot_radial(radial_func, l, xx, yy):
    """Plot interpolant function.

    Args:
        xx (np.array) = array of x coordinated created through meshgrid
        yy (np.array) = array of y coordinated created through meshgrid
        radial func (obj func) = radial function
        l (float) = parameter for sharpness of radial func.

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pp = phi_invquad(metric_2d(xx, yy, 1, 0), l)
    ax.plot_surface(xx, yy, pp, rstride=1, cstride=1, linewidth=0, cmap=cm.coolwarm)
    plt.show()


def plot_interpolation(f, interpolated_f, x, plot_error=False):
    """Plot intepolant and original function side by side.

    Args:
        f (obj func)= original functions.
        interpolated_f (np.array) = interpolant.
        x (np.array) = domain.
        plot_error (bool : optional) = True to plot error
    Returns:
        error_max(float : optional) = max error in the interpolation.

    """
    plt.plot(x, f(x), '-', label='Original function')
    plt.plot(x, interpolated_f, '-', label='Interpolated function')
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend()
    if plot_error:
        error = abs(f(x) - interpolated_f)
        plt.plot(x, error, '-', label='error')
        error_max = np.max(error)
        return error_max
    plt.show()


def reshape_pts(nodal_pts):
    nodal_pts = nodal_pts.reshape(np.shape(grid.nodal_pts)[0] * np.shape(grid.nodal_pts)[1], 2)
    return nodal_pts

# examples
if __name__ == '__main__':
    # # 1d example
    # a, b = -10, 10
    # grid = Grid(a, b, 10)
    # grid.uniform()
    # x = np.linspace(a, b, 101)
    # basis = construct_basis(phi_multiquad, 0.1, grid.nodal_pts, x)
    # coeffs = interpolation_coeff(phi_multiquad, 0.1, grid.nodal_pts, test_func)
    # interpolated_f = coeffs @ basis
    # plot_interpolation(test_func, interpolated_f, x)

    # # 2d example
    # init domain

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)
    # dist = metric_2d(xx, yy, 0, 0)
    # # returns distances of the domain pts from (0,0) in a 11x11 array
    # # ex distance of domain[i,j] wrt to (0,0) is dist[i,j] cool!
    grid = Grid2d((-5, -5), (5, 5), (20, 20))   # generate grid
    grid.uniform()  # uniform grid

    # reshape nodal pts for the discrete interpolation
    nodal_pts_reshaped = reshape_pts(grid.nodal_pts)
    # feed new grid nodal pts to func to generate an array of pts
    test_func_data = test_func2d(nodal_pts_reshaped[:, 0], nodal_pts_reshaped[:, 1])

    # use old setting of nodal pts for coeffs func
    # calculate interp coeff
    coeffs = interpolation_coeff2d(phi_linear, 1, test_func_data, grid.nodal_pts)
    # calculate basis func on the domain
    basis_2d = construct_basis2d(phi_linear, 1, grid.nodal_pts, (xx, yy))
    # reconstruct interpolant
    interpolant = reconstruct2d(coeffs, basis_2d)
    # plotting
    # plot_radial(phi_invquad, 1, xx, yy)
    plot_function2d(interpolant, xx, yy)
    plot_function2d(test_func2d(xx, yy), xx, yy)
