"""This modules contains classes for computational grid generation."""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from legendre_functions import legendre_prime, legendre_double_prime_recursive
from root_finding import newton_method


class Grid(object):
    """Computational grid.

    Attributes:
         a (float) = start domain
         b (float) = end.domain
         n (int)  = number of points - 1
         nodal_pts (np.array) = array containing the nodal points.
    """

    def __init__(self, start, end, num):
        """Initialize grid.

        Args:
            start (float) = start domain.
            end (float) = end domain.
            num (int)  = number of points - 1.
        """
        self.a = start
        self.b = end
        self.n = num
        self.nodal_pts = np.empty(num + 1)

    def linear_scaling(self):
        """Linear map : [-1,1] --> [a,b]."""
        self.nodal_pts = (-self.nodal_pts + 1) * (self.b - self.a) / 2 + self.a

    def uniform(self):
        """Generate a uniform grid."""
        self.nodal_pts = np.linspace(self.a, self.b, self.n + 1)
        return self.nodal_pts

    def chebychev(self):
        """Calculate roots of the n+1 Chebychev polynomial of the first kind.

        Returns:
            nodal_pts (np.array)
        """
        i = np.arange(0, self.n + 1)
        # nodal point in [-1,1]
        self.nodal_pts = np.cos((2 * i + 1) / (2 * (self.n + 1)) * np.pi)
        self.linear_scaling()
        return self.nodal_pts

    def gauss_lobatto(self):
        """Calculate the roots of (1+x**2)*L'_n(x) and populate nodal_pts.

        Returns:
            nodal_pts (np.array)
        """
        # Chebychev nodes as initial value for newton method
        x_0 = np.cos(np.arange(1, self.n) / self.n * np.pi)
        self.nodal_pts = np.empty(self.n + 1)
        # Last and first pts are fixed for every n
        self.nodal_pts[-1] = -1
        self.nodal_pts[0] = 1
        # Newton's method to find the root
        for i, ch_pt in enumerate(x_0):
            leg_p = partial(legendre_prime, n=self.n)
            leg_pp = partial(legendre_double_prime_recursive, n=self.n)
            self.nodal_pts[i + 1] = newton_method(leg_p, leg_pp, ch_pt, 40)[0]
        # scale to arbitrary domain
        self.linear_scaling()
        return self.nodal_pts

    def plot(self, show=True):
        """Plot the grid."""
        plt.plot(self.nodal_pts, np.zeros(self.n + 1), '-o')
        if show:
            plt.show()

    def show(self):
        # TODO(modify it to __str__ method)
        print('Nodal point of the grid: \n', self.nodal_pts)


def plot_grids(grids, *args, show=True, save=False):
    """Plot multiple grids.

    Args:
        grids (list) = list of Grids
        args = label for the plot, grid ith receive label ith
    """
    # add the grids to the plot with labels
    if np.ndim(grids) > 1:
        for i, grid in enumerate(grids):
            if args:
                plt.plot(grid, np.ones(np.size(grid)) * i, '-o', label=args[i])
            else:
                plt.plot(grid, np.ones(np.size(grid)) * i, '-o')
    else:
        plt.plot(grids, np.zeros(len(grids)), '-o')
    plt.title(args[-1] + "for N = {0}" .format(np.size(grids[0]) - 1))
    plt.xlabel(r'$\xi$')
    plt.ylim(-1, np.ndim(grids) + 1)
    plt.legend()
    if save:
        plt.savefig('imamges/Grid_N_' +
                    str(np.size(grids[0]) - 1) + '.png', bbox_inches='tight')
    if show:
        plt.show()


class Grid2d(object):
    """2D computational grid.

    Attributes:
         xx (2d np.array) = meshgrid of x interval.
         yy (2d np.array) = meshgrid of y interval.
         n (tuple)  = (n_x,n_y) where n_x is the number of points in the x direction and
         n_y is the number of points in the y direction.
         grid1d (np.array) = array of two elements, in order 1d grid in x and 1d grid in y.
         nodal_pts (3d np.array) = array containing the nodal points.
         nodal_pts[i,j] returns the coordinates [x_i,y_j] of the nodal point.
    """

    def __init__(self, start, end, n):
        """Initialize 2d grid.

        Args:
            start (tuple) = (a_x,a_y), tuple of floats determining start of the domain in x and y.
            end (tuple) = (b_x,b_y), tuple of floats determining end of the domain in x and y.
            n (tuple)  = (n_x,n_y) where n_x is the number of points in the x direction and
            n_y is the number of points in the y direction.
        """
        self.n = n
        self.nodal_pts = None
        self.grid1d = np.array((Grid(start[0], end[0], n[0]), Grid(start[1], end[1], n[1])))
        self.xx = None
        self.yy = None

    def uniform(self):
        """Generate a uniform 2d grid."""
        # generate 1d uniform grid
        for grid in self.grid1d:
            grid.uniform()
        self.xx, self.yy = np.meshgrid(self.grid1d[0].nodal_pts, self.grid1d[1].nodal_pts)
        self.nodal_pts = np.dstack((self.xx, self.yy))

    def chebychev(self):
        """Generate a chebychev 2d grid."""
        # generate 1d chebychev grids
        for grid in self.grid1d:
            grid.chebychev()
        self.xx, self.yy = np.meshgrid(self.grid1d[0].nodal_pts, self.grid1d[1].nodal_pts)
        # store 3d array the nodal pts, nodal_pts[i,j] returns [x_i,y_j]
        self.nodal_pts = np.dstack((self.xx, self.yy))

    def gauss_lobatto(self):
        """Generate a gauss_lobatto 2d grid."""
        # generate 1d gauss_lobatto grids.
        for grid in self.grid1d:
            grid.gauss_lobatto()
        self.xx, self.yy = np.meshgrid(self.grid1d[0].nodal_pts, self.grid1d[1].nodal_pts)
        self.nodal_pts = np.dstack((self.yy, self.xx))

    def plot(self):
        """Plot the grid."""
        # plotting through scattering
        plt.scatter(self.xx, self.yy, s=10)
        plt.title('Computational grid')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    def __str__(self):
        """New reprensentation for the grid."""

        nodalpts = "The nodal points are \n" + str(self.nodal_pts)
        description = "\nThe grid is organized as follows:\ngrid[0,0] = " + str(self.nodal_pts[0, 0]) + "\ngrid[0,-1] = " + str(self.nodal_pts[0, -1]) + "\ngrid[-1,0] = " + \
            str(self.nodal_pts[-1, 0]) + "\ngrid[-1,-1] = " + str(self.nodal_pts[-1, -1])
        return nodalpts + description


# examples

    # grid_0 = Grid(-1, 1, 4)
    #
    # uni_grid_0 = grid_0.uniform()
    # lob_grid_0 = grid_0.gauss_lobatto()
    # ch_grid_0 = grid_0.chebychev()
    #
    # plot_grids([uni_grid_0, ch_grid_0, lob_grid_0], 'Uniform',
    #            'Chebychev', 'Gauss-Lobatto', 'Grid comparison', save=False)

    # grid2d = Grid2d((-1, -1), (1, 1), (4, 4))
    # grid2d.gauss_lobatto()
    # grid2d.plot()
    # print(grid2d)

    # grid2d.plot()
    # print(np.shape(grid2d.nodal_pts))
    # print(grid2d.nodal_pts[0, 0], grid2d.nodal_pts[1, 0], grid2d.nodal_pts[-1, -1])
    # print(grid2d.nodal_pts)
