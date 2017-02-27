import matplotlib.pyplot as plt
from legendre_functions import *
from root_finding import newton_method
from functools import partial


class Grid:

    def __init__(self,start,end,num):
        self.a = start
        self.b = end
        self.n = num
        self.nodal_pts = np.empty(num+1)

    def linear_scaling(self):
        """
        linear map : [-1,1] --> [a,b]
        """
        self.nodal_pts = (-self.nodal_pts + 1) * (self.b - self.a) / 2 + self.a

    def uniform(self):
        self.nodal_pts = np.linspace(self.a, self.b, self.n+1)
        return self.nodal_pts

    def chebychev(self):
        """
        Calculates roots of the n+1 Chebychev polynomial of the first king
        returns: nodal pts
        """
        i = np.arange(0, self.n + 1)
        # nodal point in [-1,1]
        self.nodal_pts = np.cos((2 * i + 1) / (2 * (self.n + 1)) * np.pi)
        self.linear_scaling()
        return self.nodal_pts

    def gauss_lobatto(self):
        """
        Calculates the roots of (1+x**2)*L'_n(x)
        :return: nodal pts
        """
        # Chebychev nodes as IV for newton method
        x_0 = np.cos(np.arange(1,self.n)/self.n*np.pi)
        # Last and first pts are fixed for every n
        # This line is introduced to avoid modifying previously calculated nodal_pts
        self.nodal_pts = np.empty(self.n+1)
        self.nodal_pts[-1] = -1
        self.nodal_pts[0] = 1
        for i, ch_pt in enumerate(x_0):
            leg_p = partial(legendre_prime, n=self.n)
            leg_pp = partial(legendre_double_prime_recursive, n=self.n)
            self.nodal_pts[i+1] = newton_method(leg_p, leg_pp, ch_pt, 40)[0]
        self.linear_scaling()
        return self.nodal_pts

    def plot(self, show=True):
        plt.plot(self.nodal_pts, np.zeros(self.n+1), '-o')
        if show:
            plt.show()

    def show(self):
        print('Nodal point of the grid: \n', self.nodal_pts)


def plot_grids(grids, *args, show=True, save=False):
    if np.ndim(grids)>1:
        for i, grid in enumerate(grids):
            if args:
                plt.plot(grid, np.ones(np.size(grid))*i, '-o', label=args[i])
            else:
                plt.plot(grid, np.ones(np.size(grid)) * i, '-o')
    else:
        plt.plot(grids, np.zeros(len(grids)), '-o')
    plt.title(args[-1] + 'for N = {0}' .format(np.size(grids[0])-1))
    plt.xlabel(r'$\xi$')
    plt.ylim(-1, np.ndim(grids)+1)
    plt.legend()
    if save:
        plt.savefig('../Images_numerical/Grid_N_' + str(np.size(grids[0])-1) + '.png', bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    grid_0 = Grid(-1,1,4)

    uni_grid_0 = grid_0.uniform()
    lob_grid_0 = grid_0.gauss_lobatto()
    ch_grid_0 = grid_0.chebychev()

    plot_grids([uni_grid_0, ch_grid_0, lob_grid_0], 'Uniform', 'Chebychev', 'Gauss-Lobatto',
               'Grid comparison', save=True)