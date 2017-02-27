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

    def edge(self):
        self.basis = np.ones((self.n, len(self.x)))
        # lagrange poly
        self.lagrange()

        # derivative lagrange poly
        np.seterr(divide='ignore', invalid='ignore')
        for j in range(self.n):
            self.basis[j] = 1/(x-self.nodal_pts[j]) * (P_prime(self.x,self.nodal_pts)/
                                                 P_prime(self.nodal_pts[j],self.nodal_pts)-self.basis[j])
            # Since the pts in the interval coincide with nodal_pts[j]
            # we have to add the limit of the function
            limit_value = 0
            for l in range(self.n):
                if l != j:
                    limit_value += 1/(self.nodal_pts[j]-self.nodal_pts[l])
            self.basis[j, np.where(abs(self.x - self.nodal_pts[j]) < 1.e-10)] = limit_value

        # calculation of the edge poly
        edge = np.zeros((self.n, len(self.x)))
        for i in range(1,self.n):
            for k in range(0,i):
                edge[i] += self.basis[k]
        self.basis = -edge

    def lagrange_1(self):
        for i in range(self.n):
            np.seterr(divide='ignore', invalid='ignore')
            self.basis[i] = P(self.x,self.nodal_pts)/\
                                (P_prime(self.nodal_pts[i],self.nodal_pts)*(self.x-self.nodal_pts[i]))
            self.basis[i, np.where(abs(self.x - self.nodal_pts[i]) < 1.e-10)] = 1

    def plot(self, *args, save=False):
        y_min, y_max = 0,0
        for base in self.basis:
            plt.plot(self.x,base)
            if y_min > np.min(base):
                y_min = np.min(base)
            if y_max < np.max(base):
                y_max = np.max(base)
        plt.ylim(y_min*1.1,y_max*1.1)
        if args:
            plt.title(args[0] + ' interpolation')
            plt.ylabel(args[1])
        plt.xlabel(r'$\xi$')
        if save:
            plt.savefig('../Images_numerical/'+args[0]+'_N_'+str(self.n-1)+'.png', bbox_inches='tight')
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



def P(x, nodal_pts):
    ans = 1
    for i, nodal_pt in enumerate(nodal_pts):
        ans *= (x-nodal_pt)
    return ans


def P_prime(x, nodal_pts):
    ans = 0
    for n in range(len(nodal_pts)):
        prod = 1
        for l in range(len(nodal_pts)):
            if l != n:
                prod *= (x-nodal_pts[l])
        ans += prod
    return ans


if __name__ == '__main__':
    a,b = -1,1
    x = np.linspace(a,b, 101)
    #
    grid = Grid(a,b,4)
    grid.gauss_lobatto()
    #
    basis = PolyBasis(x,grid)
    basis.lagrange()
    basis.plot('GL nodal', r'$l_i(\xi)$', save=True)

    basis_1 = PolyBasis(x,grid)
    basis_1.edge()
    basis_1.plot('GL edge', r'$e_i(\xi)$', save=True)

    # basis_e = basis_edge(x,grid.nodal_pts)
    # plot_basis(x,basis_e,grid.nodal_pts)

    # basis = PolyBasis(x,grid)
    # basis.lagrange()
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
    # funcs = [f,f_poly5,f_abs,f_abs3,f_gauss,f_runge,f_step]
    # plot_funcs(funcs)