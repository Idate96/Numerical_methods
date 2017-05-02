import poisson
import numpy as np
import matplotlib.pyplot as plt
import incidence_matrices
from grid import Grid
from poly_interpolation import PolyBasis
import pickle
import poisson_dual


def multiple_poisson(n_0, n_f, a, b):
    """Run one element poisson for different p values."""
    m = 151
    error_history = np.zeros((n_f - n_0))
    for n in range(n_0, n_f):
        func = poisson.rhs_sin2
        f_exact = poisson.original_f_sin2
        x = np.linspace(a, b, m)
        phi, c, error_norm = poisson.poisson_homo(func, a, b, n, 151, f_exact=poisson.original_f)
        poisson.plot_solution(phi, poisson.original_f, x,
                              'sin2pix', str(n), save=False, show=True)
        error_history[n - n_0] = error_norm
    print(error_history)
    with open('convergence_sin2pix.csv', 'wb') as f_handle:
        np.savetxt(f_handle, error_history)


def run_hp_poisson(a, b, h, p, m):
    """Run multielement direct formulation of the poisson problem"""
    func = poisson.rhs_sin2
    f_exact = poisson.original_f_sin2
    phi, x, error = poisson.hp_poisson(h, p, m, func, a, b, f_exact)
    poisson.plot_solution(phi, f_exact, x,
                          'sin2pix_h' + str(h), str(p), save=False, show=True)


def run_hp_poisson_mixed(a, b, h, p, m):
    """Run multielement version of the mixed formulation of the problem."""
    func = poisson.rhs_sin2
    f_exact = poisson.original_f_sin2
    phi, sigma, error = poisson.mixed_hp_poisson(h, p, a, b, m, func, f_exact)
    x = np.linspace(a, b, len(phi))
    poisson.plot_solution(phi, f_exact, x, 'sin2pix_h' + str(h), str(p), save=False, show=True)
    # poisson.plot_solution(sigma, poisson.d_original_f_sin2_dx, x, 'sin2pix_h' +
    #   str(h), str(p), save=False, show=True)


def convergence_analysis(hs, ps):
    """Progressive convergence analalsis. (only for direct formulation)

    Args:
        hs(tuple) = (h_0, h_f) : h_0 number of element to start with and h_f # el. to stop at.
        ps(tuple) = (p_0, p_f) : initial # of edges, final # of edges
    """
    error_hystory_h = []
    error_hystory_p = []
    func = poisson.rhs_sin2
    f_exact = poisson.original_f_sin2
    a, b = -2, 2
    m = 500
    if hs is not None:
        for h in range(hs[0], (hs[1] + 1)):
            p = 6
            phi, x, error_h = poisson.hp_poisson(h, p, int(m / h), func, a, b, f_exact)
            error_hystory_h.append((h, error_h))
            poisson.plot_solution(phi, f_exact, x,
                                  'sin2pixh-11',  str(h), str(p), save=True, show=False)

        pickle.dump(error_hystory_h, open(
            "data/convergence/convergence_h" + str(hs[0]) + "-" + str(hs[1]) + "-p" + str(p) + ".p", "wb"))
    if ps is not None:
        for p in range(ps[0], ps[1] + 1):
            h = 1
            phi, x, error_p = poisson.hp_poisson(h, p, int(m / h), func, a, b, f_exact)
            error_hystory_p.append((p, error_p))

        pickle.dump(error_hystory_p, open(
            "data/convergence/convergence_p" + str(ps[0]) + "-" + str(ps[1]) + "-h" + str(h) + ".p", "wb"))


def convergence_analysis_specific(hs, ps, type='mixed'):
    """Convergence analysis for specific values of h and p.

    Args:
        hs (np.array/list) = list of h values to be used
        ps (np.array/list) = list of p values to be used
    """
    error_hystory_h = []
    error_hystory_p = []

    # rhs function
    func = poisson.rhs_sin2
    f_exact = poisson.original_f_sin2
    # domain
    a, b = -1, 1
    m = 500

    if hs is not None:
        for h in hs:
            p = 6

            if type == 'dual_direct':
                phi, x, error_h = poisson_dual.direct_dual_poisson_hp(
                    h, p, a, b, int(m / h), func, f_exact)
            if type == 'mixed':
                phi, sigma, error_h = poisson.mixed_hp_poisson(
                    h, p, a, b, int(m / h), func, f_exact)
                x = np.linspace(a, b, len(phi))
            else:
                phi, x, error_h = poisson.hp_poisson(h, p, int(m / h), func, a, b, f_exact)
            error_hystory_h.append((h, error_h))
            poisson.plot_solution(phi, f_exact, x,
                                  'sin2pixh-11',  str(h), str(p), save=False, show=False)

        if type == 'dual_direct':
            pickle.dump(error_hystory_h, open(
                "data/convergence/dual_direct/convergence_h_specific" + str(hs[0]) + "-" + str(hs[-1]) + "-p" + str(p) + ".p", "wb"))
        if type == 'mixed':
            pickle.dump(error_hystory_h, open(
                "data/convergence/mixed/convergence_h_specific" + str(hs[0]) + "-" + str(hs[-1]) + "-p" + str(p) + ".p", "wb"))
        else:
            pickle.dump(error_hystory_h, open(
                "data/convergence/convergence_h_specific" + str(hs[0]) + "-" + str(hs[-1]) + "-p" + str(p) + ".p", "wb"))

    if ps is not None:
        for p in ps:
            h = 1
            if type == 'dual_direct':
                phi, x, error_p = poisson_dual.direct_dual_poisson_hp(
                    h, p, a, b, int(m / h), func, f_exact)

            if type == 'mixed':
                phi, sigma, error_p = poisson.mixed_hp_poisson(
                    h, p, a, b, int(m / h), func, f_exact)
                x = np.linspace(a, b, len(phi))
            else:
                phi, x, error_p = poisson.hp_poisson(h, p, int(m / h), func, a, b, f_exact)
            error_hystory_p.append((p, error_p))
            poisson.plot_solution(phi, f_exact, x, r'exact $f(x) = sin(2 \pi x)$',
                                  str(h), str(p), save=False, show=False)
        if type == 'dual_direct':
            pickle.dump(error_hystory_p, open(
                "data/convergence/dual_direct/convergence_p_specific" + str(ps[0]) + "-" + str(ps[-1]) + "-p" + str(p) + ".p", "wb"))
        if type == 'mixed':
            pickle.dump(error_hystory_p, open(
                "data/convergence/mixed/convergence_p_specific" + str(ps[0]) + "-" + str(ps[-1]) + "-p" + str(p) + ".p", "wb"))
        else:
            pickle.dump(error_hystory_p, open(
                "data/convergence/convergence_p_specific" + str(ps[0]) + "-" + str(ps[-1]) + "-p" + str(p) + ".p", "wb"))

if __name__ == '__main__':

    a, b = -1, 1
    # mixed problem
    # run_hp_poisson_mixed(a, b, h=2, p=8, m=150)
    # poisson.mixed_hp_poisson(3, 2, a, b, 4, poisson.rhs_sin2, poisson.original_f_sin2)

    #
    cases = [15, 20, 25, 50, 100, 200]
    cases_h_specific = list(range(2, 11)) + cases
    # print(cases_h_specific)
    cases_p_specific = list(range(2, 15))
    convergence_analysis_specific(None, cases_p_specific, type='dual_direct')
    error_h = pickle.load(
        open('data/convergence/dual_direct/convergence_h_specific2-200-p6.p', "rb"))
    poisson.plot_convergence(error_h, 'h', 6, type='Dual direct')
    # error_p = pickle.load(open('data/convergence/convergence_p_specific2-4-p4.p', "rb"))
    # error_p_mixed = pickle.load(open('data/convergence/convergence_p_specific2-17-p17.p', "rb"))
    # print("error h", error_h)
    # # print('error_h ', error_h)
    # # print('error_h', error_h)
    # poisson.plot_convergence(error_p, 'p', 6)
    # poisson.plot_convergence(error_p_mixed, 'p', 6, mixed=False)
    # run_hp_poisson(a, b, 3, 4, 50)
    # difference = poisson.reverse_engineer(4, 6, 26, poisson.rhs_sin, a, b, poisson.original_f)
    # n = 3
    # m = 150
    #§ gridi = Grid(a, b, ,in)
    # grid.gauss_lobatto()
    # print("nodal values : ", grid.nodal_pts)
    # x = np.linspace(a, b, m)
    # basis = PolyBasis(x, grid)
    # basis.edge()
    # basis.plot()
    #
    # multiple_poisson(n_0, n_f, a, b)

    # load convergence_sin2pix
    # error_conv = np.loadtxt('data/convergence/convergence_sin2pix.csv')
    # poisson.plot_convergence(error_conv, 2, 'p', save=True)

    # interval domain
    # a, b = -1, 1
    # # number of subelements
    # n = 5
    # # domain resolution
    # m = 151
    # func = poisson.rhs_sin
    # f_exact = poisson.original_f
    # x = np.linspace(a, b, m)
    # phi, c, error_norm = poisson.poisson_homo(func, a, b, n, m, f_exact=poisson.original_f)
    # poisson.plot_solution(phi, poisson.original_f, x,
    #                       'sinpix', str(n), save=True, show=False)
    # # print('Value of FEM solution : \n', phi)
    #
    # plt.plot(x, f_exact(x))
    # plt.plot(x, phi)
    # print("Error norm for n = " + str(n) + ": ", error_norm)
    # plt.show()
