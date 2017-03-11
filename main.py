import poisson
import numpy as np
import matplotlib.pyplot as plt
import incidence_matrices


def multiple_poisson(n_0, n_f):
    a, b = -1, 1
    m = 151
    error_history = np.zeros((n_f - n_0))
    for n in range(n_0, n_f):
        # number of subelements
        # domain resolution
        func = poisson.rhs_sin
        f_exact = poisson.original_f
        x = np.linspace(a, b, m)
        phi, c, error_norm = poisson.poisson_homo(func, a, b, n, m, f_exact=poisson.original_f)
        poisson.plot_solution(phi, poisson.original_f, x,
                              'sin2pix', str(n), save=False, show=True)
        error_history[n - n_0] = error_norm
    print(error_history)
    with open('convergence_sin2pix.csv', 'wb') as f_handle:
        np.savetxt(f_handle, error_history)


if __name__ == '__main__':
    n_0 = 5
    n_f = 7
    multiple_poisson(n_0, n_f)

    # load convergence_sin2pix
    # error_conv = np.loadtxt('data/convergence_sin2pix.csv')
    # poisson.plot_convergence(error_conv, n_0)

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
