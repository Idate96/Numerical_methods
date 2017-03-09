from poisson import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a, b = -1, 1
    n = 5
    m = 51
    func = rhs_sin
    f_exact = original_f
    x = np.linspace(a, b, m)
    phi = poisson_homo(func, a, b, n, m)
    print('Value of FEM solution : \n', phi)
    plt.plot(x, f_exact(x))
    plt.plot(x, phi)
    plt.show()
