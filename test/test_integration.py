import unittest
from scipy import integrate
import numpy as np
# change path to project home
import sys
sys.path[0] += '/../'
# list directory files
# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(sys.path[0]) if isfile(join(sys.path[0], f))]
from integration import quad_glob


def f(x):
    return np.sin(np.pi * x / 2.)


def f_poly5(x):
    return 0.001 * x ** 5 + 0.02 * x ** 3 - x  # Polynomial


def f_step(x):
    return np.sign(x - 1)                 # Discontinuous at x=2


# Discontinuous derivative at x=2
def f_abs(x):
    return np.abs(x - 2)


# Discontinuous 3rd derivative at x=2
def f_abs3(x):
    return np.abs((x - 2) ** 3)

# Infinitely differentiable (everywhere)


def f_runge(x):
    return 1. / (1 + (4. * x) ** 2)


def f_gauss(x):
    return np.exp(-(x - 2) ** 2 / 2.)


class Test_quad_glob(unittest.TestCase):

    def test(self):
        funcs = [f_poly5, f_runge, f_gauss, f_abs3, f_abs, f]
        for i, func in enumerate(funcs):
            self.assertAlmostEqual(integrate.quad(func, -1, 1)
                                   [0], quad_glob([func], -1, 1, 15), msg='Failed at func ' + str(i), places=2)


if __name__ == '__main__':
    print(integrate.quad(f_runge, -1, 1))
    print(quad_glob([f_runge], -1, 1, 10))  # less accurate as not as precise
    unittest.main()
