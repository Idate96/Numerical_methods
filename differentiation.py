"""This modules contains methods to take numerical derivatives"""
import numpy as np
from grid import Grid
import matplotlib.pyplot as plt


def test_func(x):
    return np.sin(x)


def test_func_dx(x):
    return np.cos(x)


def test_func_ddx(x):
    return -np.sin(x)


def forward_diff_1o1(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(len(x) - 1):
        df[i] = (func_pts[i + 1] - func_pts[i]) / (x[i + 1] - x[i])
        # df = (func_pts[1:] - func_pts[:-1]) / (x[1:] - x[:-1])
    return df


def backward_diff_1o1(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(1, len(x)):
        df[i] = (func_pts[i] - func_pts[i - 1]) / (x[i] - x[i - 1])
    return df


def central_diff_1o2(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(1, len(x) - 1):
        df[i] = (func_pts[i + 1] - func_pts[i - 1]) / (x[i + 1] - x[i - 1])
    return df


def forward_diff_1o2(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(len(x) - 2):
        df[i] = (-3 * func_pts[i] + 4 * func_pts[i + 1] - func_pts[i + 2]) / (
            x[i + 2] - x[i])
    return df


def backward_diff_1o2(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(2, len(x)):
        df[i] = (3 * func_pts[i] - 4 * func_pts[i - 1] + func_pts[i - 2]) / (
            x[i] - x[i - 2])
    return df


def backward_diff_1o3(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(2, len(x) - 1):
        df[i] = (2 * func_pts[i + 1] + 3 * func_pts[i] - 6 * func_pts[i - 1] + func_pts[i - 2]) / (
            6 * x[i + 1 - x[i]])
    return df


def central_diff_1o4(func_pts, x):
    df = np.zeros((np.size(x)))
    for i in range(2, len(x) - 2):
        df[i] = 1 / 12 * (func_pts[i - 2] - 8 * func_pts[i - 1] - 8 * func_pts[i + 1]
                          - func_pts[i + 2]) / (x[i + 1] - x[i])
    return df


def central_diff_2o4(func_pts, x):
    ddf = np.zeros((np.size(x)))
    for i in range(2, len(x) - 2):
        ddf[i] = (-func_pts[i + 2] + 16 * func_pts[i + 1] - 30 * func_pts[i] + 16 * func_pts[i - 1]
                  - func_pts[i - 2]) / (12 * (x[i + 1] - x[i]) ** 2)
    return ddf


if __name__ == '__main__':
    x = np.linspace(-5, 5, 1001)
    grid = Grid(-5, 5, 20)
    grid.uniform()
    dfunc_ana = test_func_dx(x)
    dfunc_num_cent = central_diff_1o4(test_func(grid.nodal_pts), grid.nodal_pts)
    dfunc_num_forw = forward_diff_1o1(test_func(grid.nodal_pts), grid.nodal_pts)
    dfunc_num_back = backward_diff_1o1(test_func(grid.nodal_pts), grid.nodal_pts)
    plt.plot(x, dfunc_ana, '-b')
    plt.plot(grid.nodal_pts, dfunc_num_back, 'og', label='backward o(h)')
    plt.plot(grid.nodal_pts, dfunc_num_forw, 'ob', label='forward o(h)')
    plt.plot(grid.nodal_pts, dfunc_num_cent, 'ok', label='central o(h^2)')
    plt.legend()
    plt.show()
