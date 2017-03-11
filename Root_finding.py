"""Module to calculare roots.

The module contains methods to calculate roots of general functions and function to reppresent convergence of algorithms.
Present algorithms: recursive bisection, fixed point methods, newton methos and newton multidimensional method.
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest


def recursive_bisection(f, a, b, min_error=1.e-8, recursion_history=[]):
    '''
    Recursive algorith for root finding: max error is halved at every iteration
    return the root of the function.

    Args:
        param f: function
        param a: lower bound interval
        param b: upper bound interval
        param min_error: accuracy required

    Returns:
        return: root of the function
    '''
    # check sufficient condition for existence of a root
    assert f(a) * f(b) < 0
    # keep track of recursion to display it later on
    recursion_history.append((a, b))
    if f(a) < 0:
        if abs(f(a)) < min_error:
            return a, recursion_history
        else:
            if f((a + b) / 2) < 0:
                return recursive_bisection(f, (a + b) / 2, b)
            else:
                return recursive_bisection(f, a, (a + b) / 2)
    if f(b) < 0:
        if abs(f(b)) < min_error:
            return b, recursion_history
        else:
            if f((a + b) / 2) < 0:
                return recursive_bisection(f, a, (a + b) / 2)
            else:
                return recursive_bisection(f, (a + b / 2), b)


def bisection_convergene(x_reference, recursion_history, func):
    '''
    Plots three errors: approximate error ( x_k - x_final),
    residual error ( func(x_k)) and error estimate ( abs(a-b)/2)

    Args:
        param x_reference: converged solution of the root
        param recursion_history: recursive history [(ak,bk)]
        param func: function to find residual
    '''
    approximate_error, residual_error, error_estimate = np.empty(len(recursion_history)),\
        np.empty(len(recursion_history)),\
        np.empty(len(recursion_history))
    iterations = np.arange(len(recursion_history))
    # calculate errors
    for i, (a, b) in enumerate(recursion_history):
        approximate_error[i] = abs((a + b) / 2 - x_reference)
        residual_error[i] = abs(func((a + b) / 2))
        error_estimate[i] = abs((a - b) / 2)
    # setup plotting
    plt.plot(iterations, np.log(approximate_error), '-o', label=r'$\hat e_k$ - Approximate error')
    plt.plot(iterations, np.log(residual_error), '-o', label=r'$\epsilon$ - Residual error')
    plt.plot(iterations, np.log(error_estimate), '-o', label=r'$E_k$ - Error upper bound')
    plt.xlabel(r'\$n$ - Iteration number')
    plt.ylabel(r'log(e)')
    plt.legend()
    plt.show()


def fixed_point(phi, x_0, n, min_error=1e-8):
    """Fixed point method for rootfinding.

    Iterative algorithm that garantees linear convergence over an interval where abs(ϕ')<1.

    Args:
        phi (obj func) = function whose root is wanted.
        x_0 (float) = Initial point.
        n (int) = max iteration number.
        min_error (float) = minimum allowed error.

    Returns:
        history[-1] (float) = root.
        history (np.array) = history of convergence.
    """
    history = [x_0]
    for i in range(n):
        history.append(phi(history[i]))
        if abs(history[-1] - history[-2]) < min_error:
            print('Convergence at iteration ', i)
            break
    return history[-1], history


def convergence_fixed_pt(phi, history):
    """Display history of convergence.

    Args:
        phi (obj func) = function whose root has been found
        history (np.array) = constains all intermediate points founds with the fixed pt method
    """
    iterations = np.arange(len(history) - 1)
    approximate_e, residual_e = np.empty(len(history) - 1), np.empty(len(history) - 1)
    for i in range(len(history) - 1):
        approximate_e[i] = abs(history[i] - history[-1])
        residual_e[i] = abs(phi(history[i + 1]) - history[i])
    plt.plot(iterations, np.log(approximate_e), '-o', label=r'$e_k$ - Approximate error')
    plt.plot(iterations, np.log(residual_e), '-o', label=r'$\epsilon$ - Residual error')
    plt.legend()
    plt.xlabel(r'$n$ - Iteration number')
    plt.ylabel(r'log(e)')
    plt.show()


def newton_method(f, dfdx, x_0, n_max, min_error=1e-20):
    """Newton method for rootfinding.

    It garantees quadratic convergence given f'(root) != 0 and abs(f'(ξ)) < 1
    over the domain considered.

    Args:
        f (obj func) = function
        dfdx (obj func) = derivative of f
        x_0 (float) = starting point
        n_max (int) = max number of iterations
        min_error (float) = min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(n_max - 1):
        x.append(x[i] - f(x[i]) / dfdx(x[i]))
        if abs(x[i + 1] - x[i]) < min_error:
            return x[-1], x
    return x[-1], x


def convergence_newton(f, history_convergence, x_ref=None):
    """Display history of convergence.

    Args:
        f (obj func) = function whose root has been found
        history_convergence (np.array) = constains all intermediate points
        founds with the fixed pt method
    """
    residual_e, approximate_e = np.empty(
        len(history_convergence) - 1), np.empty(len(history_convergence) - 1)
    iteration = np.arange(len(history_convergence) - 1)
    for i in range(len(history_convergence) - 1):
        approximate_e[i] = abs(history_convergence[i] - history_convergence[-1])
        residual_e[i] = abs(f(history_convergence[i + 1]) - f(history_convergence[i]))
    plt.plot(iteration, np.log(residual_e), '-o', label=r'$\epsilon$ - Residual error')
    plt.plot(iteration, np.log(approximate_e), '-o', label=r'$e_k$ - Approximate error')
    plt.xlabel(r'$n$ - Number iterations'), plt.ylabel(r'log(e)')
    plt.legend(), plt.show()


def newton_multidim(f, dfdx, x_0, n_max, min_error=1e-20):
    """Multidimensional newton methods.

    Docstring equivalent to 1d case.
    """
    x = [x_0]
    assert np.size(f(x_0)) == np.size(x_0)
    assert np.size(dfdx(x_0)) == np.size(x_0) ** 2
    for i in range(n_max - 1):
        x.append(x[i] - np.linalg.solve(dfdx(x[i]), f(x[i])))
        if all(x[i + 1] - x[i]) < min_error:
            return x[-1],  x
    return x[-1], x


# examples
# if __name__ == '__main__':
#     # recursion testing
#     a,b = -2,2
#     x = np.linspace(a,b,100)
#     # plt.plot(x,f_exp(x)), plt.show()
#     # root, recursion_history = recursive_bisection(f_tan,a,b)
#     # bisection_convergene(root,recursion_history,f_tan)
#     # print("Approximated root: {0}" .format(root))
#
#     # fixed pts testing
#     # x_0 = 0.3
#     # a,b = -2,2
#     # fixed_pt, history = fixed_point(phi, x_0, 50)
#     # print('Fixed point ', fixed_pt)
#     # convergence_fixed_pt(phi,history)
#
#     # newton testing
#     # x_0 = 1
#     # root, history = newton_method(f_poly, dfdx_poly, x_0, 30)
#     # convergence_newton(f_poly, history)
#     # print('Approximated root: ', root)
#
#     # plt.plot(x,f_xexp(x)),plt.show()
#     # root, history = newton_method(f_xexp, dfdx_xexp, x_0, 30)
#     # convergence_newton(f_xexp, history)
#     # print('Approximated root: ', root)
#
#     # newton multidim testing
#     x_0 = np.array([-2.0, 1.5])
#     root, history = newton_multidim(f_2d, dfdx_2d, x_0, 30)
#     print('Approximated root: ', root)
#     # print('Value at root ', f_2d(root))
