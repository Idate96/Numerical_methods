"""This module contains explicit and implic methods for odes integration."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def pendulum_ode(u, t, g, l, c):
    theta, omega = u
    dydt = np.array((omega, -g / l * np.sin(theta) - c * omega))
    return dydt


def my_pendulum_ode(u):
    theta, omega = u
    dydt = np.array((omega, -9.81 / 1 * np.sin(theta) - 0.1 * omega))
    return dydt


def forward_euler(f, u, dt):
    """Explicit forward euler scheme."""
    if callable(f):
        u = u + dt * (f(u))
    elif isinstance(f, (float, np.ndarray)):
        u = u + dt * (f)
    return u


def backward_euler(f, J, u, dt):
    """Backward euler scheme."""
    delta_u = np.linalg.solve(np.eye(np.shape(J)) - dt * J(u), dt * f(u))
    return u + delta_u


def adam_bashfort(f, u, u_0, dt):
    """Adam bashfort 2 step method.

    Args:
        f (obj func) = function coming from the ODE u' = f(u)
        u (np.ndarray) = state vector at t_i
        u_0 (np.ndarray) = state vector at t_i-1
        dt (float) = time step

    Returns:
        u (np.ndarray) = state vector at t_i+1
    """
    if callable(f):
        u = u + dt * (3 / 2 * f(u) - f(u_0))
    elif isinstance(f, (float, np.ndarray)):
        u = u + dt * (3 / 2 * f * u - f * u_0)
    return u


def rk4(f, u, dt):
    """
    Runge-Kutta explicit 4-stage scheme - 1 step.

      f  - function defining ODE dy/dt = f(t,u), two arguments
      t  - time
      u  - solution at t
      dt - step size
    Return: approximation of y at t+dt.
    """
    if callable(f):
        k1 = f(u)
        k2 = f(u + dt * k1 / 2)
        k3 = f(u + dt * k2 / 2)
        k4 = f(u + dt * k3)
    elif isinstance(f, (float, np.ndarray)):
        k1 = f * u
        k2 = f * (u + dt * k1 / 2)
        k3 = f * (u + dt * k2 / 2)
        k4 = f * (u + dt * k3)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == '__main__':
    # alpha = -0.5
    # M = 1
    # N, deltat = 100, 0.1
    # Tmax = N * deltat
    # u = np.zeros((N, M))
    # u[0] = [1]     # Initial condition
    # for i in range(N - 1):
    #     u[i + 1] = rk4(alpha, u[i], deltat)
    #
    # t = np.linspace(0, Tmax, N)
    #
    # # plt.figure()
    # # plt.plot(t, u, '-b', label=r'$\phi$')
    # # plt.legend()
    # # plt.show()
    #

    # example ode_int
    g = 9.81
    l = 1
    c = 0.1
    dt = 0.01
    t_0 = 0
    t_f = 10
    y_0 = np.array((0, 1))
    t = np.linspace(t_0, t_f, (t_f - t_0) / dt + 1)

    sol = odeint(pendulum_ode, y_0, t, args=(g, l, c))

    sol_1 = np.zeros((len(t), 2))
    sol_1[0, :] = y_0
    for i in range(len(t) - 1):
        sol_1[i + 1] = forward_euler(my_pendulum_ode, sol_1[i], dt)

    plt.plot(t, sol_1[:, 0], 'k', label='theta(t) rk4')
    # plt.plot(t, sol_1[:, 1], 'y', label='omega(t) rk4')
    plt.plot(t, sol[:, 0], 'b', label='theta(t)')
    # plt.plot(t, sol[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()
