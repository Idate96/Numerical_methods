"""This modules presents functions to calculate incidence matrices."""
import numpy as np


def incidence_m1d(n):
    """Calculate incidence matrix.

    Matrix E_(01) is the boundary operator showing the relationship between nodal point and the edges that connect them. The sense of the transformation is 1 - > 0.
    The coboundary operator, the discrete version of the exterior derivative is the transpose (being the adjoint).
    The inner orientation of the points is assumed to be 'sink'.
    The inner orientation of the edges is assumed positive if directed on the right.
    Returns a n+1 x n matrix with n+1 being the number of nodal points.

    Args:
        n (float) : number of elements

    Returns:
        E_01 (np.array n × n-1) = incidence matrix
    """
    assert (n >= 1), "There should be at least 1 element"
    E_01 = np.zeros((n + 1, n))
    for i in range(1, n):
        E_01[i, i - 1:i + 1] = np.array((1, -1))
    # Account for initial and final points
    E_01[0, 0] = -1
    E_01[-1, -1] = 1
    return E_01
