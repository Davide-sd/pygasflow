import itertools
import numpy as np
from pygasflow.utils.decorators import as_array


def cotan(x):
    return 1 / np.tan(x)


def arccotan(x):
    """Computes the inverse function of the cotangent.

    Notes
    -----
    According to Wolfram (see References), there are at least two possible
    conventions to define this function:

    1. ``arccotan(x)`` to have range ``(-pi/2, pi/2]`` and a discontinuity
       at ``x=0``.
    2. ``arccotan(x)`` to have range ``(0, pi)``, thus giving a function that
       is continuous on the real line.

    This implementation follows the second convention.

    References
    ----------
    https://mathworld.wolfram.com/InverseCotangent.html
    """
    return np.pi / 2 - np.arctan(x)


def body2wind(alpha, beta):
    """Rotation matrix from the body frame to the wind frame.

    Parameters
    ----------
    alpha, beta : float
        Angle of attack and side slip angle [radians].

    Returns
    -------
    R : np.ndarray [3 x 3]
        Rotation matrix.
    """
    return np.array([
        [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
        [-np.sin(beta) * np.cos(alpha), np.cos(beta), -np.sin(alpha) * np.sin(beta)],
        [-np.sin(alpha), 0, np.cos(alpha)]])


@as_array([2, 3])
def lift_drag_crosswind(CA, CY, CN, alpha, beta=0):
    """Compute the lift, drag and crosswind coefficients in the wind frame
    starting from the axial, side force, normal coefficients in the body
    frame.

    Parameters
    ----------
    CA, CY, CN : array_like
        Axial, Side force and Normal coefficients in the body frame.
    alpha : array_like
        Angle of attack [radians].
    beta : array_like
        Side slip angle [radians]. Default to 0 (no sideslip).

    Returns
    -------
    CL, CD, CS : array_like
        Lift, Drag and Crosswind coefficients in the wind frame.
    """
    if not hasattr(beta, "__iter__"):
        beta = np.atleast_1d(beta)
    n = max(len(alpha), len(beta))
    body_forces = np.dstack([CA, CY, CN]).reshape((n, 3))

    res = np.zeros((n, 3))
    iterable = lambda x, y: itertools.cycle(x) if len(x) < len(y) else x
    for i, (a, b) in enumerate(zip(iterable(alpha, beta), iterable(beta, alpha))):
        res[i, :] = np.matmul(body2wind(a, b), body_forces[i, :])
    CD, CS, CL = res[:, 0], res[:, 1], res[:, 2]
    return CL, CD, CS
