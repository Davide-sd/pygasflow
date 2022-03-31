import numpy as np

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
