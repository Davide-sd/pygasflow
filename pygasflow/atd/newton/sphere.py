# import numpy as np
from pygasflow.atd.newton.utils import cotan, arccotan
from numpy import sin, cos, tan, arctan, arcsin, arccos, pi, isnan, isclose, ones, zeros, sqrt, abs, ones_like, zeros_like, degrees, radians, rad2deg, deg2rad, inf, atleast_1d


def sphere_solver(R, alpha, beta, phi_1=0, phi_2=2*pi, sigma=pi):
    """Compute the aerodynamic characteristics over a generalized spheric
    body segment.

    Parameters
    ----------
    R : float
        Radius of the sphere.
    alpha : array_like
        Angle of attack [radians].
    beta : array_like
        Angle of sideslip [radians].
    phi_1 : float, optional
        Initial angle of the slice [radians]. Default to 0.
    phi_2 : float, optional
        Final angle of the slice [radians]. Default to 2*pi.
    sigma : float, optional
        Longitudinal spheric cut-off angle [radians]. Defaulto to pi.
    """
    alpha = atleast_1d(alpha)

    # substitutions to shorten the expressions
    lamb = cos(alpha) * cos(beta)
    tau = sqrt(1 - lamb**2)
    nu = -sin(beta)
    omega = sin(alpha) * cos(beta)

    def sign(x):
        res = ones_like(x)
        res[x < 0] = -1
        return res

    def phi_func(sigma):
        """Solve eq (50) = 0 with Weierstrass substitution."""
        A = lamb * cos(sigma)
        B = nu * sin(sigma)
        C = omega * sin(sigma)

        t = B**2 - (A**2 - C**2)
        # avoid NaN values due to rounding errors inside sqrt
        t[t < 0] = 0

        num1 = -B + sqrt(t)
        num2 = -B - sqrt(t)
        den = A + C
        t1 = num1 / den
        t2 = num2 / den

        # avoid NaN values due to division by 0
        idx = den == 0
        t1[idx] = sign(num1[idx]) * inf
        t2[idx] = sign(num2[idx]) * inf

        phi_i = 2 * arctan(t1)
        phi_f = 2 * arctan(t2) + 2 * pi
        return phi_i, phi_f

    # split equation (54) into manageable pieces
    def CN_func_1(sigma):
        return (
            (1 / 4 * sin(4 * sigma) - sigma) * lamb**2 / 2 * sin(phi_1)
            - sin(phi_1) / 3 * (3 * omega**2 + (nu**2 - omega**2) * sin(phi_1)**2) * (3 * sigma / 2 - sin(2 * sigma) + 1 / 8 * sin(4 * sigma))
            - lamb * omega * (pi - phi_1 - 1/2 * sin(2 * phi_1) * sin(sigma)**4)
        )

    def CN_func_2(sigma):
        phi_1c, phi_2c = phi_func(sigma)

        t = tau**2 - cos(sigma)**2
        # avoid NaN values due to rounding errors inside sqrt
        t[t < 0] = 0

        t2 = cos(sigma) / tau
        # avoid 1+eps or -(1+eps) values inside arcsin
        o = ones_like(t2)
        t2[isclose(t2, o)] = 1
        t2[isclose(t2, -o)] = -1

        num = lamb * cos(sigma)
        t3 = num / sqrt(t)
        idx = t == 0
        t3[idx] = sign(num[idx]) * inf

        return (
            (1 / 4 * sin(4 * sigma) - sigma) * lamb**2 / 4 * sin(phi_1)
            + (nu * omega / 3 * cos(phi_1)**3 - ((nu**2 - omega**2) * sin(phi_1)**2 + 3 * omega**2) * sin(phi_1) / 6) * (3 * sigma / 2 - sin(2 * sigma) + 1 / 8 * sin(4 * sigma))
            + (lamb * nu / 2 * sin(phi_1)**2 - lamb * omega / 2 * (2 * pi - phi_1 - 1 / 2 * sin(2 * phi_1))) * sin(sigma)**4
            - lamb** 3 * nu / (6 * tau**2) * cos(sigma)**4
            - lamb * omega * phi_1c * cos(sigma)**2
            - lamb * omega / 2 * phi_1c * cos(sigma)**4
            + omega / (6 * tau**2) * (3 * tau**2 - 1) * (t)**1.5 * cos(sigma)
            + omega / 2 * (lamb**2 + 1) * sqrt(t) * cos(sigma)
            + omega / 2 * arcsin(t2)
            - lamb * omega / 2 * arctan(t3)
        )

    def CN_func_3(sigma):
        phi_1c, phi_2c = phi_func(sigma)

        t = tau**2 - cos(sigma)**2
        # avoid NaN values due to rounding errors inside sqrt
        t[t < 0] = 0

        t2 = cos(sigma) / tau
        # avoid 1+eps or -(1+eps) values inside arcsin
        o = ones_like(t2)
        t2[isclose(t2, o)] = 1
        t2[isclose(t2, -o)] = -1

        return (
            omega / (3 * tau**2) * (3 * tau**2 - 1) * (t)**1.5 * cos(sigma)
            + omega * (lamb**2 + 1) * sqrt(t) * cos(sigma)
            + lamb * omega * (phi_2c - phi_1c) * cos(sigma)**2
            - lamb * omega / 2 * (phi_2c - phi_1c) * cos(sigma)**4
            + omega * arcsin(t2)
            - lamb * omega * arctan((lamb * cos(sigma)) / sqrt(t))
        )

    def sigma_func(phi):
        # eq (61)
        return arccotan((-nu * sin(phi) + omega * cos(phi)) / lamb)

    sigma_1 = pi / 2 - arctan(tau / lamb)
    sigma_2 = sigma_func(phi_1)
    sigma_3 = sigma_func(phi_2)
    sigma_4 = pi / 2 + arctan(tau / lamb)

    # body corrections
    sigma_1[sigma_1 > sigma] = sigma
    sigma_2[sigma_2 > sigma] = sigma
    sigma_3[sigma_3 > sigma] = sigma
    sigma_4[sigma_4 > sigma] = sigma

    S = pi * R**2
    CN = -R**2 / S * (
        (CN_func_1(sigma_2) - CN_func_1(0))
        + (CN_func_2(sigma_3) - CN_func_2(sigma_2))
        + (CN_func_3(sigma_4) - CN_func_3(sigma_3))
    )
    return CN
