# import numpy as np
from numpy import sin, cos, tan, arctan, arcsin, arccos, pi, isnan, isclose, ones, zeros, sqrt, abs, ones_like, zeros_like, degrees, radians, rad2deg, deg2rad, inf, atleast_1d, log
from pygasflow.atd.newton.utils import cotan, arccotan


def elliptic_cone(theta_xy, theta_xz, alpha, beta, l=1, phi_1=0):
    # substitutions to shorten the expressions
    m = tan(theta_xz) / tan(theta_xy)
    s = sin(theta_xz) / sin(theta_xy)
    lamb = cos(alpha) * cos(beta)
    tau = sqrt(1 - lamb**2)
    nu = -sin(beta)
    omega = sin(alpha) * cos(beta)

    Sigma_func = lambda phi: 1 + (m**2 - 1) * sin(phi)**2
    Psi_func = lambda phi: 1 + (m**2 * s**2 - 1) * sin(phi)**2
    Lambda_func = lambda phi: m * sqrt(s**2 - 1) * sin(phi)
    Lambda_p_func = lambda phi: m * sqrt(1 - s**2) * sin(phi)

    def CN_func_1(phi):
        Sigma = Sigma_func(phi)
        Psi = Psi_func(phi)
        Lambda = Lambda_func(phi)
        Lambda_p = Lambda_p_func(phi)
        return(
            (
                lamb**2 * sin(theta_xz)**2 * tan(theta_xy) / sqrt(1 - s**2)
                + s**2 * nu**2 * sin(2 * theta_xy) / (2 * (1 - s**2)**1.5)
                - s**2 * omega**2 * sin(2 * theta_xz) / (2 * m * (1 - s**2)**1.5)
            ) * log((sqrt(Sigma) + Lambda_p) / (sqrt(Sigma) - Lambda_p))
            + (2 * lamb * nu * sin(theta_xz)**2) / (s**2 - 1) * log(Psi / Sigma)
            + ((omega**2 * sin(2 * theta_xz)) / (1 - s**2) - (m * s**2 * nu**2 * sin(2 * theta_xy)) / (1 - s**2)) * sin(phi) / sqrt(Sigma)
            + (2 * lamb * omega * sin(2 * theta_xz) * tan(theta_xy)) / (1 - s**2) * (arctan(cotan(phi) / m) - s * arctan(cotan(phi) / (m * s)))
            - (4 * s * nu * omega * sin(theta_xy) * cos(theta_xz)) / (1 - s**2) * (s / sqrt(1 - s**2) * arctan(cos(phi) * sqrt(1 - s**2) / (s * sqrt(Sigma))) - cos(phi) / sqrt(Sigma))
        )

    def CN_func_2(phi):
        Sigma = Sigma_func(phi)
        Psi = Psi_func(phi)
        Lambda = Lambda_func(phi)
        Lambda_p = Lambda_p_func(phi)
        return(
            (
                2 * lamb**2 * sin(theta_xz)**2 * tan(theta_xy) / sqrt(s**2 - 1)
                - s**2 * nu**2 * sin(2 * theta_xy) / ((s**2 - 1)**1.5)
                + s**2 * omega**2 * sin(2 * theta_xz) / (m * (s**2 - 1)**1.5)
            ) * arctan(Lambda / sqrt(Sigma))
            + (2 * lamb * nu * sin(theta_xz)**2) / (s**2 - 1) * log(Psi / Sigma)
            + ((omega**2 * sin(2 * theta_xz)) / (1 - s**2) - (m * s**2 * nu**2 * sin(2 * theta_xy)) / (1 - s**2)) * sin(phi) / sqrt(Sigma)
            + (2 * lamb * omega * sin(2 * theta_xz) * tan(theta_xy)) / (1 - s**2) * (arctan(cotan(phi) / m) - s * arctan(cotan(phi) / (m * s)))
            - (4 * s * nu * omega * sin(theta_xy) * cos(theta_xz)) / (1 - s**2) * (s / (2 * sqrt(s**2 - 1)) * log((sqrt(Sigma) + cos(phi) * sqrt(s**2 - 1) / s) / (sqrt(Sigma) - cos(phi) * sqrt(s**2 - 1) / s)) - cos(phi) / sqrt(Sigma))
        )

    # eq (24a) and (24b)
    common_1 = s**2 * nu * omega * cos(theta_xy)**2
    common_2 = lamb * m * sin(theta_xz) * sqrt(s**2 * nu**2 * cos(theta_xy)**2 - (lamb**2 * sin(theta_xz)**2 - omega**2 * cos(theta_xz)**2))
    common_3 = lamb**2 * sin(theta_xz)**2 - omega**2 * cos(theta_xz)**2
    phi_i = arccotan((-common_1 - common_2) / common_3)
    phi_f = arccotan((-common_1 + common_2) / common_3)

    l = 1
    S = l**2 * tan(theta_xy) * tan(theta_xz) * (pi / 2 * arctan(cotan(phi_1) / m))

    print("m", m)
    print("s", s)
    if (m < 1) and (s < 1):
        CN = (CN_func_1(phi_f) - CN_func_1(phi_i))
    elif (m > 1) and (s > 1):
        CN = (CN_func_2(phi_f) - CN_func_2(phi_i))
    else:
        raise ValueError(
            "Sorry, can't understand the input parameters."
        )
    return -l**2 / (2 * S) * CN
