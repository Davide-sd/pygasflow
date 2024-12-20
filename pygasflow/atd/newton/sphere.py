# import numpy as np
from pygasflow.utils.common import ret_correct_vals, _should_solver_return_dict
from pygasflow.atd.newton.utils import cotan, arccotan, lift_drag_crosswind
from numpy import sin, cos, tan, arctan, arcsin, arccos, pi, isnan, isclose, ones, zeros, sqrt, abs, ones_like, zeros_like, degrees, radians, rad2deg, deg2rad, inf, atleast_1d
from scipy.integrate import dblquad


###############################################################################
###################### ANALYTICAL SOLUTIONS UP TO SIGMA_2 #####################
###############################################################################

# NOTE: the solutions given in the paper doesn't appear to be correct.
# The following solutions have been computed with SymPy, with limits of
# integration (phi, phi_1, phi_2) and (sigma, 0, sigma_c). The results of
# the integration have been simplified and then `cse` has been used to
# optmize the code.
#
# In particular, the results of the paper assume that `phi_2=2*pi - phi_1`.
# Still, there are errors in the analytical solutions. For example,
# expression (56), the second term inside from 0 to sigma_2, it is missing a
# multiplication. Results from page 70 confirms that they have been computed
# with such errors.
#
# From sigma_2 to sigma_3 and from sigma_3 to sigma_4, numerical integration
# has been used (I'm too NOOB to find analytical solution in these regions).
# The results are in good agreement with the paper for sigma_cut <= 90deg.
# For 90deg < sigma_cut <= 180deg the results are different: while the trends
# are captured, numerical integration appears to compute sligthly lower values.
# After checking and re-checking my procedure, I have confidence that my
# results are correct.


def CN_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_c):
    x0 = sigma_c / 4
    x1 = sin(phi_1)
    x2 = x1**3
    x3 = sin(phi_2)
    x4 = x3**3
    x5 = x2 - x4
    x6 = -nu**2 * x5
    x7 = sin(sigma_c)
    x8 = lamb * x7**4
    x9 = cos(phi_2)
    x10 = cos(phi_1)
    x11 = cos(3 * phi_1) - cos(3 * phi_2)
    x12 = nu * omega
    x13 = omega**2
    x14 = 2 * phi_1
    x15 = 2 * phi_2
    x16 = 2 * sigma_c
    x17 = sin(x16)
    x18 = x12 * (3 * x10 + x11 - 3 * x9)
    x19 = x1 - x3
    x20 = x13 * (3 * x19 - x2 + x4)
    x21 = x7**3 * cos(sigma_c)
    x22 = 4 * sigma_c
    x23 = -x22

    return (
        lamb**2 * (-8 * sigma_c * x19 - cos(phi_1 + x22) + cos(phi_1 + x23) + cos(phi_2 + x22) - cos(phi_2 + x23)) / 32
        + nu * x8 * (-x1**2 + x3**2) / 2
        + omega * x8 * (x14 - x15 + sin(x14) - sin(x15)) / 4
        + sigma_c * x12 * (-3 * x10 - x11 + 3 * x9) / 8
        + x0 * x13 * (-3 * x1 + 3 * x3 + x5)
        + x0 * x6 + x17 * x18 / 16 + x17 * x20 / 8
        + x17 * x6 * (cos(x16) - 4) / 24 + x18 * x21 / 12 + x20 * x21 / 6
    )


def CA_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_c):
    x0 = nu / 2
    x1 = lamb * sigma_c
    x2 = sin(phi_1)
    x3 = sin(phi_2)
    x4 = sin(sigma_c)**4
    x5 = 2 * phi_1
    x6 = sin(x5)
    x7 = sin(2 * phi_2)
    x8 = -2 * phi_2 + x5
    x9 = x4 / 8
    x10 = 4 * sigma_c
    x11 = -x10
    x12 = phi_1 + x11
    x13 = phi_2 + x10
    x14 = phi_1 + x10
    x15 = phi_2 + x11
    x16 = lamb / 16

    return (
        lamb**2 * (phi_1 - phi_2) * (cos(sigma_c)**4 - 1) / 2
        + nu**2 * x9 * (x6 - x7 - x8)
        + nu * x16 * (sin(x12) + sin(x13) - sin(x14) - sin(x15))
        + omega**2 * x9 * (-x6 + x7 - x8)
        + omega * x0 * x4 * (x2**2 - x3**2)
        + omega * x1 * (x2 - x3) / 2
        + omega * x16 * (-cos(x12) - cos(x13) + cos(x14) + cos(x15))
        + x0 * x1 * (cos(phi_1) - cos(phi_2))
    )

def CY_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_c):
    x0 = lamb**2
    x1 = cos(phi_1)
    x2 = cos(phi_2)
    x3 = x1 - x2
    x4 = lamb * sin(sigma_c)**4
    x5 = 2 * phi_1
    x6 = nu**2
    x7 = -cos(3 * phi_1) + cos(3 * phi_2)
    x8 = 4 * sigma_c
    x9 = 2 * sigma_c
    x10 = sin(x9)
    x11 = 12 * sigma_c - 8 * x10 + sin(x8)
    x12 = -x8

    return (
        nu * omega * x11 * (sin(phi_1)**3 - sin(phi_2)**3) / 24
        + nu * x4 * (2 * phi_2 - x5 - sin(2 * phi_2) + sin(x5)) / 4
        + omega**2 * x11 * (x1**3 - x2**3) / 48
        + omega * x4 * (-x1**2 + x2**2) / 2
        + sigma_c * x0 * x3 / 4
        + sigma_c * x6 * (9 * x3 + x7) / 16
        + x0 * (sin(phi_1 + x12) - sin(phi_1 + x8) - sin(phi_2 + x12) + sin(phi_2 + x8)) / 32
        + x10 * x6*(cos(x9) - 4) * (9 * x1 - 9 * x2 + x7) / 96
    )


###############################################################################
############################ NUMERICAL INTEGRATION ############################
###############################################################################


def Cp_func(lamb, nu, omega, sigma, phi):
    # equation (50)
    return 2 * (lamb * cos(sigma) + nu * sin(sigma) * sin(phi) - omega * sin(sigma) * cos(phi))**2


def CN_func_to_integrate(phi, sigma, lamb, nu, omega):
    # equation (51)
    Cp = Cp_func(lamb, nu, omega, sigma, phi)
    return Cp * sin(sigma)**2 * cos(phi)


def CA_func_to_integrate(phi, sigma, lamb, nu, omega):
    # equation (52) (note that it is wrongly written in the paper).
    Cp = Cp_func(lamb, nu, omega, sigma, phi)
    return Cp * cos(sigma) * sin(sigma)


def CY_func_to_integrate(phi, sigma, lamb, nu, omega):
    # equation (53)
    Cp = Cp_func(lamb, nu, omega, sigma, phi)
    return Cp * sin(sigma)**2 * sin(phi)


###############################################################################
################################## SOLVER #####################################
###############################################################################


def sign(x):
    res = ones_like(x)
    res[x < 0] = -1
    return res


def phi_func(lamb, nu, omega, sigma):
    """Solve eq (50) = 0 with Weierstrass substitution."""
    A = lamb * cos(sigma)
    B = nu * sin(sigma)
    C = omega * sin(sigma)

    t = atleast_1d(B**2 - (A**2 - C**2))
    # avoid NaN values due to rounding errors inside sqrt
    t[isclose(t, 0) & (t < 0)] = 0

    num1 = -B + sqrt(t)
    num2 = -B - sqrt(t)
    den = A + C
    t1 = num1 / den
    t2 = num2 / den

    # avoid NaN values due to division by 0
    idx = den == 0
    t1[idx] = sign(num1[idx]) * inf
    t2[idx] = sign(num2[idx]) * inf

    phi_1 = 2 * arctan(t1)
    phi_2 = 2 * arctan(t2) + 2 * pi
    return phi_1, phi_2


def sphere_solver(R, alpha, beta=0, phi_1=0, phi_2=2*pi, sigma_cut=pi, to_dict=False):
    """Compute axial/normal/lift/drag/moments coefficients over a sphere or a
    generalized spheric body segment.

    Parameters
    ----------
    R : float
        Radius of the sphere.
    alpha : array_like
        Angle of attack [radians].
    beta : array_like
        Angle of sideslip [radians]. Default to 0 (no sideslip).
    phi_1 : float, optional
        Initial angle of the slice [radians]. Default to 0.
    phi_2 : float, optional
        Final angle of the slice [radians]. Default to 2*pi.
    sigma_cut : float, optional
        Longitudinal spheric cut-off angle [radians]. Defaulto to pi.
    to_dict : bool, optional
        Default value to False, which would return a tuple of results. If True,
        a dictionary will be returned, whose keys are listed in the Returns
        section.

    Returns
    -------
    CN : float or array_like
        Normal force coefficient.
    CA : float or array_like
        Axial force coefficient.
    CY : float or array_like
        Side-force coefficient.
    CL : float or array_like
        Lift coefficient.
    CD : float or array_like
        Drag coefficient.
    CS : float or array_like
        Crosswind coefficient.
    L/D : float or array_like
        Lift/Drag ratio.
    Cl : float or array_like
        Rolling-moment coefficient.
    Cm : float or array_like
        Pitching-moment coefficient.
    Cn : float or array_like
        Yawing-moment coefficient.

    References
    ----------

    * "Tables of aerodynamic coefficients obtained from developed newtonian
      expressions for complete and partial conic and spheric bodies at combined
      angle of attack and sideslip with some comparison with hypersonic
      experimental data", by William R. Wells and William O. Armstrong, 1962.

    See Also
    --------
    pressure_coefficient, sharp_cone_solver

    Examples
    --------

    Compute the aerodynamic characteristics for a spherical cap of radius 1,
    for different angle of attacks with no sideslip:

    >>> import numpy as np
    >>> from pygasflow.atd.newton.sphere import sphere_solver
    >>> alpha = np.linspace(0, np.pi / 2, 10)
    >>> sigma_cut = np.deg2rad(30)
    >>> res = sphere_solver(1, alpha, sigma_cut=sigma_cut, to_dict=True)

    Compute the aerodynamic characteristics for a slice of the previous
    spherical cap, with ``phi_1=90 deg`` and ``phi_2=270 deg``:

    >>> res = sphere_solver(1, alpha, phi_1=np.pi/2, phi_2=1.5*np.pi,
    ...     sigma_cut=sigma_cut, to_dict=True)

    Notes
    -----

    The reference system is:

    .. code-block:: text

                -z
                 |
                 |
       y ---------
                /|
               / |
              /  |
           x /   |
                 z

    ``phi_1`` and ``phi_2`` are the angles starting from the -z axis, rotating
    counter-clockwise around the x-axis. ``sigma_cut`` represents the angle of
    rotation around the y-axis starting from the +x axis.

    NOTE: Performance of computation depends on ``sigma_cut``, ``alpha`` and
    ``beta``. For small values of these parameters the computation is likely
    to be done by the analytical solution (hence very fast). As we increase
    these values, parts of the integration will be done by numerical
    procedures, hence slow computations.
    """
    to_dict = _should_solver_return_dict(to_dict)
    alpha = atleast_1d(alpha)
    if (sigma_cut < 0) or (sigma_cut > pi):
        raise ValueError("It must be: 0 <= sigma_cut <= pi")

    # substitutions to shorten the expressions
    lamb = cos(alpha) * cos(beta)
    tau = sqrt(1 - lamb**2)
    nu = -sin(beta)
    omega = sin(alpha) * cos(beta)

    t = -nu / tau
    # avoid NaN values inside arcsin
    t[isclose(t, 1) & (t > 1)] = 1
    t[isclose(t, -1) & (t < -1)] = -1
    phi_p = arcsin(t)
    phi_p[isnan(phi_p)] = 0

    def sigma_func(phi):
        # eq (61)
        return arccotan((-nu * sin(phi) + omega * cos(phi)) / lamb)

    sigma_1 = pi / 2 - arctan(tau / lamb)
    sigma_2 = sigma_func(phi_1)
    sigma_3 = sigma_func(phi_2)
    sigma_4 = pi / 2 + arctan(tau / lamb)

    # depending on the body geometry and wind parameters, it might be happens
    # that sigma_2 > sigma_3. Let's fix that.
    tmp = sigma_2.copy()
    idx = sigma_2 > sigma_3
    sigma_2[idx] = sigma_3[idx]
    sigma_3[idx] = tmp[idx]

    # # For the generalized spheric segment where phi_1 > 90deg, there are some
    # # combinations of alpha and beta which cause P4 to fall outside the limits
    # # of the body. For this case P3 and not P4 becomes the last point of
    # # shadow.
    # idx = phi_p + pi > phi_2
    # sigma_4[idx] = sigma_3[idx]

    # body corrections
    sigma_1[sigma_1 > sigma_cut] = sigma_cut
    sigma_2[sigma_2 > sigma_cut] = sigma_cut
    sigma_3[sigma_3 > sigma_cut] = sigma_cut
    sigma_4[sigma_4 > sigma_cut] = sigma_cut

    # analytical solution from sigma=0 to sigma=sigma_2
    CA_A = CA_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_2)
    CN_A = CN_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_2)
    CY_A = CY_up_to_sigma_2(lamb, nu, omega, phi_1, phi_2, sigma_2)

    def upper_limit(l, n, o, sigma):
        # take into account the body geometry to define the upper limit
        phi_lim = phi_func(l, n, o, sigma)[1]
        if phi_lim > phi_2:
            return phi_2
        return phi_lim

    # perform numerical integration in the region limited by the shadow plane
    nu = nu * ones(len(alpha))
    CN_B = zeros_like(CN_A)
    CN_C = zeros_like(CN_A)
    CA_B = zeros_like(CA_A)
    CA_C = zeros_like(CA_A)
    CY_B = zeros_like(CY_A)
    CY_C = zeros_like(CY_A)
    for i, (l, n, o) in enumerate(zip(lamb, nu, omega)):
        CY_A[i] = dblquad(CY_func_to_integrate, 0, sigma_2[i], lambda sigma: phi_1, lambda sigma: phi_2, args=(l, n, o))[0]
        if not isclose(sigma_2[i], sigma_3[i]):
            CN_B[i] = dblquad(CN_func_to_integrate, sigma_2[i], sigma_3[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: upper_limit(l, n, o, sigma), args=(l, n, o))[0]
            CA_B[i] = dblquad(CA_func_to_integrate, sigma_2[i], sigma_3[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: upper_limit(l, n, o, sigma), args=(l, n, o))[0]
            CY_B[i] = dblquad(CY_func_to_integrate, sigma_2[i], sigma_3[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: upper_limit(l, n, o, sigma), args=(l, n, o))[0]

        if not isclose(sigma_3[i], sigma_4[i]):
            CN_C[i] = dblquad(CN_func_to_integrate, sigma_3[i], sigma_4[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: phi_func(l, n, o, sigma)[1], args=(l, n, o))[0]
            CA_C[i] = dblquad(CA_func_to_integrate, sigma_3[i], sigma_4[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: phi_func(l, n, o, sigma)[1], args=(l, n, o))[0]
            CY_C[i] = dblquad(CY_func_to_integrate, sigma_3[i], sigma_4[i], lambda sigma: phi_func(l, n, o, sigma)[0], lambda sigma: phi_func(l, n, o, sigma)[1], args=(l, n, o))[0]
    CN_B[isnan(CN_B)] = 0
    CN_C[isnan(CN_C)] = 0
    CA_B[isnan(CA_B)] = 0
    CA_C[isnan(CA_C)] = 0
    CY_B[isnan(CY_B)] = 0
    CY_C[isnan(CY_C)] = 0

    S = (pi - phi_1) * R**2

    CN = -R**2 / S * (CN_A + CN_B + CN_C)
    CA = R**2 / S * (CA_A + CA_B + CA_C)
    CY = R**2 / S * (CY_A + CY_B + CY_C)

    CL, CD, CS = lift_drag_crosswind(CA, CY, CN, alpha, beta)
    Cl = Cm = Cn = zeros_like(alpha)

    if to_dict:
        return ret_correct_vals({
            "CN": CN,
            "CA": CA,
            "CY": CY,
            "CL": CL,
            "CD": CD,
            "CS": CS,
            "L/D": CL / CD,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn
        })
    return ret_correct_vals((CN, CA, CY, CL, CD, CS, CL / CD, Cl, Cm, Cn))
