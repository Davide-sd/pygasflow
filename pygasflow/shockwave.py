import numpy as np
import param
from scipy.optimize import bisect, minimize_scalar
from scipy.integrate import solve_ivp

from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.roots import apply_bisection
from pygasflow.generic import characteristic_mach_number
from pygasflow.utils.decorators import check_shockwave, check, deprecated
import warnings
from numbers import Number

# NOTE:
# In the following module:
#   beta: shock wave angle.
#   theta: flow deflection angle.

#####################################################################################
############# The following methods are specific for normal shock waves. ############
##### They can also be used for calculation with oblique shock wave, just use #######
##################### the normal component of the Mach number. ######################
#####################################################################################

@check_shockwave
def pressure_ratio(M1, gamma=1.4):
    """Compute the static pressure ratio P2/P1.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Pressure Ratio P2/P1
    """
    return (2 * gamma * M1**2 - gamma + 1) / (gamma + 1)


@check_shockwave
def temperature_ratio(M1, gamma=1.4):
    """ Compute the static temperature ratio T2/T1.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Temperature Ratio T2/T1
    """
    return ((2 * gamma * M1**2 - gamma + 1) * (2 + (gamma - 1) * M1**2)
            / ((gamma + 1)**2 * M1**2))


@check_shockwave
def density_ratio(M1, gamma=1.4):
    """ Compute the density ratio rho2/rho1.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Density Ratio rho2/rho1
    """
    return ((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2)


@check_shockwave
def total_pressure_ratio(M1, gamma=1.4):
    """ Compute the total pressure ratio P02/P01.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Total Pressure Ratio P02/P01
    """
    a = (gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2)
    b = (gamma + 1) / (2 * gamma * M1**2 - gamma + 1)
    return a**(gamma / (gamma - 1)) * b**(1 / (gamma - 1))


@check_shockwave
def total_temperature_ratio(M1, gamma=1.4):
    """ Compute the total temperature ratio T02/T01.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Total Temperature Ratio T02/T01 (spoiler: always equal to 1 :P )
    """
    return np.ones_like(M1)


@check_shockwave
def rayleigh_pitot_formula(M1, gamma=1.4):
    """Compute the ratio Pt2 / P1, between the stagnation pressure behind a
    normal shock wave and the static pressure ahead of the shock wave.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Ratio Pt2 / P1

    See Also
    --------
    m1_from_rayleigh_pitot_pressure_ratio

    References
    ----------

    "Equations, Tables and Charts for compressible flow", NACA R-1135, 1953
    """
    return ((gamma + 1) * M1**2 / 2)**(gamma / (gamma - 1)) * ((gamma + 1) / (2 * gamma * M1**2 - (gamma - 1)))**(1 / (gamma - 1))


def m1_from_rayleigh_pitot_pressure_ratio(ratio, gamma=1.4):
    """Compute the upstream Mach number of a normal shock wave starting from
    the pressure ratio Pt2 / P1, where Pt2 is the stagnation pressure behind a
    normal shock wave and the static pressure ahead of the shock wave.

    Parameters
    ----------
    pr : float or array_like
        Pressure ratio.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    M1 : float or ndarray
        Upstream Mach number.

    See Also
    --------
    rayleigh_pitot_formula
    """
    is_scalar = isinstance(ratio, Number)
    ratio = np.atleast_1d(ratio)
    ratio_lower_lim = rayleigh_pitot_formula(1, gamma)
    M1 = np.zeros_like(ratio, dtype=float)
    M1[ratio < ratio_lower_lim] = np.nan

    def func(Mu, gamma, pr):
        return rayleigh_pitot_formula(Mu, gamma) - pr

    for i, r in enumerate(ratio):
        if r >= ratio_lower_lim:
            M1[i] = bisect(func, a=1, b=100, args=(gamma, r))
    if is_scalar:
        return M1[0]
    return M1


@check_shockwave
def entropy_difference(M1, gamma=1.4):
    """ Compute the dimensionless entropy difference, (s2 - s1) / C_p.
    Eq (3.60) Anderson's.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Dimensionless Entropy Difference, (s2 - s1) / C_p
    """
    a = np.log((1 + 2 * gamma / (gamma + 1) * (M1**2 - 1)) * ((2 + (gamma - 1) * M1**2) / ((gamma + 1) * M1**2)))
    b = (gamma - 1) / gamma * np.log(1 + 2 * gamma / (gamma + 1) * (M1**2 - 1))
    return a - b


# TODO: here I can't use the decorator @check_shockwave to perform
# the argument checks, because it would check for M1 >= 1. But this
# function can also be used for M1 > 0. Therefore I do the checks
# into this function.
# Evaluate the possibility to pass a comparison value trough the decorator.
@check
def mach_downstream(M1, gamma=1.4):
    """ Compute the downstream Mach number M2.
    Note that this function can also be used to compute M1 given M2.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Because this function can be used to compute M1
        given M2, it will not perform a check wheter M1 >= 1.
        Be careful on your use!
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Downstream Mach Number M2
    """
    if np.any(M1 < 0):
        raise ValueError("The provided Mach number must be positive")
    return ((1 + (gamma - 1) / 2 * M1**2) / (gamma * M1**2 -
            (gamma - 1) / 2))**(0.5)


@check_shockwave
def m1_from_pressure_ratio(ratio, gamma=1.4):
    """ Compute M1 from the pressure ratio.

    Parameters
    ----------
    ratio : array_like
        Pressure Ratio P2/P1. If float, list, tuple is given as input,
        a conversion will be attempted. Must be P2/P1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Upstream Mach number M1.
    """
    if np.any(ratio < 1):
        raise ValueError("The pressure ratio must be P2/P1 >= 1")

    return np.sqrt((ratio * (gamma + 1) + gamma - 1) / (2 * gamma))


@check_shockwave
def m1_from_temperature_ratio(ratio, gamma=1.4):
    """ Compute M1 from the temperature ratio.

    Parameters
    ----------
    ratio : array_like
        Temperature Ratio T2/T1. If float, list, tuple is given as input,
        a conversion will be attempted. Must be T2/T1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Upstream Mach number M1.
    """
    if np.any(ratio < 1):
        raise ValueError("The temperature ratio must be T2/T1 >= 1")

    a = 2.0 * gamma * (gamma - 1)
    b = 4.0 * gamma - (gamma - 1)**2 - ratio * (gamma + 1)**2
    c = -2.0 * (gamma - 1)
    return np.sqrt((-b + np.sqrt(b**2 - 4.0 * a * c)) / 2.0 / a)


@check_shockwave
def m1_from_density_ratio(ratio, gamma=1.4):
    """ Compute M1 from the density ratio.

    Parameters
    ----------
    ratio : array_like
        Density Ratio rho2/rho1. If float, list, tuple is given as input,
        a conversion will be attempted.
        Must be 1 <= rho2/rho1 < (gamma + 1) / (gamma - 1).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Upstream Mach number M1.
    """
    gr = (gamma + 1) / (gamma - 1)
    if np.any(ratio < 1) or np.any(ratio > gr):
        raise ValueError("The density ratio must be 1 < rho2/rho1 < " + str(gr))

    return np.sqrt(2.0 * ratio / (gamma + 1 - ratio * (gamma - 1)))


@check_shockwave
def m1_from_total_pressure_ratio(ratio, gamma=1.4):
    """ Compute M1 from the total pressure ratio.

    Parameters
    ----------
    ratio : array_like
        Total Pressure Ratio. If float, list, tuple is given as input,
        a conversion will be attempted. Must be 0 <= P02/P01 <= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Upstream Mach number M1.
    """
    if np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("The total pressure ratio must be 0 <= P02/P01 <= 1")

    func = lambda M1, r: r - ((gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2))**(gamma / (gamma - 1)) * ((gamma + 1) / (2 * gamma * M1**2 - gamma + 1))**(1 / (gamma - 1))

    return apply_bisection(ratio, func, "sup")


@check_shockwave
def m1_from_m2(M2, gamma=1.4):
    """ Compute M1 from the downstream Mach number M2.

    Parameters
    ----------
    M2 : array_like
        Downstream Mach Number. If float, list, tuple is given as input,
        a conversion will be attempted.
        Must be ((gamma - 1) / 2 / gamma) < M_2 < 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Upstream Mach number M1.
    """
    lower_lim = (gamma - 1) / 2 / gamma
    if np.any(M2 < lower_lim) or np.any(M2 > 1):
        raise ValueError("The downstream M2 must be " + str(lower_lim) + " < M2 < 1")

    return mach_downstream(M2, gamma)


#######################################################################################
############## The following methods are specific for oblique shock waves #############
#######################################################################################


@check_shockwave
def theta_from_mach_beta(M1, beta, gamma=1.4):
    """ Compute the flow turning angle Theta accordingly to the input
    parameters.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    beta : float
        Shock wave angle in degrees.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    theta : ndarray
        Flow angle Theta [degrees]. If detachment is detected, np.nan will be
        returned and a warning message will be raised.
    """
    M1 = np.atleast_1d(M1)
    beta = np.deg2rad(beta)
    # it must be len(beta)=len(theta) for indexing
    if isinstance(beta, Number):
        beta *= np.ones_like(M1)
    elif (len(M1) > 1) and (len(beta) == 1):
        beta = beta[0] * np.ones_like(M1)

    num = M1**2 * np.sin(beta)**2 - 1
    # prevents numerical errors which will cause negative num values,
    # when M1 being too close to 1 and beta is too close to 90
    num[np.isclose(num, 0)] = 0
    den = M1**2 * (gamma + np.cos(2 * beta)) + 2
    # avoid `RuntimeWarning: divide by zero encountered in divide` when beta=0.
    theta = np.zeros_like(num)
    idx = np.isclose(beta, 0, atol=1e-10)
    theta[~idx] = np.arctan(2 / np.tan(beta[~idx]) * num[~idx] / den[~idx])
    theta[idx] = -np.inf
    # need to take into account that we are considering only positive
    # values of the Flow Angle Theta.
    theta[theta < 0] = np.nan
    if np.any(np.isnan(theta)):
        warnings.warn(
            "WARNING: detachment detected in at least one element of"
            " the flow turning angle theta array. Be careful!")

    return ret_correct_vals(np.rad2deg(theta))


@check_shockwave
def beta_from_mach_theta(M1, theta, gamma=1.4):
    """ Compute the shock angle Beta accordingly to the input parameters.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    theta : float
        Flow turning angle in degrees.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Dictionary of Shock angle beta [degrees] if it exists, else NaN:
        ``{"weak": beta_weak, "strong": beta_strong}``.
    """
    # Exact and Approximate Solutions to the Oblique Shock Equations for
    # Real-Time Applications, T.T. Hartley, R. Brandis, and F. Mossayebi, 1991

    theta = np.deg2rad(theta)

    # equations 3, 4, 5
    b = -((M1**2 + 2) / M1**2 + gamma * np.sin(theta)**2)
    c = (2 * M1**2 + 1) / M1**4 + ((gamma + 1)**2 / 4 + (gamma - 1) / M1**2) * np.sin(theta)**2
    d = -np.cos(theta)**2 / M1**4

    # equation 6
    Q = (3 * c - b**2) / 9
    R = (9 * b * c - 27 * d - 2 * b**3) / 54
    D = Q**3 + R**2

    # into this function, _Q,_R,_D,_b are scalar values
    def func(_Q, _R, _D, _b):
        # TODO:
        # here I try to deal with rounding errors. What value should I chose
        # for the threashold?
        if _D > 0 and _D < 1e-12:
            _D = 0

        # check for detached shock
        if _D > 0:
            return np.nan, np.nan

        # equation 10
        delta = 0
        if _R < 0:
            delta = np.pi

        # equation 9
        phi = (np.arctan(np.sqrt(-_D) / _R) + delta) / 3

        # equation 8
        Xs = -_b / 3 + 2 * np.sqrt(-_Q) * np.cos(phi)
        Xw = -_b / 3 - np.sqrt(-_Q) * (np.cos(phi) - np.sqrt(3) * np.sin(phi))

        # try to deal with numerical errors
        if Xw >= 1:
            beta_weak = np.pi / 2
        else:
            beta_weak = np.arctan(np.sqrt(Xw / (1 - Xw)))
        if Xs >= 1:
            beta_strong = np.pi / 2
        else:
            beta_strong = np.arctan(np.sqrt(Xs / (1 - Xs)))

        return beta_weak, beta_strong

    beta_weak, beta_strong = np.zeros_like(M1), np.zeros_like(M1)

    if M1.size == 1:    # scalar case
        if M1 == 1:
            beta_weak, beta_strong = np.pi / 2, np.pi / 2
        else:
            beta_weak, beta_strong = func(Q, R, D, b)
    else:
        for i, _d in np.ndenumerate(D):
            beta_weak[i], beta_strong[i] = func(Q[i], R[i], _d, b[i])
        # idx = M1 == 1
        # beta_weak[idx], beta_strong[idx] = 90, 90
    beta_weak = ret_correct_vals(np.rad2deg(beta_weak))
    beta_strong = ret_correct_vals(np.rad2deg(beta_strong))
    return { "weak": beta_weak, "strong": beta_strong }


@check_shockwave
def oblique_mach_downstream(M1, beta=None, theta=None, gamma=1.4, flag='weak'):
    """Compute the downstream Mach number M2 accordingly from the input
    parameters.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    beta : float, optional
        The shock wave angle in degrees. If beta=None you must give in theta.
    theta : float, optional
        The flow deflection angle in degrees. If theta=None you must give
        in beta.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    flag : string, optional
        Can be either ``'weak'`` or ``'strong'``. Default to ``'weak'``.
        Chose what value to compute if theta is provided.

    Returns
    -------
    out : ndarray
        Downstream Mach Number M2
    """
    # TODO: with the current check_shockwave decorator, flag can only be 'weak'
    # or 'strong'. If 'both' an error will be raised.
    if (beta is None) and (theta is None):
        raise ValueError(
            "To compute the downstream Mach number you have to provide"
            " either theta or beta.")
    flag = flag.lower()
    if flag not in ["weak", "strong"]:
        raise ValueError("Flag must be either 'weak' or 'strong' or 'both'.")

    if beta is not None:
        beta = np.deg2rad(beta)
        pr = pressure_ratio(M1 * np.sin(beta), gamma=gamma)
        tpr = total_pressure_ratio(M1 * np.sin(beta), gamma=gamma)
    elif theta is not None:
        beta = beta_from_mach_theta(M1, theta, gamma=gamma)
        beta = np.deg2rad(beta[flag])
        pr = pressure_ratio(M1 * np.sin(beta), gamma=gamma)
        tpr = total_pressure_ratio(M1 * np.sin(beta), gamma=gamma)

    # Solve Flack's equation (C.8.27) to isolate downstream Mach number M2,
    # provided we first solve for both the Pressure ratio (pr) and the Total
    # Pressure ratio (tpr).
    # Flack RD., "Fundamentals of Jet Propulsion with Power Generation
    # Applications," Cambridge University Press, 2023, 978-1-316-51736-9.
    # www.cambridge.org/highereducation/isbn/9781316517369
    rat = (gamma - 1) / 2.0
    ex = (gamma - 1) / gamma
    M2 = np.sqrt( ( ((tpr/pr) ** ex) * (1 + rat*M1*M1) - 1 ) / rat )
    return M2


@check_shockwave([0, 1])
def beta_from_upstream_mach(M1, MN1):
    """ Compute the shock wave angle beta from the upstream Mach number and
    its normal component.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    MN1 : array_like
        Normal Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be ``MN1.shape == M1.shape``.

    Returns
    -------
    out : ndarray
        Shock angle Beta [degrees]
    """
    MN1 = np.asarray(MN1)
    if np.any(M1 < 1):
        raise ValueError("The upstream Mach number must be > 1.")
    if M1.shape != MN1.shape:
        raise ValueError("M1 and MN1 must have the same number of elements and the same shape.")
    if np.any(M1 - MN1 < 0):
        raise ValueError("Upstream Mach number must be >= of the normal upstream Mach number.")
    return np.rad2deg(np.arcsin(MN1 / M1))


@check_shockwave
def normal_mach_upstream(M1, beta=None, theta=None, gamma=1.4, flag="weak"):
    """ Compute the upstream normal Mach Number, which can then be used
    to evaluate all other ratios.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    beta : float, optional
        The shock wave angle in degrees. If beta=None you must give in theta.
    theta : float, optional
        The flow deflection angle in degrees. If theta=None you must give
        in beta.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    flag : string, optional
        Possible options: ``'weak', 'strong', 'both'``. Default to ``'weak'``.
        Chose what value to compute if theta is provided.

    Returns
    -------
    out : ndarray
        Normal Mach number upstream of the shock wave.
        If theta is given, and ``flag="both"`` it returns a dictionary of
        Normal Mach numbers: ``{"weak":weak_MN1, "strong":strong_MN1}``.
    """
    # TODO: with the current check_shockwave decorator, flag can only be 'weak'
    # or 'strong'. If 'both' an error will be raised.
    if (beta is None) and (theta is None):
        raise ValueError(
            "To compute the normal component of the upstream Mach number,"
            " you have to provide either theta or beta.")
    flag = flag.lower()
    if flag not in ["weak", "strong", "both"]:
        raise ValueError("Flag must be either 'weak' or 'strong' or 'both'.")

    MN1 = -1
    if beta is not None:
        beta = np.deg2rad(beta)
        MN1 = M1 * np.sin(beta)
    elif theta is not None:
        # check for detachment (when theta > theta_max(M1))
        theta_max = max_theta_from_mach(M1, gamma)
        if np.any(theta > theta_max):
            raise ValueError("Detachment detected: can't solve the flow when theta > theta_max.\n" +
            "M1 = {}\n".format(M1) +
            "theta_max(M1) = {}\n".format(theta_max) +
            "theta = {}\n".format(theta))
        beta = beta_from_mach_theta(M1, theta, gamma=gamma)
        MN1 = dict()
        for k,v in beta.items():
            beta[k] = np.deg2rad(v)
            MN1[k] = M1 * np.sin(beta[k])
        if flag != "both":
            MN1 = MN1[flag]

    return MN1


@check_shockwave([1])
def get_upstream_normal_mach_from_ratio(ratioName, ratio, gamma=1.4):
    """
    Compute the upstream Mach number given a ratio as an argument.

    Parameters
    ----------
    ratioName : string
        Name of the ratio given in input. Can be either one of:

        * ``'pressure'``: P2/P1
        * ``'temperature'``: T2/T1
        * ``'density'``: rho2/rho1
        * ``'total_pressure'``: P02/P01
        * ``'mnd'``: Normal Mach downstream of the shock wave

    ratio : array_like
        Actual value of the ratio. If float, list, tuple is given as input,
        a conversion will be attempted.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        The upstream Mach number.
    """
    ratioName = ratioName.lower()
    if ratioName == "mn2":
        warnings.warn(
            "Key 'mn2' is deprecated and will be removed in the future."
            " Use 'mnd' instead.",
            stacklevel=1
        )
        ratioName="mnd"
    if isinstance(ratio, (list, tuple)):
        ratio = np.asarray(ratio)

    ratios = {
        "pressure": m1_from_pressure_ratio,
        "temperature": m1_from_temperature_ratio,
        "density": m1_from_density_ratio,
        "total_pressure": m1_from_total_pressure_ratio,
        "mnd": m1_from_m2.__no_check__,
    }

    if ratioName not in ratios.keys():
        raise ValueError("Unrecognized ratio '{}'".format(ratioName))

    # TODO: should I implement the necessary checks and then call __no_check?
    return ratios[ratioName](ratio, gamma)


@check_shockwave
def get_ratios_from_normal_mach_upstream(Mn, gamma=1.4):
    """
    Compute the ratios of the quantities across a Shock Wave given the Normal Mach number.

    Parameters
    ----------
    Mn : array_like
        Normal Mach number upstream of the shock wave. If float, list, tuple
        is given as input, a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    pr : array_like
        Pressure ratio across the shock wave.
    dr : array_like
        Density ratio across the shock wave.
    tr : array_like
        Temperature ratio across the shock wave.
    tpr : array_like
        Total Pressure ratio across the shock wave.
    mn2 : array_like
        Normal Mach number dowstream of the shock wave.
    """
    pr = pressure_ratio.__no_check__(Mn, gamma)
    dr = density_ratio.__no_check__(Mn, gamma)
    tr = temperature_ratio.__no_check__(Mn, gamma)
    tpr = total_pressure_ratio.__no_check__(Mn, gamma)
    mn2 = mach_downstream.__no_check__(Mn, gamma)

    return pr, dr, tr, tpr, mn2


@check_shockwave
def maximum_mach_from_deflection_angle(theta, gamma=1.4):
    """
    Compute the maximum Mach number from a given Deflection angle theta.

    Parameters
    ----------
    theta : float
        Deflection angle in degrees. Must be 0 <= theta <= 90.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Max_M : float
        The maximum Mach number for the specified theta.
    """
    upper_lim = max_theta_from_mach(np.finfo(np.float32).max, gamma)
    if theta >= upper_lim:
        raise ValueError("The flow deflection angle must be < {}, which correspond to Mach=infinity.".format(upper_lim))

    def function(t):
        def func(M):
            theta_max = max_theta_from_mach(M, gamma)
            return theta_max - t

        # TODO:
        # 1. this is a slow procedure, can it be done faster, differently?
        a = 1
        b = 1e06
        return bisect(func, a, b)

    Max_M = np.zeros_like(theta)
    for i, t in enumerate(theta):
        Max_M[i] = function(t)
    return Max_M


@check_shockwave
def mimimum_beta_from_mach(M1):
    """
    Compute the minimum shock wave angle for a given upstream Mach number.

    Parameters
    ----------
    M : array_like
        Upstream Mach number. Must be >= 1.

    Returns
    -------
    beta: float
        Shock wave angle in degrees
    """
    return np.rad2deg(np.arcsin(1 / M1))


@check_shockwave
def max_theta_from_mach(M1, gamma=1.4):
    """
    Compute the maximum deflection angle for a given upstream Mach number.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    theta_max : ndarray
        Maximum deflection angle theta in degrees
    """
    # http://www.pdas.com/maxwedge.xml

    # Rewrite Anderson's eq. 4.17 to have cotan(theta) = tan(beta)/2 * ....
    # We minimize the right hand side, which is a function of Beta. This means we find
    # the minimum of cotan(theta), which corresponds to the maximum deflection angle theta.

    def function(M):
        # Right hand side function of beta
        def func(beta):
            num = (M**2 * (gamma + 1))
            den = (2 * (M**2 * np.sin(beta)**2 - 1))
            if np.isclose(den, 0):
                # avoid raising warnings about division by 0
                return np.inf
            return np.tan(beta) * (num / den - 1)

        # bound a correspond to the discontinuity of func
        a = np.arcsin(1 / M)
        b = np.pi / 2

        # result of the minimization. res.x correspond to the value of Beta where function is
        # minimized
        res = minimize_scalar(func, bounds=(a,b), method='bounded')
        # cotan(theta_max) = func(beta_min)
        # Therefore theta_max = arctan(1 / func(beta_min))
        return np.rad2deg(np.arctan(1 / func(res.x)))

    theta_max = np.zeros_like(M1)
    for i, m in enumerate(M1):
        theta_max[i] = function(m)
    return theta_max


@check_shockwave
def detachment_point_oblique_shock(M1, gamma=1.4):
    """
    Compute the detachment point, ie the shock wave angle beta corresponding
    to the maximum deflection angle theta given an upstream Mach number.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    beta : array_like
        The shock angle in degrees.
    theta_max : array_like
        The maximum deflection angle in degrees associated to the provided
        upstream Mach number.

    See Also
    --------
    sonic_point_oblique_shock
    """

    theta_max = np.atleast_1d(
        max_theta_from_mach.__no_check__(M1, gamma))

    beta = np.zeros_like(M1)
    for i, (m, t) in enumerate(zip(M1, theta_max)):
        # here I chose 'weak', but in this case it's the same as 'strong'!
        beta[i] = beta_from_mach_theta(m, t, gamma=gamma)["weak"]
    return beta, theta_max


@deprecated("Use ``detachment_point_oblique_shock`` instead.")
@check_shockwave
def beta_from_mach_max_theta(M1, gamma=1.4):
    """
    Compute the shock wave angle beta corresponding to the maximum deflection
    angle theta given an upstream Mach number.

    Parameters
    ----------
    M1 : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Beta : array_like
        The shock angle in degrees.
    """
    beta, _ = detachment_point_oblique_shock(M1, gamma)
    return beta


@deprecated("Use ``sonic_point_oblique_shock`` instead.")
@check_shockwave
def beta_theta_max_for_unit_mach_downstream(M1, gamma=1.4):
    """
    Compute the shock maximum deflection angle, theta_max, as well as the
    wave angle beta corresponding to the unitary downstream Mach
    number, M2 = 1. This function is useful to compute the location of the
    sonic line.

    This function relies on root-finding algorithms. If a root can't be
    found, np.nan will be used as the result.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Beta : array_like
        The shock angle in degrees corresponding to M2 = 1.
    Theta max : array_like
        The maximum deflection angle in degrees corresponding to M2 = 1.
    """
    theta_max = np.atleast_1d(max_theta_from_mach.__no_check__(M1, gamma))
    beta, _ = sonic_point_oblique_shock(M1, gamma)
    return beta, theta_max


@check_shockwave
def sonic_point_oblique_shock(M1, gamma=1.4):
    """
    Compute the wave angle, beta, and the shock deflection angle, theta,
    corresponding to the unitary downstream Mach number, M2 = 1.

    This function is useful to compute the location of the sonic line.
    It relies on root-finding algorithms: if a root can't be found, np.nan
    will be used as the result.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    beta : array_like
        The shock angle in degrees corresponding to M2 = 1.
    theta : array_like
        The deflection angle in degrees corresponding to M2 = 1.

    See Also
    --------
    detachment_point_oblique_shock
    """

    # TODO: this approach is wrong. why?
    # # Start with the equation to compute M2 from M1 (normal shock case).
    # # In oblique shock, this relation is still valid for normal machs.
    # # Remember that Mn1 = M1 sin(beta) and Mn2 = M2 sin(beta - theta).
    # # Sustitute them into the aformentioned equation, and remember that
    # # M2 = 1. This is what you are going to get.
    # # NOTE: in the practical range of interest (1 < gamma < 2) and for M1 > 1,
    # # this function has from 2 to 4 zeros (inside the range 0 <= beta <= 90).
    # # We are interested in the closest zero to 90 deg, which is progressevely
    # # harder to compute as gamma -> 0 and M1 -> inf.
    # def func(b, M, t, g):
    #     return ((1 + (g - 1) / 2 * (M * np.sin(b))**2) / (g * (M * np.sin(b))**2 - (g - 1) / 2)) - np.sin(b - t)**2

    def func(beta, M1, gamma):
        MN1 = M1 * np.sin(np.deg2rad(beta))
        MN2 = mach_downstream.__no_check__(MN1, gamma)
        # delta is the flow deflection angle (Anderson's Figure 10.4)
        delta = theta_from_mach_beta.__no_check__(M1, beta, gamma)
        M2 = MN2 / np.sin(np.deg2rad(beta - delta))
        return M2 - 1

    beta_sonic = np.zeros_like(M1)
    theta_sonic = np.zeros_like(M1)
    for i, m in enumerate(M1):
        if np.isclose(m, 1):
            beta_sonic[i] = 90
            theta_sonic[i] = 0
            continue
        try:
            a = mimimum_beta_from_mach.__no_check__(m)
            # TODO: I should be able to run a bisection from this upper limit.
            # However, for high Mach numbers the bisection fails with
            # `ValueError f(a) and f(b) must have different signs`. Why?
            # b = beta_from_mach_max_theta.__no_check__([m], gamma)
            b = 90
            beta_sonic[i] = bisect(func, a=a, b=b, args=(m, gamma))
            # TODO: figure it out a better way to skip check and to deal with
            # scalar input to functions that requires arrays
            theta_sonic[i] = theta_from_mach_beta(m, beta_sonic[i], gamma)
        except ValueError:
            beta_sonic[i] = np.nan
            theta_sonic[i] = np.nan

    return beta_sonic, theta_sonic


@check_shockwave([0, 1])
def mach_from_theta_beta(theta, beta, gamma=1.4):
    """
    Compute the upstream Mach number given the flow deflection angle and the
    shock wave angle.

    Parameters
    ----------
    theta : array_like
        Flow deflection angle in degrees. If float, list, tuple is given
        as input, a conversion will be attempted. Must be 0 <= theta <= 90.
    beta : array_like
        Shock wave angle in degrees. If float, list, tuple is given as input,
        a conversion will be attempted. Must be 0 <= beta <= 90.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Mach : ndarray
        The upstream Mach number.
    """
    if beta.shape != theta.shape:
        raise ValueError(
            "Flow deflection angle and Shock wave angle must have"
            " the same shape.")

    # make sure (beta, theta) is in the allowable space: consider for example
    # M1 = inf and the Oblique Shock Diagram. Then, there exis a theta_max
    # corresponding to the provided beta. If theta > theta_max, there is no
    # solution.
    M1_inf = 1000000000 * np.ones_like(beta)
    theta_max = theta_from_mach_beta(M1_inf, beta, gamma)
    if np.any(theta > theta_max):
        raise ValueError("There is no solution for the current choice of" +
        " parameters. Please check the Oblique Shock diagram with the following"
        " parameters:\n" +
        "beta = {}\n".format(beta) +
        "theta = {}\n".format(theta))
    # case beta == 90 and theta == 0, from which M = 1
    idx0 = np.bitwise_and(beta == 90, theta == 0)
    # if beta == 0 and theta == 0, mach goes to infinity. But num and den both
    # go to infinity resulting in NaN. Need to catch it.
    idx1 = np.bitwise_and(beta == 0, theta == 0)

    # all other cases can be resolved
    idx = np.invert(np.bitwise_or(idx0, idx1))

    beta = np.deg2rad(beta)
    theta = np.deg2rad(theta)

    num = np.ones_like(beta, dtype=float)
    den = np.ones_like(beta, dtype=float)

    num[idx] = 2 * (1 / np.tan(theta[idx]) + np.tan(beta[idx]))
    den[idx] = np.sin(beta[idx])**2 * num[idx] - np.tan(beta[idx]) * (gamma + 1)

    mach = np.zeros_like(beta, float)
    mach[den > 0] = np.sqrt(num[den > 0] / den[den > 0])
    mach[den <= 0] = np.nan

    mach[idx0] = 1
    mach[idx1] = np.inf
    return mach


def mach_beta_from_theta_ratio(theta, ratio_name, ratio_value, gamma=1.4):
    """Compute the upstream Mach numbers and the shockwave angles starting
    from a ratio across the oblique shock wave the the flow deflection angle.
    Usually, there are two solutions.

    Parameters
    ----------
    theta : float
    ratio_name : str
        Can be either one of: 'pressure', 'temperature', 'density',
        'total_pressure'.
    ratio_value : float
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Mu : list
        List of upstream Mach numbers.
    beta : list
        List of shockwave angles associated to the upstream Mach numbers.
    """
    ratio_name = ratio_name.lower()
    allowed_ratios = ['pressure', 'temperature', 'density', 'total_pressure']
    if ratio_name not in allowed_ratios:
        raise ValueError(
            f"`ratio_name` must be one of the following: {allowed_ratios}."
            f" Instead, '{ratio_name}' was received.")

    def func(M1, theta, gamma, Mn_target, region):
        beta = beta_from_mach_theta(M1, theta, gamma)[region]
        Mn1 = M1 * np.sin(np.deg2rad(beta))
        return Mn1 - Mn_target

    # TODO: is there a more efficient way to do this? Here is the current
    # procedure:
    # 1. Compute normal upstream Mach number
    # 2. Given theta, find the minimum Mach number that can support the flow.
    #    This require a bisection procedure.
    # 3. Run bisection over func for strong region to find a Mach number that
    #    gives the computed normal Mach number.
    # 4. Run bisection over func for weak region to find a Mach number that
    #    gives the computed normal Mach number.
    # 5. Compute the shock wave angles with these Mach roots.

    # find suitable range to run bisection for step 3 and 4.
    b = 100
    def find_min_mach_from_theta(M):
        theta_max = max_theta_from_mach(M, gamma)
        return theta_max - theta
    a = bisect(
        find_min_mach_from_theta, a=1, b=max_theta_from_mach(1e06, gamma))

    Mn1 = get_upstream_normal_mach_from_ratio(ratio_name, ratio_value, gamma)

    # NOTE: to understand the following logic, plot func with
    # theta=20, Mn_target=1.65, over a range of Mach numbers, for both regions
    # (weak and strong). Look for the roots.

    if func(a, theta, gamma, Mn1, "strong") < 0:
        # if there is a strong solution, then there is also at most one
        # weak solution.
        M_r1 = bisect(func, a=a, b=b, args=(theta, gamma, Mn1, "strong"))
        M_r2 = bisect(func, a=a, b=b, args=(theta, gamma, Mn1, "weak"))
        beta1 = beta_from_mach_theta(M_r1, theta, gamma)["strong"]
        beta2 = beta_from_mach_theta(M_r2, theta, gamma)["weak"]
    else:
        # there should be two weak solutions
        # TODO: if func(a, theta, gamma, Mn1, "strong") == 0 there are two
        # identical solutions. Is this procedure up to the task?

        try:
            # Find the minimum of the weak curve
            _min = minimize_scalar(
                func, bounds=(a, b), method="bounded",
                args=(theta, gamma, Mn1, "weak"))
            _min = _min.x
            M_r1 = bisect(func, a=a, b=_min, args=(theta, gamma, Mn1, "weak"))
            M_r2 = bisect(func, a=_min, b=b, args=(theta, gamma, Mn1, "weak"))
            beta1 = beta_from_mach_theta(M_r1, theta, gamma)["weak"]
            beta2 = beta_from_mach_theta(M_r2, theta, gamma)["weak"]
        except ValueError:
            # assume there is no solution
            raise ValueError(
                "There is no solution for the current choice of parameters."
                " Please check the ObliqueShockDiagram, plotting the"
                f" {ratio_name} ratio, with the following parameters:\n"
                f"{ratio_name} ratio = {ratio_value}\n"
                f"theta = {theta}\n"
            )
    return [M_r1, M_r2], [beta1, beta2]


def shock_polar_equation(Vx_as_ratio, M1s, gamma=1.4):
    """Analytical equation for the shock polar.

    Parameters
    ----------
    Vx_as_ratio : float or array_like
        This is Vx/a*
    M1s : float
        Characteristic upstream Mach number.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Vy/a* : float
    """
    is_scalar = isinstance(Vx_as_ratio, Number)
    Vx_as_ratio = np.atleast_1d(Vx_as_ratio)
    # equation 4.22 (Anderson)
    num = (M1s - Vx_as_ratio)**2 * (Vx_as_ratio * M1s - 1)
    # account for numerical errors leading (num) to be proportional to a very small
    # negative value
    num[num < 0] = 0
    den = (2 / (gamma + 1)) * M1s**2 - Vx_as_ratio * M1s + 1
    res = np.sqrt(num / den)
    return res[0] if is_scalar else res


@check_shockwave
def shock_polar(M1, gamma=1.4, N=100, include_mirror=True):
    """
    Compute the ratios (Vx/a*), (Vy/a*) for plotting a Shock Polar.

    Parameters
    ----------
    M1 : float
        Upstream Mach number of the shock wave. Must be > 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    N : int, optional
        Number of discretization steps in the range [0, pi]. Must be > 1.
    include_mirror : bool
        If True, return results for polar angle in [0, 2*pi]. Otherwise,
        return results for polar angle in [0, pi].

    Returns
    -------
    (Vx/a*) : ndarray [1 x N]
        x-coordinate for the shock polar plot
    (Vy/a*) : ndarray [1 x N]
        y-coordinate for the shock polar plot
    """
    if (not isinstance(N, int)) or (N <= 1):
        raise ValueError("The number of discretization steps must be integer and > 1.")

    M1s = characteristic_mach_number(M1, gamma)
    # downstream Mach number to a normal shock wave
    M2 = mach_downstream(M1, gamma)
    M2s = characteristic_mach_number(M2, gamma)

    # polar coordinate
    alpha = np.linspace(0, np.pi, N)
    r = (M1s - M2s) / 2
    Vx_as = M2s + r + r * np.cos(alpha)
    Vy_as = shock_polar_equation(Vx_as, M1s, gamma)

    if include_mirror:
        Vx_as = np.append(Vx_as, Vx_as[::-1])
        Vy_as = np.append(Vy_as, -Vy_as[::-1])

    return Vx_as, Vy_as


def _check_pressure_deflection(N):
    if (not isinstance(N, int)) or (N <= 1):
        raise ValueError("The number of discretization steps must be integer and > 1.")


@check_shockwave
def pressure_deflection(M1, gamma=1.4, N=100):
    """
    Helper function to build Pressure-Deflection plots. Computes the locus
    of all possible static pressure values behind oblique shock wave for
    given upstream conditions.

    Parameters
    ----------
    M1 : float
        Upstream Mach number. Must be > 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    N : int, optional
        Number of points discretizing the range [0, theta_max].
        This function compute N points in the range [0, theta_max] for
        the `'weak'` solution, then compute N points in the range
        [theta_max, 0] for the `'strong'` solution.
        If ``include_mirror=False``, the length of the returned arrays is
        2*N, otherwise is 4*N.

    Returns
    -------
    theta : array_like
        Deflection angles
    pr : array_like
        Pressure ratios computed for the given Mach and the above
        deflection angles.

    """
    locus = PressureDeflectionLocus(M=M1, gamma=gamma)
    return locus.pressure_deflection(N=N, include_mirror=False)


class _BasePDLocus:
    """I need this base class in order to initialize `upstream_locus`
    in the subclass, which is a param.ClassSelector.
    """
    pass


class PressureDeflectionLocus(param.Parameterized, _BasePDLocus):
    """Represent the locus of all possible static pressure values behind
    oblique shock wave for given upstream conditions.

    This class implements the logic to deal with pressure-deflection locus
    from a numerical standpoint. It is meant to be used together with
    :class:`~pygasflow.interactive.diagrams.pressure_deflection.PressureDeflectionDiagram`
    in order to draw pressure-deflection locuses.

    """

    label = param.String(doc="""
        Name to be shown on the legend of the plot.""")
    M = param.Number(1, bounds=(1, None), doc="""
        Upstream Mach number for which the pressure-deflection locus
        should be computed.""")
    gamma = param.Number(1.4, bounds=(1, None), softbounds=(1, 2),
        inclusive_bounds=(False, False), doc="""
        Specific heats ratio.""")
    theta_origin = param.Number(0, doc="""
        Deflection angle at which `M` occurs.""")
    pr_to_fs_at_origin = param.Number(1, bounds=(1, None), doc="""
        Pressure ratio multiplier between the pressure at the current locus
        origin and the free-stream pressure.""")
    tr_to_fs_at_origin = param.Number(1, bounds=(1, None), doc="""
        Temperature ratio multiplier between the temperature at the current
        locus origin and the free-stream temperature.""")
    dr_to_fs_at_origin = param.Number(1, bounds=(1, None), doc="""
        Density ratio multiplier between the density at the current locus
        origin and the free-stream density.""")
    tpr_to_fs_at_origin = param.Number(1, bounds=(0, 1), doc="""
        Total pressure ratio multiplier between the total pressure at the
        current locus origin and the free-stream total pressure.""")
    theta_max = param.Number(constant=True, doc="""
        The maximum deflection angle possible with this pressure-deflection
        locus. Note that `theta_max` is considered relative to `theta_origin`
        along the left-running branch.""")
    _shockwave_at_theta = param.Parameter(constant=True, doc="""
        A numerical function used to evaluate the quantities across a
        shockwave located at the specified theta. This function will be wrapped
        by ``shockwave_at_theta`` in order to add appropriate docstring.""")
    upstream_locus = param.ClassSelector(
        class_=_BasePDLocus, constant=True, doc="""
        The locus associated with the upstream condition.""")
    sonic_point = param.Tuple(constant=True, doc="""
        Coordinates (theta, pr_to_fs) where the Mach downstream of the shock
        wave is sonic. `theta` (in degrees) is considered relative to
        ``theta_origin``.""")
    detachment_point = param.Tuple(constant=True, doc="""
        Coordinates (theta_max, pr_to_fs) indicating the maximum deflection
        by which a given supersonic flow can be deflected by an oblique
        shock wave. `theta` (in degrees) is considered relative to
        ``theta_origin``.""")

    def new_locus_from_shockwave(self, theta, label=""):
        """Create a new PressureDeflectionLocus object using the upstream
        conditions of this instance.

        Parameters
        ----------
        theta : float
            Deflection angle [degrees] of the oblique shockwave.
        label : str
            Label to be assigned to the new instance, which will later be shown
            on plots.

        Returns
        -------
        obj : PressureDeflectionLocus
        """
        # this import is here in order to avoid circular imports
        from pygasflow.solvers import oblique_shockwave_solver

        res = oblique_shockwave_solver(
            "mu", self.M, "theta", abs(theta), gamma=self.gamma, to_dict=True)
        return type(self)(
            M=res["md"], gamma=self.gamma, label=label,
            theta_origin=self.theta_origin + theta,
            pr_to_fs_at_origin=res["pr"] * self.pr_to_fs_at_origin,
            tr_to_fs_at_origin=res["tr"] * self.tr_to_fs_at_origin,
            dr_to_fs_at_origin=res["dr"] * self.dr_to_fs_at_origin,
            tpr_to_fs_at_origin=res["tpr"] * self.tpr_to_fs_at_origin,
            upstream_locus=self)

    def __str__(self):
        s = f"Pressure-Deflection Locus of {self.label}\n"
        s += f"M\t\t{self.M}\n"
        s += f"p/p_inf\t\t{self.pr_to_fs_at_origin}\n"
        s += f"T/T_inf\t\t{self.tr_to_fs_at_origin}\n"
        s += f"rho/rho_inf\t{self.dr_to_fs_at_origin}\n"
        s += f"P0/P0_inf\t{self.tpr_to_fs_at_origin}\n"
        s += f"th_orig\t\t{self.theta_origin} [deg]\n"
        return s

    @param.depends(
        "M", "gamma", "theta_origin", "pr_to_fs_at_origin",
        watch=True, on_init=True
    )
    def update_func(self):
        beta_detachment, theta_max = detachment_point_oblique_shock(
            self.M, self.gamma)

        def func(theta, region="weak"):
            actual_theta = abs(self.theta_origin - theta)
            beta = beta_from_mach_theta(
                self.M, actual_theta, self.gamma)[region]
            Mn_up = self.M * np.sin(np.deg2rad(beta))
            # avoid rounding errors that would make Mn < 1
            if np.isclose(Mn_up, 1):
                Mn_up = 1

            # compute the ratios across the shockwave
            pr, dr, tr, tpr, Mn_down = get_ratios_from_normal_mach_upstream(
                Mn_up, self.gamma)
            pr *= self.pr_to_fs_at_origin
            tr *= self.tr_to_fs_at_origin
            dr *= self.dr_to_fs_at_origin
            tpr *= self.tpr_to_fs_at_origin
            M_down = Mn_down /  np.sin(np.deg2rad(beta - actual_theta))

            sign = 1 if theta >= self.theta_origin else -1
            theta_corrected = sign * actual_theta

            keys = [
                "mu", "mnu", "md", "mnd", "beta", "theta",
                "pr", "tr", "dr", "tpr"
            ]
            values = [
                self.M, Mn_up, M_down, Mn_down, beta, theta_corrected,
                pr, tr, dr, tpr
            ]
            res = {k: v for k, v in zip(keys, values)}
            return res

        # sonic point
        beta_sonic, theta_sonic = sonic_point_oblique_shock(self.M, self.gamma)
        Mn_sonic = self.M * np.sin(np.deg2rad(beta_sonic))
        pr_sonic = pressure_ratio(Mn_sonic, self.gamma) * self.pr_to_fs_at_origin
        # detachment point
        Mn_detachment = self.M * np.sin(np.deg2rad(beta_detachment))
        pr_detachment = pressure_ratio(
            Mn_detachment, self.gamma) * self.pr_to_fs_at_origin

        with param.edit_constant(self):
            self.theta_max = theta_max
            self._shockwave_at_theta = func
            self.sonic_point = (theta_sonic + self.theta_origin, pr_sonic)
            self.detachment_point = (theta_max + self.theta_origin, pr_detachment)

    def shockwave_at_theta(self, theta, region="weak"):
        """Evaluates the current pression-deflection locus at a specified
        flow-deflection angle in the specified region for the current locus.

        Parameters
        ----------
        theta : float
            The flow deflection angle in degrees. This quantity is not relative
            to the origin of the current locus. Instead, it must be an angle
            taken from a pressure-deflection diagram.
        region : str
            Possible values are ``"weak", "strong"``.

        Returns
        -------
        res : dict
            Contains the the ratios and mach numbers across the shockwave.
            The following keys will be used:

            * ``"mu"``: upstream Mach number.
            * ``"mnu"``: upstream normal Mach number.
            * ``"md"``: downstream Mach number.
            * ``"mnd"``: downstream normal Mach number.
            * ``"beta"``: shock wave angle [degrees] relative to the upstream
              locus.
            * ``"theta"``: flow deflection angle [degrees] relative to the
              locus' origin. Note that this value can be negative.
            * ``"pr"``: pressure ratio to freestream.
            * ``"tr"``: temperature ratio to freestream.
            * ``"dr"``: density ratio to freestream.
            * ``"tpr"``: total pressure ratio to freestream.

        Examples
        --------

        This is a simple regular reflection from a solid boundary
        (refer to figure 4.18 of "Modern Compressible Flow, Anderson"):

        >>> gamma = 1.4
        >>> M1 = 2.8
        >>> theta1 = 16  # deg
        >>> T1 = 519     # °R
        >>> p1 = 1       # atm
        >>> l1 = PressureDeflectionLocus(M=M1, gamma=gamma, label="1")
        >>> l2 = l1.new_locus_from_shockwave(theta1, label="2")

        First shock wave:

        >>> shock_1 = l1.shockwave_at_theta(theta1)
        >>> shock_1["theta"]
        16
        >>> shock_1["beta"]
        np.float64(34.9226304011263)

        Reflected shock wave:

        >>> shock_2 = l2.shockwave_at_theta(0)
        >>> shock_2["theta"]
        -16
        >>> shock_2["beta"]
        np.float64(45.33424941323747)

        """
        return self._shockwave_at_theta(theta, region)

    def flow_quantities_at_locus_origin(
        self, p_fs=None, T_fs=None, rho_fs=None
    ):
        """Compute the flow quantities at a locus origin, using isentropic
        relations, which represent a state of the flow. Note that the locus
        origin corresponds to a situation in which the flow state behind the
        oblique shock wave is identical to the flow state ahead of it.

        Parameters
        ----------
        p_fs : float or None
            Freestream pressure.
        T_fs : float or None
            Freestream temperature.
        rho_fs : float or None
            Freestream density.

        Returns
        -------
        res : dict
            Contains the quantities just downstream of the specified shock
            wave. The following keys will be used:

            * ``"M"``: Mach number downstream of the shock wave.
            * ``"T"``: Temperature downstream of the shock wave.
            * ``"p"``: Pressure downstream of the shock wave.
            * ``"rho"``: Density downstream of the shock wave.
            * ``"T0"``: Total temperature downstream of the shock wave.
            * ``"p0"``: Total pressure downstream of the shock wave.
            * ``"rho0"``: Total density downstream of the shock wave.

        Examples
        --------

        Consider a simple regular reflection from a solid boundary:

        >>> from pygasflow.shockwave import PressureDeflectionLocus
        >>> gamma = 1.4
        >>> M1 = 2.8
        >>> theta1 = 16  # deg
        >>> T1 = 519     # °R
        >>> p1 = 1       # atm
        >>> l1 = PressureDeflectionLocus(M=M1, gamma=gamma, label="1")
        >>> l2 = l1.new_locus_from_shockwave(theta1, label="2")

        To compute the missing flow quantities in the free stream region:

        >>> region1 = l1.flow_quantities_at_locus_origin(p1, T1, None)
        >>> region1["M"]
        2.8
        >>> region1["T"]
        519
        >>> region1["p"]
        1
        >>> region1["p0"]
        np.float64(27.13829555269978)
        >>> region1["T0"]
        np.float64(1332.7919999999997)

        To compute the missing flow quantities after the first shock wave:

        >>> region2 = l2.flow_quantities_at_locus_origin(p1, T1, None)
        >>> region2["M"]
        np.float64(2.0585267649107384)
        >>> region2["T"]
        np.float64(721.4004347373847)
        >>> region2["p"]
        np.float64(2.830893893824571)
        >>> region2["p0"]
        np.float64(24.26467588950667)
        >>> region2["T0"]
        np.float64(1332.7919999999997)
        """
        return self.flow_quantities_after_shockwave(
            self.theta_origin, p_fs, T_fs, rho_fs)

    def flow_quantities_after_shockwave(
        self, theta, p_fs=None, T_fs=None, rho_fs=None, region="weak"
    ):
        """Compute the flow quantities after a shock wave.

        Parameters
        ----------
        theta : float
            The flow deflection angle in degrees. This quantity is not relative
            to the origin of the current locus. Instead, it must be an angle
            taken from a pressure-deflection diagram.
        p_fs : float or None
            Freestream pressure.
        T_fs : float or None
            Freestream temperature.
        rho_fs : float or None
            Freestream density.
        region : str
            Possible values are ``"weak", "strong"``.

        Returns
        -------
        res : dict
            Contains the quantities just downstream of the specified shock
            wave. The following keys will be used:

            * ``"M"``: Mach number downstream of the shock wave.
            * ``"T"``: Temperature downstream of the shock wave.
            * ``"p"``: Pressure downstream of the shock wave.
            * ``"rho"``: Density downstream of the shock wave.
            * ``"T0"``: Total temperature downstream of the shock wave.
            * ``"p0"``: Total pressure downstream of the shock wave.
            * ``"rho0"``: Total density downstream of the shock wave.

        Examples
        --------

        Consider a simple regular reflection from a solid boundary:

        >>> from pygasflow.shockwave import PressureDeflectionLocus
        >>> gamma = 1.4
        >>> M1 = 2.8
        >>> theta1 = 16  # deg
        >>> T1 = 519     # °R
        >>> p1 = 1       # atm
        >>> l1 = PressureDeflectionLocus(M=M1, gamma=gamma, label="1")
        >>> l2 = l1.new_locus_from_shockwave(theta1, label="2")

        To compute quantities downstream of the second shock wave:

        >>> region3 = l2.flow_quantities_after_shockwave(0, p1, T1, None)

        """
        from pygasflow.isentropic import (
            pressure_ratio as ise_pressure_ratio,
            temperature_ratio as ise_temperature_ratio,
            density_ratio as ise_density_ratio,
        )

        # retrieve the freestream Mach number.
        M_fs = None
        locus = self
        while locus is not None:
            M_fs = locus.M
            locus = locus.upstream_locus

        M_fs, T_fs, p_fs, rho_fs = [
            t if t is not None else np.nan for t in [M_fs, T_fs, p_fs, rho_fs]]

        if (
            (self.upstream_locus is None)
            and np.isclose(theta, 0)
            and (region == "weak")
        ):
            M_d = M_fs
            T_d = T_fs
            p_d = p_fs
            rho_d = rho_fs
        else:
            shock = self.shockwave_at_theta(theta, region)
            M_d = shock["md"]
            T_d = shock["tr"] * T_fs
            p_d = shock["pr"] * p_fs
            rho_d = shock["dr"] * rho_fs

        p0_d = 1 / ise_pressure_ratio(M_d, self.gamma) * p_d
        rho0_d = 1 / ise_density_ratio(M_d, self.gamma) * rho_d
        # total temperature doesn't change across a shock wave
        T0_d = 1 / ise_temperature_ratio(M_fs, self.gamma) * T_fs

        res = {
            "M": M_d, "T": T_d, "p": p_d, "rho": rho_d,
            "T0": T0_d, "p0": p0_d, "rho0": rho0_d
        }
        return res

    @staticmethod
    def create_path(*segments, concatenate=True, **kwargs):
        """Create a path connecting one or more segments.

        Parameters
        ----------
        segments : tuple
            2-elements tuple where the first element is a PressureDeflectionLocus
            and the second element is the deflection angle [degrees] of the
            end of the segment.
        concatenate : bool
            If True, concatenate the results of each segments, thus returning
            two numpy arrays. If False, returns two lists where each element
            is the result for each segment.
        **kwargs :
            Keyword arguments passed to
            :func:`~PressureDeflectionLocus.pressure_deflection_segment`.

        Returns
        -------
        theta : np.ndarray or list
            Array of deflection angles [degrees] or list of arrays of
            deflection angles [degrees], depending on ``concatenate``.
        pr : np.ndarray or list
            Array of pressure ratios to freestream, or list of arrays of
            pressure ratios, depending on ``concatenate``.

        Examples
        --------

        .. plot::
            :context: reset
            :include-source: True

            from pygasflow.shockwave import PressureDeflectionLocus
            import matplotlib.pyplot as plt

            gamma = 1.4
            M1 = 3
            theta = 25
            locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
            locus2 = locus1.new_locus_from_shockwave(theta)

            theta1, pr1 = locus1.pressure_deflection(include_mirror=True)
            theta2, pr2 = locus2.pressure_deflection(include_mirror=True)
            theta_segment, pr_segment = locus1.create_path(
                (locus1, locus2.theta_origin),
                (locus2, 15)
            )

            fig, ax = plt.subplots()
            ax.plot(theta1, pr1, ":", label="M1")
            ax.plot(theta2, pr2, ":", label="M2")
            ax.plot(theta_segment, pr_segment, label="segment")
            ax.set_xlabel(r"Deflection Angle $\theta$ [deg]")
            ax.set_ylabel("Pressure Ratio to Freestream")
            ax.legend()
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', alpha=0.7)
            ax.grid(which='minor', linestyle=':', alpha=0.5)
            plt.show()

        """
        if not all(
            isinstance(t, (tuple, list))
            and (len(t) == 2)
            and isinstance(t[0], PressureDeflectionLocus)
            and isinstance(t[1], Number)
            for t in segments
        ):
            raise ValueError(
                "Each segment of the path must be a 2-elements tuple, where"
                " the first element is an instance of PressureDeflectionLocus"
                " and the second element is the final deflection angle"
                " [degrees] for the specified segment."
            )
        theta_list = []
        pr_list = []
        for (pdl, final_theta) in segments:
            final_theta -= pdl.theta_origin
            theta, pr = pdl.pressure_deflection_segment(final_theta, **kwargs)
            theta_list.append(theta)
            pr_list.append(pr)
        if concatenate:
            theta = np.concatenate(theta_list)
            pr = np.concatenate(pr_list)
            return theta, pr
        return theta_list, pr_list

    def intersection(self, other, region="weak", a=None, b=None):
        """Find the intersection point of this locus with another one.

        Note: this is an experimental function.

        Parameters
        ----------
        other : PressureDeflectionLocus
        region : str
            Can be ``"weak"`` or ``"strong"``.

        Returns
        -------
        theta_intersection : float
            The deflection angle [degrees] of the intersection point. If there
            is no intersection, None will be returned.
        pr_intersection : float
            The pressure ratio of the intersection point. If there
            is no intersection, None will be returned.

        Examples
        --------

        Example of the intersection of shocks of opposite families:

        >>> from pygasflow.shockwave import PressureDeflectionLocus
        >>> gamma = 1.4
        >>> M1 = 3
        >>> theta2 = 20
        >>> theta3 = -15
        >>> locus1 = PressureDeflectionLocus(M=M1, gamma=1.4, label="1")
        >>> locus2 = locus1.new_locus_from_shockwave(theta2, label="2")
        >>> locus3 = locus1.new_locus_from_shockwave(theta3, label="2")
        >>> phi, p4_p1 = locus2.intersection(locus3)
        >>> phi
        4.795958931693682
        >>> p4_p1
        np.float64(8.352551913417367)

        """
        if self.theta_origin < other.theta_origin:
            theta_origin1, theta_origin2 = self.theta_origin, other.theta_origin
            theta_max1, theta_max2 = self.theta_max, other.theta_max
            f1, f2 = self.shockwave_at_theta, other.shockwave_at_theta
        else:
            theta_origin1, theta_origin2 = other.theta_origin, self.theta_origin
            theta_max1, theta_max2 = other.theta_max, self.theta_max
            f1, f2 = other.shockwave_at_theta, self.shockwave_at_theta

        if theta_origin1 + theta_max1 < theta_origin2 - theta_max2:
            # no intersection found
            return None, None

        # TODO: This approach is not going to work if the intersection happens
        # between the weak region in one locus and the strong region in
        # the other. Maybe I should run a bisection with func(pressure_ratio)..
        def func(theta, region="weak"):
            res1 = f1(theta, region)
            res2 = f2(theta, region)
            pr1 = res1["pr"]
            pr2 = res2["pr"]
            return pr1 - pr2

        if a is None:
            a = min(theta_origin1 + theta_max1, theta_origin2 - theta_max2)
        if b is None:
            b = max(theta_origin1 + theta_max1, theta_origin2 - theta_max2)
        theta_intersection = bisect(func, a=a, b=b, args=(region,))
        pr_intersection = f1(theta_intersection, region=region)["pr"]
        return theta_intersection, pr_intersection

    def pressure_deflection(self, N=100, include_mirror=True):
        """
        Helper function to build Pressure-Deflection plots. Computes the locus
        of all possible static pressure values behind oblique shock wave for
        given upstream conditions.

        Parameters
        ----------
        N : int, optional
            Number of points discretizing the range [0, theta_max].
            This function compute N points in the range [0, theta_max] for
            the `'weak'` solution, then compute N points in the range
            [theta_max, 0] for the `'strong'` solution.
            If ``include_mirror=False``, the length of the returned arrays is
            2*N, otherwise is 4*N.
        include_mirror : bool
            If False, return numerical arrays for 0 <= theta <= theta_max.
            If True, mirror the arrays in order to get data from
            -theta_max <= theta <= theta_max.

        Returns
        -------
        theta : array_like
            Deflection angles
        pr : array_like
            Pressure ratios computed for the given Mach and the above
            deflection angles.

        Examples
        --------

        Plot half locus for the upstream condition, and the full locus for the
        downstream condition:

        .. plot::
            :context: reset
            :include-source: True

            from pygasflow.shockwave import PressureDeflectionLocus
            import matplotlib.pyplot as plt

            gamma = 1.4
            M1 = 3
            theta = 25
            locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
            locus2 = locus1.new_locus_from_shockwave(theta)

            theta_1, pr_1 = locus1.pressure_deflection(include_mirror=False)
            theta_2, pr_2 = locus2.pressure_deflection(include_mirror=True)

            fig, ax = plt.subplots()
            ax.plot(theta_1, pr_1, label=f"M = {locus1.M}")
            ax.plot(theta_2, pr_2, label=f"M = {locus2.M}")
            ax.set_xlabel(r"Deflection Angle $\theta$ [deg]")
            ax.set_ylabel("Pressure Ratio to Freestream")
            ax.legend()
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', alpha=0.7)
            ax.grid(which='minor', linestyle=':', alpha=0.5)
            plt.show()

        """
        _check_pressure_deflection(N)

        # mandatory step in order to skip checks
        M1 = np.atleast_1d(self.M).astype(np.float64)
        gamma = self.gamma

        theta_max = max_theta_from_mach.__no_check__(M1, gamma)
        # place the majority of points in the proximity of theta_max
        spacing = 1 - np.geomspace(1e-04, 1, N)
        spacing[0] = 1
        theta = theta_max * spacing
        theta = np.append(theta[::-1], theta)
        beta = np.zeros_like(theta)

        for i in range(N):
            betas = beta_from_mach_theta.__no_check__(M1, theta[i], gamma)
            beta[i] = betas["weak"]
            beta[len(theta) -1 - i] = betas["strong"]

        # TODO:
        # it may happend to get a NaN, especially for theta=0, in that case
        # manual correction:
        # idx = np.where(np.isnan(beta))
        # beta[idx] = 1

        Mn = M1 * np.sin(np.deg2rad(beta))
        pr = pressure_ratio.__no_check__(Mn, gamma)

        if include_mirror:
            theta = np.append(theta, -theta[::-1])
            pr = np.append(pr, pr[::-1])

        theta += self.theta_origin
        pr *= self.pr_to_fs_at_origin

        return theta, pr

    def pressure_deflection_split(
        self, N=100, include_mirror=False, mode="region"
    ):
        """
        Helper function to build Pressure-Deflection plots. Computes the locus
        of all possible static pressure values behind oblique shock wave for
        given upstream conditions, and split them according to the user
        selected mode.

        Parameters
        ----------
        N : int, optional
            Number of points discretizing the range [0, theta_max].
            This function compute N points in the range [0, theta_max] for
            the `'weak'` solution, then compute N points in the range
            [theta_max, 0] for the `'strong'` solution.
            If ``include_mirror=False``, the length of the returned arrays is
            2*N, otherwise is 4*N.
        include_mirror : bool
            If False, return numerical arrays for 0 <= theta <= theta_max.
            If True, mirror the arrays in order to get data from
            -theta_max <= theta <= theta_max.
        mode : str
            Split the locus at some point. It can be:

            * ``"region"``: the locus is splitted at the detachment point,
              where ``theta=theta_max``.
            * ``"sonic"``: the locus is splitted at the sonic point (where
              the downstream Mach number is 1).

        Returns
        -------
        theta_split_1 : array_like
        pr_split_1 : array_like
        theta_split_2 : array_like
        pr_split_2 : array_like

        Examples
        --------

        Consider a simple regular reflection process, in which there
        are two shock waves.
        Plot half locus for the upstream condition and split it between the
        weak and strong regions. Also plot another full locus, splitted between
        the subsonic and supersonic region:

        .. plot::
            :context: reset
            :include-source: True

            from pygasflow.shockwave import PressureDeflectionLocus
            import matplotlib.pyplot as plt

            gamma = 1.4
            M1 = 3
            theta = 25
            locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
            locus2 = locus1.new_locus_from_shockwave(theta)

            theta_1w, pr_1w, theta_1s, pr_1s = locus1.pressure_deflection_split(
                include_mirror=False, mode="region")
            theta_2sup, pr_2sup, theta_2sub, pr_2sub = locus2.pressure_deflection_split(
                include_mirror=True, mode="sonic")

            fig, ax = plt.subplots()
            ax.plot(theta_1w, pr_1w, label="M1 weak")
            ax.plot(theta_1s, pr_1s, label="M1 strong")
            ax.plot(theta_2sup, pr_2sup, label="M2 (M3 supersonic)")
            ax.plot(theta_2sub, pr_2sub, label="M2 (M3 subsonic)")
            ax.set_xlabel(r"Deflection Angle $\theta$ [deg]")
            ax.set_ylabel("Pressure Ratio to Freestream")
            ax.legend()
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', alpha=0.7)
            ax.grid(which='minor', linestyle=':', alpha=0.5)
            plt.show()

        """
        mode = mode.lower()
        allowed_modes = ["region", "sonic"]
        if mode not in allowed_modes:
            raise ValueError(
                f"`mode='{mode}' not recognized. Available modes are:"
                f" {allowed_modes}")

        t, pr = self.pressure_deflection(N=N, include_mirror=False)

        if mode == "region":
            idx = np.array(list(range(len(t))), dtype=int)
            n = len(t) / 2
            weak_idx = idx[idx < n]
            strong_idx = idx[idx >= n]
            theta_split_1, pr_split_1 = t[weak_idx], pr[weak_idx]
            theta_split_2, pr_split_2 = t[strong_idx], pr[strong_idx]
        else:
            idx = (pr <= self.sonic_point[1])
            theta_split_1, pr_split_1 = t[idx], pr[idx]
            theta_split_2, pr_split_2 = t[~idx], pr[~idx]

        if include_mirror:
            theta_split_1 -= self.theta_origin
            theta_split_2 -= self.theta_origin

            theta_split_1 = np.concatenate([-theta_split_1[::-1], theta_split_1])
            pr_split_1 = np.concatenate([pr_split_1[::-1], pr_split_1])
            theta_split_2 = np.concatenate([theta_split_2, -theta_split_2[::-1]])
            pr_split_2 = np.concatenate([pr_split_2, pr_split_2[::-1]])

            theta_split_1 += self.theta_origin
            theta_split_2 += self.theta_origin

        return theta_split_1, pr_split_1, theta_split_2, pr_split_2

    def pressure_deflection_segment(self, theta, region="weak", N=100):
        """
        Helper function to build Pressure-Deflection plots. Computes the static
        pressure values behind oblique shock wave for given upstream
        conditions, in a specified deflection angle range [0, theta].

        Parameters
        ----------
        theta : float
            The deflection angle up to which the segment has to be computed.
        region : string
            Can be ``'weak'`` or ``'strong'``. If the latter is chosen, the segment
            starts from theta=0, goes to theta=theta_max and then it proceeds to
            the user provided value.
        N : int, optional
            Number of points discretizing the range [0, theta].

        Returns
        -------
        theta : array_like
            Deflection angles
        pr : array_like
            Pressure ratios computed for the given Mach and the above
            deflection angles.

        Examples
        --------

        Segment connecting the upstream condition to the downstream condition:

        .. plot::
            :context: reset
            :include-source: True

            from pygasflow.shockwave import PressureDeflectionLocus
            import matplotlib.pyplot as plt

            gamma = 1.4
            M1 = 3
            theta = 25
            locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
            locus2 = locus1.new_locus_from_shockwave(theta)

            theta1, pr1 = locus1.pressure_deflection(include_mirror=True)
            theta2, pr2 = locus2.pressure_deflection(include_mirror=True)
            theta_segment, pr_segment = locus1.pressure_deflection_segment(theta)

            fig, ax = plt.subplots()
            ax.plot(theta1, pr1, ":", label="M1")
            ax.plot(theta2, pr2, ":", label="M2")
            ax.plot(theta_segment, pr_segment, label="segment")
            ax.set_xlabel(r"Deflection Angle $\theta$ [deg]")
            ax.set_ylabel("Pressure Ratio to Freestream")
            ax.legend()
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', alpha=0.7)
            ax.grid(which='minor', linestyle=':', alpha=0.5)
            plt.show()

        """
        _check_pressure_deflection(N)

        region = region.lower()
        if region not in ["weak", "strong"]:
            raise ValueError("`region` must be 'weak' or 'strong'.")

        is_right_running_wave = theta < 0
        theta = abs(theta)
        theta_max = max_theta_from_mach(self.M, self.gamma)
        betas = np.zeros(N)
        pr = np.zeros(N)

        if region == "weak":
            theta = np.linspace(0, theta, N)
            for i, t in enumerate(theta):
                betas[i] = beta_from_mach_theta(
                    self.M, theta[i], self.gamma)["weak"]
        else:
            N1 = int((theta_max / (2 * theta_max - theta)) * N)
            N2 = N - N1
            theta = np.append(
                np.linspace(0, theta_max, N1),
                np.linspace(theta_max, theta, N2 + 1)[1:]
            )
            for i, t in enumerate(theta):
                b = beta_from_mach_theta(self.M, theta[i], self.gamma)
                betas[i] = b["weak"] if i < N1 else b["strong"]

        Mn = self.M * np.sin(np.deg2rad(betas))
        # avoid rounding errors that would make Mn < 1
        idx = np.isclose(Mn, 1)
        Mn[idx] = 1
        pr = pressure_ratio(Mn, self.gamma)

        if is_right_running_wave:
            theta = -theta

        theta += self.theta_origin
        pr *= self.pr_to_fs_at_origin

        return theta, pr


###########################################################################################
################################# Conical Flow Relations ##################################
###########################################################################################


def taylor_maccoll(theta, V, gamma=1.4):
    """
    Taylor-Maccoll differential equation for conical shock wave.

    Parameters
    ----------
    theta : float
        Polar coordinate, angle in radians.
    V : list
        Velocity Vector with components:

        * V_r: velocity along the radial direction
        * V_theta: velocity along the polar direction

    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    dV_dtheta : list
        Taylor-Maccoll differential equation.

    See Also
    --------
    mach_cone_angle_from_shock_angle
    """
    # Reorganizing Anderson's equation 10.15:
    V_r, V_theta = V

    dV_dtheta = [
        V_theta,
        (V_r * V_theta**2 - (gamma - 1) / 2 * (1 - V_r**2 - V_theta**2) * (2 * V_r + V_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - V_r**2 - V_theta**2) - V_theta**2)
    ]

    return dV_dtheta


@check_shockwave
def nondimensional_velocity(M, gamma=1.4):
    """
    Compute the Nondimensional Velocity given the Mach number.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M >= 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    V : array_like
        Nondimensional Velocity
    """
    # Anderson's equation 10.16
    return np.sqrt((gamma - 1) * M**2 / (2 + (gamma - 1) * M**2))


@check_shockwave
def mach_from_nondimensional_velocity(V, gamma=1.4):
    """
    Compute the Mach number given the Nondimensional Velocity.

    Parameters
    ----------
    V : array_like
        Nondimensional Velocity. If float, list, tuple is given as input,
        a conversion will be attempted. Must be V > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    M : array_like
        Mach number
    """
    if np.any(V < 0) or np.any(V > 1):
        raise ValueError("Nondimensional velocity must be 0 <= V <= 1.")
    # TODO: can I modify check_shockwave to perform the conversion for V?
    is_scalar = isinstance(V, Number)
    V = np.atleast_1d(V)
    M = np.inf * np.ones_like(V)
    idx = ~np.isclose(V, 1)
    # inverse Anderson's equation 10.16
    M[idx] = np.sqrt(2 * V[idx]**2 / ((gamma - 1) * (1 - V[idx]**2)))
    if is_scalar:
        return M[0]
    return M


# @check
def mach_cone_angle_from_shock_angle(M, beta, gamma=1.4):
    """
    Compute the half-cone angle and the Mach number at the surface of the cone.
    NOTE: this function is undecorated, hence no check is performed to assure
    the validity of the input parameters. It's up to the user to assure that.

    Parameters
    ----------
    M : float
        Upstream Mach number. Must be > 1.
    beta : float
        Shock Angle in degrees. Must be mach_angle <= beta <= 90.
        NOTE: no check is done over beta. If an error is raised during the
        computation, make sure beta is at least ever so slightly bigger than
        the mach angle.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Mc : float
        Mach number at the surface of the cone
    theta_c : float
        Half-cone angle in degrees.


    See Also
    --------
    taylor_maccoll
    """
    # NOTE:
    # Here we implement the first 3-ish steps of Anderson's "Modern Compressible
    # Flow" procedure for solving the flow over a circular cone (Chapter 10,
    # Section 4). The 'ish' part indicates that this function only solve the
    # first part of step 3. The other part is left for other functions.

    # Step 1. Compute M2 and theta (delta, in the book), just behind the shock.
    # delta = flow deflection angle
    delta = theta_from_mach_beta.__no_check__(M, beta, gamma)

    # check for detachment
    if delta < 0 or np.isnan(delta):
        raise ValueError("Detachment detected for the following conical flow condition:\n" +
            "M = {}\n".format(M) +
            "beta = {}\n".format(beta) +
            "gamma = {}".format(gamma))

    # Mach downstream of the shock wave
    MN1 = M * np.sin(np.deg2rad(beta))
    MN2 = mach_downstream(MN1, gamma)
    M_2 = MN2 / np.sin(np.deg2rad(beta - delta))

    # Step 2. Compute V' (Vn), Vr' and Vtheta' directly behind of the shock.
    # compute the nondimensional velocity components
    V0 = nondimensional_velocity.__no_check__(M_2, gamma)
    V0_r = V0 * np.cos(np.deg2rad(beta - delta))
    V0_theta = -V0 * np.sin(np.deg2rad(beta - delta))

    # range of integration (better avoid zero, thus suppressing warnings)
    thetas = [np.deg2rad(beta), 1e-08]
    # initial condition
    V0_ic = [V0_r, V0_theta]

    # Step 3. Solve the Taylor-Maccoll equation
    # event for detecting the root V_theta = 0. When found, stop the integration
    def event(theta, V):
        return V[1]
    event.terminal = True

    # solve_ivp vs odeint: solve_ivp is the new wrapper to ode integration methods.
    # It also allows to stop the integration when certain root-finding events are detected.
    # In our case it translates to better performance, since we son't need to integrate
    # over the entire thetas range.
    # Differently from odeint, solve_ivp don't use "args=(,)" to pass arguments to the
    # differential equation, therefore we need to wrap such function with a lambda.
    result = solve_ivp(lambda theta, V: taylor_maccoll(theta, V, gamma), thetas, V0_ic,
            events=event, rtol=1e-08, atol=1e-10)

    if not result.success:
        raise Exception("Could not successfully integrate Taylor-Maccoll equation.\n" +
            "Here some useful data for debug:\n" +
            "\tInput Mach number: {}\n".format(M) +
            "\tInput Shock Wave angle [degrees]: {}\n".format(beta) +
            "\tRange of integration [degrees]: {}\n".format(np.rad2deg(thetas)) +
            "\tInitial Conditions: [Vr = {}, V_theta = {}]\n".format(V0_r, V0_theta)
            )

    # the cone angle is the angle where V_theta = 0.
    theta_c = np.rad2deg(result.t[-1])
    # at the cone surface, V_theta = 0, therefore V = V_r
    Vc = result.y[0, -1]
    # Mach number at the cone surface
    Mc = mach_from_nondimensional_velocity.__no_check__(Vc, gamma)

    return Mc, theta_c


@check_shockwave
def shock_angle_from_mach_cone_angle(M1, theta_c, gamma=1.4, flag="weak"):
    """
    Compute the shock wave angle given the upstream mach number and the
    half-cone angle.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. Must be > 1.
    theta_c : float
        Half cone angle in degrees. Must be 0 < theta_c < 90
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    flag : string, optional
        Can be either ``'weak'`` or ``'strong'``. Default to ``'weak'``
        (in conical shockwaves, the strong solution is rarely encountered).

    Returns
    -------
    Mc : float
        Mach number at the surface of the cone, computed.
    theta_c : float
        Half cone angle of the cone in degrees (which was provided in input).
    beta : float
        Shock wave angle in degrees.
    """
    if theta_c < 0:
        raise ValueError("The half-cone angle must be > 0.")

    def function(M):
        # find the theta_c_max associated to the given Mach number in order to
        # chose the correct bisection interval for 'weak' or 'strong' solution.
        Mc, tcmax, bmax = max_theta_c_from_mach(M, gamma)
        if theta_c > tcmax:
            raise ValueError("Detachment detected: can't solve the flow when theta_c > theta_c_max.\n" +
                "M1 = {}\n".format(M1) +
                "theta_c_max(M1) = {}\n".format(tcmax) +
                "theta_c = {}\n".format(theta_c))

        # need to add a small offset otherwise mach_cone_angle_from_shock_angle
        # could crash
        bmin = np.rad2deg(np.arcsin(1 / M)) + 1e-08
        if flag == "strong":
            bmin, bmax = bmax, bmin
            bmax = 90

        def func(beta):
            m, t = mach_cone_angle_from_shock_angle(M, beta, gamma)
            return theta_c - t

        # shockwave angle associated to the provided M, theta_c
        beta = bisect(func, bmin, bmax)
        Mc, theta_c_comp = mach_cone_angle_from_shock_angle(M, beta)
        return Mc, theta_c, beta

    theta_c_comp = np.zeros_like(M1)
    beta = np.zeros_like(M1)
    Mc = np.zeros_like(M1)
    for i, m in enumerate(M1):
        Mc[i], theta_c_comp[i], beta[i] = function(m)
    return ret_correct_vals(Mc), ret_correct_vals(theta_c_comp), ret_correct_vals(beta)


@check_shockwave
def shock_angle_from_machs(M1, Mc, gamma=1.4, flag="weak"):
    """
    Compute the shock wave angle given the upstream mach number and the mach
    number at the surface of the cone.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. Must be > 1.
    Mc : float
        Mach number at the surface of the cone.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    flag : string, optional
        Can be either ``'weak'`` or 'strong'. Default to ``'weak'``
        (in conical shockwaves, the strong solution is rarely encountered).

    Returns
    -------
    Mc : float
        Mach number at the surface of the cone, computed.
    theta_c : float
        Half cone angle of the cone in degrees.
    beta : float
        Shock wave angle in degrees.
    """

    def function(M):
        # find the theta_c_max associated to the given Mach number in order to
        # chose the correct bisection interval for 'weak' or 'strong' solution.
        _, tcmax, bmax = max_theta_c_from_mach(M, gamma)
        # need to add a small offset to avoid stalling the integration process
        bmin = np.rad2deg(np.arcsin(1 / M)) + 1e-08
        if flag == "strong":
            bmin, bmax = bmax, bmin
            bmax = 90

        def func(beta):
            m, t = mach_cone_angle_from_shock_angle(M, beta, gamma)
            return m - Mc

        # shockwave angle associated to the provided M, theta_c
        beta = bisect(func, bmin, bmax)
        Mc_comp, theta_c_comp = mach_cone_angle_from_shock_angle(M, beta, gamma)

        # TODO: can I remove this check?
        if theta_c_comp > tcmax:
            raise ValueError(
                "The provided half-cone angle is greater than the maximum half-cone " +
                "angle for the provided Mach number, M = {}. The shockwave is ".format(M) +
                "detached, hence the solution can't be computed with this function."
            )

        return Mc_comp, theta_c_comp, beta

    theta_c_comp = np.zeros_like(M1)
    beta_comp = np.zeros_like(M1)
    Mc_comp = np.zeros_like(M1)
    for i, m1 in enumerate(M1):
        Mc_comp[i], theta_c_comp[i], beta_comp[i] = function(m1)
    return ret_correct_vals(Mc_comp), ret_correct_vals(theta_c_comp), ret_correct_vals(beta_comp)


@check_shockwave
def max_theta_c_from_mach(M1, gamma=1.4):
    """
    Compute the maximum cone angle and the corresponding shockwave angle for a
    given upstream Mach number.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Mc : ndarray
        Mach number at the cone surface
    Theta_c_max : ndarray
        Maximum cone angle theta_c in degrees
    beta : ndarray
        Shockwave angle corresponding to the maximum cone angle and provided
        Mach number.
    """
    def function(M):
        # initial search interval
        offset = 1e-08
        betas = [np.rad2deg(np.arcsin(1 / M)) + offset, 90 - offset]

        def func(beta):
            Mc, tc = mach_cone_angle_from_shock_angle(M, beta, gamma)
            # Look at Mach-Theta_s-Theta_c plot to understand why the minus sign
            return -tc

        result = minimize_scalar(func, bounds=betas, method='bounded')
        if not result.success:
            raise Exception("Could not successfully find the maximum cone angle " +
                "for M = {}".format(M))
        # shockwave angle corresponding to theta_c_max
        beta = result.x
        Mc, theta_c_max = mach_cone_angle_from_shock_angle(M, beta, gamma)
        return Mc, theta_c_max, beta

    theta_c_max = np.zeros_like(M1)
    beta = np.zeros_like(M1)
    Mc = np.zeros_like(M1)
    for i, m in enumerate(M1):
        Mc[i], theta_c_max[i], beta[i] = function(m)
    return ret_correct_vals(Mc), ret_correct_vals(theta_c_max), ret_correct_vals(beta)


@check_shockwave
def detachment_point_conical_shock(M1, gamma=1.4):
    """
    Compute the maximum cone angle and the corresponding shockwave angle for a
    given upstream Mach number.

    Note: this is a wrapper function to
    :func:`~pygasflow.shockwave.max_theta_c_from_mach`.

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    beta : ndarray
        Shockwave angle corresponding to the maximum cone angle and provided
        Mach number.
    theta_c_max : ndarray
        Maximum cone angle theta_c in degrees

    See Also
    --------
    max_theta_c_from_mach, sonic_point_conical_shock
    """
    _, theta_c_max, beta = max_theta_c_from_mach.__no_check__(M1, gamma)
    return beta, theta_c_max


@check_shockwave
def sonic_point_conical_shock(M1, gamma=1.4):
    """Given an upstream Mach number, compute the point (beta, theta_c)
    where the downstream Mach number is sonic. This function is useful to
    compute the location of the sonic line.

    **WARNING:** this procedure is really slow!

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    beta : ndarray
        Shockwave angle corresponding in degrees.
    theta_c : ndarray
        Cone angle theta_c in degrees.

    See Also
    --------
    detachment_point_conical_shock
    """
    # TODO: can this be done faster?
    def function(M1):
        def func(theta_c):
            Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag="weak")
            MN1 = M1 * np.sin(np.deg2rad(beta))
            MN2 = mach_downstream(MN1, gamma)
            # delta is the flow deflection angle (Anderson's Figure 10.4)
            delta = theta_from_mach_beta(M1, beta, gamma)
            M2 = MN2 / np.sin(np.deg2rad(beta - delta))
            return M2 - 1

        if np.isclose(M1, 1):
            return 90, 0

        # theta_c search interval
        b = max_theta_c_from_mach(M1, gamma)[1]
        a = b / 2
        theta_c = bisect(func, a, b)
        Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag="weak")
        return beta, theta_c

    theta_c = np.zeros_like(M1)
    beta = np.zeros_like(M1)
    for i, m in enumerate(M1):
        beta[i], theta_c[i] = function(m)
    return beta, theta_c


@deprecated("Use ``sonic_point_conical_shock`` instead.")
@check_shockwave
def beta_theta_c_for_unit_mach_downstream(M1, gamma=1.4):
    """Alias of ``sonic_point_conical_shock``.
    """
    return sonic_point_conical_shock(M1, gamma)


def load_data(gamma=1.4):
    """ The :func:`sonic_point_conical_shock` function is really
    slow in computing the data. Often, that data is needed in a plot.
    Here, a few precomputed tables has been provided.

    Parameters
    ----------
    gamma : float, optional
        The specific heat ratio.

    Returns
    -------
    M1 : ndarray
        The precomputed upstream Mach numbers
    beta : ndarray
        The shockwave angle associate to M2 = 1 for the precomputed M1
    theta_c : ndarray
        The half-cone angle associate to M2 = 1 for the precomputed M1
    """
    if gamma <= 1:
        raise ValueError("The specific heat ratio must be gamma > 1")

    import pandas as pd
    import os
    # path of the folder containing this file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # path of the folder containing the data of the plot
    data_dir = os.path.join(current_dir, "data")
    filename = os.path.join(data_dir, "m-beta-theta_c-g" + str(gamma) + ".csv.zip")
    if not os.path.exists(filename):
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        raise FileNotFoundError("Could not find the file: {}\n".format(filename) +
        "The following files are available: {}\n".format(files) +
        "Insert the interested specific heat ratio!")
    df = pd.read_csv(filename)
    mach = df["M1"].values
    beta = df["beta"].values
    theta_c = df["theta_c"].values

    return mach, beta, theta_c


def create_mach_beta_theta_c_csv_file(
    M, gamma, folder="", filename="m-beta-theta_c-g%s.csv.zip"
):
    """Create a csv file for each value of gamma containing data for the
    upstream Mach, beta, theta_c where the downstream Mach number is sonic.
    This is useful to speed up the generation of the Mach-beta-theta_c diagram.

    Parameters
    ----------
    M : list or np.array
        Values of upstream Mach numbers where to compute beta, theta_c.
    gamma : list
        Values of the specific heat ratio.
    folder : str
    filename : str
        A formatted string accepting one argument, the gamma value.
    """
    if any(not hasattr(v, "__iter__") for v in [M, gamma]):
        raise TypeError(
            "`M` and `gamma` must be iterables (list, tuple, arrays).")

    import os
    import pandas as pd

    for g in gamma:
        print("Processing gamma = ", g)
        beta = np.zeros_like(M)
        theta_c = np.zeros_like(M)
        for i, m in enumerate(M):
            try:
                beta[i], theta_c[i] = sonic_point_conical_shock(m, g)
            except ValueError:
                beta[i], theta_c[i] = np.nan, np.nan
        df = pd.DataFrame.from_dict({
            "M1": M, "beta": beta, "theta_c": theta_c
        })
        filepath = os.path.join(folder, filename % g)
        df.to_csv(filepath, index=False, compression="zip")
