import numpy as np
from scipy.optimize import bisect

from pygasflow.utils.roots import apply_bisection
from pygasflow.utils.decorators import check

# TODO:
# 1. In the functions where the bisection is used, evaluate if the initial mach range
#   for supersonic case is sufficient for all gamma and ratios.
# 2. Evaluate if scipy.brentq and scipy.brenth performs better than scipy.bisect
# 3. Provide tolerance and maxiter arguments to the function where bisect is used.

# NOTE: certain function could very well be computed by using other functions.
#   For instance, Critical_Temperature_Ratio could be computed using Temperature_Ratio.
#   In doing so, a performance penalty is introduced, that could become significant when
#   computing millions of data points. This is mostly the case with functions that get
#   called by root finding methods.
#   Therefore, I use plain formulas as much as possible.

@check
def critical_velocity_ratio(M, gamma=1.4):
    """
    Compute the critical velocity ratio U/U*.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Critical velocity ratio U/U*.
    """
    # need to deal with division by 0
    ratio = np.zeros_like(M)
    idx = M != 0
    ratio[idx] = 1 / ((2 / (gamma + 1))**(1 / (gamma - 1)) / M[idx] * (1 + (gamma - 1) / 2 * M[idx]**2)**0.5)
    return ratio

@check
def critical_temperature_ratio(M, gamma=1.4):
    """
    Compute the critical temperature ratio T/T*.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Critical Temperature ratio T/T*.
    """
    # return Temperature_Ratio.__no_check(M, gamma) * 0.5 * (gamma + 1)
    return ((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2)

@check
def critical_pressure_ratio(M, gamma=1.4):
    """
    Compute the critical pressure ratio P/P*.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Critical Pressure ratio P/P*.
    """
    # return Pressure_Ratio.__no_check(M, gamma) * (0.5 * (gamma + 1))**(gamma / (gamma - 1))
    return (((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2))**(gamma / (gamma - 1))

@check
def critical_density_ratio(M, gamma=1.4):
    """
    Compute the critical density ratio rho/rho*.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Critical density ratio rho/rho*.
    """
    # TODO: this first version appears to be faster!
    return density_ratio.__no_check(M, gamma) * (0.5 * (gamma + 1))**(1 / (gamma - 1))
    # return (((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2))**(1 / (gamma - 1))

@check
def critical_area_ratio(M, gamma=1.4):
    """
    Compute the critical area ratio A/A*.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Critical area ratio A/A*.
    """
    # return 1  / Critical_Density_Ratio.__no_check(M, gamma) * np.sqrt(1 / Critical_Temperature_Ratio.__no_check(M, gamma)) / M
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return (((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))) / M


@check
def pressure_ratio(M, gamma=1.4):
    """
    Compute the pressure ratio P/P0.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Pressure ratio P/P0.
    """
    return (1 + (gamma - 1) / 2 * M**2)**(-gamma / (gamma - 1))

@check
def temperature_ratio(M, gamma=1.4):
    """
    Compute the temperature ratio T/T0.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Temperature ratio T/T0.
    """
    return 1 / (1 + (gamma - 1) / 2 * M**2)

@check
def density_ratio(M, gamma=1.4):
    """
    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Density ratio rho/rho0.
    """
    return (1 + (gamma - 1) / 2 * M**2)**(-1 / (gamma - 1))

@check
def m_from_temperature_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Temperature Ratio T/T0.

    Parameters
    ----------
    ratio : array_like
        Isentropic Temperature Ratio T/T0. If float, list, tuple is given
        as input, a conversion will be attempted. Must be 0 <= T/T0 <= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("Temperature ratio must be 0 <= T/T0 <= 1.")
    return np.sqrt(2 * (1 / ratio - 1) / (gamma - 1))

@check
def m_from_pressure_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Pressure Ratio P/P0.

    Parameters
    ----------
    ratio : array_like
        Isentropic Pressure Ratio P/P0. If float, list, tuple is given
        as input, a conversion will be attempted. Must be 0 <= P/P0 <= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("Pressure ratio must be 0 <= P/P0 <= 1.")
    return np.sqrt(2 / (gamma - 1) * (1 / ratio**((gamma - 1) / gamma) - 1))

@check
def m_from_density_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Density Ratio rho/rho0.

    Parameters
    ----------
    ratio : array_like
        Isentropic Density Ratio rho/rho0. If float, list, tuple is given
        as input, a conversion will be attempted. Must be 0 <= rho/rho0 <= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("Density ratio must be 0 <= rho/rho0 <= 1.")
    return np.sqrt(2 / (gamma - 1) * (1 / ratio**(gamma - 1) - 1))

@check
def m_from_critical_area_ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given the Isentropic Critical Area Ratio A/A*.

    Parameters
    ----------
    ratio : array_like
        Isentropic Critical Area Ratio A/A*. If float, list, tuple is given
        as input, a conversion will be attempted. Must be A/A* >= 1.
    flag : string, optional
        Can be either ``'sub'`` (subsonic) or ``'sup'`` (supersonic).
        Default to ``'sub'``.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(ratio < 1):
        raise ValueError("Area ratio must be A/A* >= 1.")

    func = lambda M, r: r - (((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))) / M
    # func = lambda M, r: r - Critical_Area_Ratio.__no_check(M, gamma)
    return apply_bisection(ratio, func, flag=flag)

# with arguments True, I want to convert to np.ndarray the first two parameters
@check([0, 1])
def m_from_critical_area_ratio_and_pressure_ratio(a_ratio, p_ratio, gamma=1.4):
    """
    Compute the Mach number given the Critical Area Ratio (A/A*) and
    the Pressure Ratio (P/P0).

    Parameters
    ----------
    a_ratio : array_like
        Isentropic Critical Area Ratio A/A*. If float, list, tuple is given
        as input, a conversion will be attempted. Must be A/A* >= 1.
    p_ratio : array_like
        Isentropic Pressure Ratio (P/P0). If float, list, tuple is given
        as input, a conversion will be attempted. Must be 0 <= P/P0 <= 1.
        If array_like, must be a_ratio.shape == p_ratio.shape.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(a_ratio < 1):
        raise ValueError("Area ratio must be A/A* >= 1.")
    if np.any(p_ratio < 0) or np.any(p_ratio > 1):
        raise ValueError("Pressure ratio must be 0 <= P/P0 <= 1.")
    if a_ratio.shape != p_ratio.shape:
        raise ValueError("The Critical Area Ratio and Pressure Ratio must have the same number of elements and the same shape.")
    # eq. 5.28, Modern Compressible Flow, 3rd Edition, John D. Anderson
    return np.sqrt(-1 / (gamma - 1) + np.sqrt(1 / (gamma - 1)**2 + 2 / (gamma - 1) * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)) / a_ratio**2 / p_ratio**2))

@check
def m_from_mach_angle(angle, gamma=1.4):
    """
    Compute the Mach number given the Mach Angle.

    Parameters
    ----------
    ratio : array_like
        Mach Angle, [degrees]. If float, list, tuple is given as input,
        a conversion will be attempted. Must be 0 <= Mach Angle <= 90.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(angle < 0) or np.any(angle > 90):
        raise ValueError("Mach angle must be between 0° and 90°.")
    return 1 / np.sin(np.deg2rad(angle))

@check
def mach_angle(M):
    """
    Compute the Mach angle given the Mach number.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted. Must be M > 0.

    Returns
    -------
    out : ndarray
        Mach Angle [degrees].
    """
    # here I use dtype=float to be able to assign np.nan where necessary (np.nan is an IEEE 754 floating point representation of Not a Number)
    angle = np.zeros_like(M, dtype=float)
    angle[M > 1] = np.rad2deg(np.arcsin(1 / M[M > 1]))
    angle[M == 1] = 90
    angle[M < 1] = np.nan
    return angle

@check
def m_from_prandtl_meyer_angle(angle, gamma=1.4):
    """
    Compute the Mach number given the Prandtl Meyer angle.

    Parameters
    ----------
    angle : array_like
        Prandtl Meyer angle [degrees].
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    M : array_like
        Mach Number.
    """
    nu_max = (np.sqrt((gamma + 1) / (gamma - 1)) - 1) * 90
    if np.any(angle < 0) or np.any(angle > nu_max):
        raise ValueError("Prandtl-Meyer angle must be between 0 and {}".format(nu_max))
    angle = np.deg2rad(angle)

    func = lambda M, a: a - (np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1)))

    return apply_bisection(angle, func, flag="sup")

@check
def prandtl_meyer_angle(M, gamma=1.4):
    """
    Compute the Prandtl Meyer function given the Mach number.

    Parameters
    ----------
    M : array_like
        Mach number. If float, list, tuple is given as input, a conversion
        will be attempted.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    nu : array_like
        Prandtl Meyer angle [degrees].
    """
    # here I use dtype=float to be able to assign np.nan where necessary (np.nan is an IEEE 754 floating point representation of Not a Number)
    nu = np.zeros_like(M, dtype=float)
    # Equation (4.44), Anderson
    nu[M > 1] = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M[M > 1]**2 - 1)))
    nu[M > 1] = nu[M > 1] - np.arctan(np.sqrt(M[M > 1]**2 - 1))
    nu[M > 1] = np.rad2deg(nu[M > 1])
    nu[M == 1] = 0
    nu[M < 1] = np.nan
    return nu

@check
def get_ratios_from_mach(M, gamma):
    """
    Compute all isentropic ratios given the Mach number.

    Parameters
    ----------
    M : array_like
        Mach number
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    pr : array_like
        Pressure Ratio P/P0
    dr : array_like
        Density Ratio rho/rho0
    tr : array_like
        Temperature Ratio T/T0
    prs : array_like
        Critical Pressure Ratio P/P*
    drs : array_like
        Critical Density Ratio rho/rho*
    trs : array_like
        Critical Temperature Ratio T/T*
    urs : array_like
        Critical Velocity Ratio U/U*
    ar : array_like
        Critical Area Ratio A/A*
    ma : array_like
        Mach Angle
    pm : array_like
        Prandtl-Meyer Angle
    """

    pr = pressure_ratio.__no_check(M, gamma)
    dr = density_ratio.__no_check(M, gamma)
    tr = temperature_ratio.__no_check(M, gamma)
    prs = critical_pressure_ratio.__no_check(M, gamma)
    drs = critical_density_ratio.__no_check(M, gamma)
    trs = critical_temperature_ratio.__no_check(M, gamma)
    urs = critical_velocity_ratio.__no_check(M, gamma)
    ar = critical_area_ratio.__no_check(M, gamma)
    ma = mach_angle.__no_check(M)
    pm = prandtl_meyer_angle.__no_check(M, gamma)

    return pr, dr, tr, prs, drs, trs, urs, ar, ma, pm
