import numpy as np

from pygasflow.utils.roots import apply_bisection
from pygasflow.utils.decorators import check

# TODO:
# 1. In the functions where the bisection is used, evaluate if the initial mach range
#   for supersonic case is sufficient for all gamma and ratios.
# 2. Evaluate if scipy.brentq and scipy.brenth performs better than scipy.bisect
# 3. Provide tolerance and maxiter arguments to the function where bisect is used.

# NOTE: certain function could very well be computed by using other functions.
#   In doing so, a performance penalty is introduced, that could become significant when
#   computing millions of data points. This is mostly the case with functions that get
#   called by root finding methods.
#   Therefore, I use plain formulas as much as possible.

@check	
def critical_total_temperature_ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Total Temperature Ratio T0/T0*.

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
        Critical Total Temperature Ratio T0/T0*.
    """
    return 2 * (1 + gamma) * M**2 / (1 + gamma * M**2)**2 * (1 + ((gamma - 1) / 2) * M**2)

@check	
def critical_temperature_ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Temperature Ratio T/T*.

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
        Critical Total Temperature Ratio T/T*.
    """
    return M**2 * (1 + gamma)**2 / (1 + gamma * M**2)**2

@check	
def critical_pressure_ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Pressure Ratio P/P*.

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
        Critical Pressure Ratio P/P*.
    """
    return (1 + gamma) / (1 + gamma * M**2)

@check	
def critical_density_ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Density Ratio rho/rho*.

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
        Critical Density Ratio rho/rho*.
    """
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return (1 + gamma * M**2) / ((gamma + 1) * M**2)

@check	
def critical_total_pressure_ratio(M, gamma=1.4):
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
        Critical Total Pressure Ratio P0/P0*.
    """
    return (1 + gamma) / (1 + gamma * M**2) * ((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1 ) / 2))**(gamma / (gamma - 1))

@check	
def critical_velocity_ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Velocity Ratio U/U*.

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
        Critical Velocity Ratio U/U*.
    """
    return (1 + gamma) * M**2 / (1 + gamma * M**2)

@check	
def critical_entropy_parameter(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Entropy parameter (s*-s)/R.

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
        Critical Entropy parameter (s*-s)/R.
    """
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return -gamma /(gamma - 1) * np.log(M**2 * ((gamma + 1) / (1 + gamma * M**2))**((gamma + 1) / gamma))

@check
def m_from_critical_total_temperature_ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Total Temperature Ratio T0/T0*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Total Temperature Ratio T0/T0*. If float, list,
        tuple is given as input, a conversion will be attempted.
        Must be 0 <= T0/T0* < 1.
    flag : string, optional
        Can be either ``'sub'`` (subsonic) or ``'super'`` (supersonic).
        Default to ``'sub'``.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
        out : ndarray
            Mach Number.
    """
    if np.any(ratio < 0) or np.any(ratio > 1):
        raise ValueError("It must be 0 <= T0/T0* <= 1.")

    # When ratio = 1 the values into the sqrt() should be zero,
    # instead it's some very small negative number (rounding/truncation
    # errors), returning NaN instead of 1.
    r = np.ones_like(ratio)
    idx = ratio != 1
    # need to use absolute value on denominator!
    den = np.abs(ratio[idx] * gamma**2 + 1 - gamma**2)
    if flag == "sub":
        r[idx] = np.sqrt(-(ratio[idx] * gamma**2 + 1 - gamma**2) * (ratio[idx] * gamma - 1 - gamma + np.sqrt(-2 * ratio[idx] * gamma - ratio[idx] * gamma**2 + 1 + 2 * gamma + gamma**2 - ratio[idx]))) / den
    else:
        r[idx] = np.sqrt(-(ratio[idx] * gamma**2 + 1 - gamma**2) * (ratio[idx] * gamma - 1 - gamma - np.sqrt(-2 * ratio[idx] * gamma - ratio[idx] * gamma**2 + 1 + 2 * gamma + gamma**2 - ratio[idx]))) / den
    r[ratio == 1] = 1
    return r

@check
def m_from_critical_temperature_ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Temperature Ratio T/T*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Temperature Ratio T/T*. If float, list, tuple is
        given as input, a conversion will be attempted.
        Must be 0 < T/T* < T/T*(M(d(CTR)/dM = 0))
    flag : string, optional
        Can be either:

        * ``'sub'`` for subsonic case.
        * ``'super'`` for supersonic case.

        Default to ``'sub'``.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    # compute the maximum value of the Critical Temperature Ratio.
    # the Mach number corresponding to this CTR has been computed with
    # d(CTR)/dM = 0
    upper_lim = critical_temperature_ratio(1 / np.sqrt(gamma))
    if np.any(ratio < 0) or np.any(ratio > upper_lim):
        raise ValueError("It must be 0 < T/T* < {}.".format(upper_lim))

    M = np.zeros_like(ratio)
    if flag == "sub":
        M[ratio == 0] = 0
        M[ratio != 0] = np.sqrt(-2 * ratio[ratio != 0] * (2 * ratio[ratio != 0] * gamma - 1 - 2 * gamma - gamma**2 + np.sqrt(1 - 4 * ratio[ratio != 0] * gamma - 8 * ratio[ratio != 0] * gamma**2 - 4 * ratio[ratio != 0] * gamma**3 + 4 * gamma + 6 * gamma**2 + 4 * gamma**3 + gamma**4))) / (2 * ratio[ratio != 0] * gamma)
    else:
        M[ratio == 0] = np.inf
        M[ratio != 0] = np.sqrt(-2 * ratio[ratio != 0] * (2 * ratio[ratio != 0] * gamma - 1 - 2 * gamma - gamma**2 - np.sqrt(1 - 4 * ratio[ratio != 0] * gamma - 8 * ratio[ratio != 0] * gamma**2 - 4 * ratio[ratio != 0] * gamma**3 + 4 * gamma + 6 * gamma**2 + 4 * gamma**3 + gamma**4))) / (2 * ratio[ratio != 0] * gamma)
    return M

@check
def m_from_critical_pressure_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Pressure Ratio P/P*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Pressure Ratio P/P*. If float, list, tuple
        is given as input, a conversion will be attempted.
        Must be 0 < P/P* < P/P*(M=0).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    upper_lim = critical_pressure_ratio(0, gamma)
    if np.any(ratio < 0) or np.any(ratio > upper_lim):
        raise ValueError("It must be 0 <= P/P* <= {}.".format(upper_lim))
    M = np.zeros_like(ratio)
    M[ratio == 0] = np.inf
    M[ratio != 0] = np.sqrt(ratio[ratio != 0] * gamma * (1 + gamma - ratio[ratio != 0])) / (ratio[ratio != 0] * gamma)
    return M

@check
def m_from_critical_total_pressure_ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Total Pressure Ratio P0/P0*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Total Pressure Ratio P0/P0*. If float, list, tuple
        is given as input, a conversion will be attempted.
        If ``flag='sub'``, it must be 1 <= P0/P0* < P0/P0*(M=0).
        Else, P0/P0* >= 1.
    flag : string, optional
        Can be either ``'sub'`` (subsonic) or ``'super'`` (supersonic).
        Default to ``'sub'``.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """

    if flag == "sub":
        upper_lim = critical_total_pressure_ratio(0, gamma)
        if np.any(ratio < 1) or np.any(ratio >= upper_lim):
            raise ValueError("It must be 1 <= P0/P0* < {}".format(upper_lim))
    else:
        if np.any(ratio < 1):
            raise ValueError("It must be P0/P0* >= 1")

    # func = lambda M, r: r - Critical_Total_Pressure_Ratio.__no_check(M, gamma)
    func = lambda M, r: r - (1 + gamma) / (1 + gamma * M**2) * ((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1 ) / 2))**(gamma / (gamma - 1))

    return apply_bisection(ratio, func, flag=flag)

@check	
def m_from_critical_density_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Density Ratio rho/rho*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Density Ratio rho/rho*. If float, list, tuple
        is given as input, a conversion will be attempted.
        Must be rho/rho* > gamma / (gamma + 1).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    lower_lim = gamma / (gamma + 1)
    if np.any(ratio <= lower_lim):
        raise ValueError("It must be rho/rho* > {}.".format(lower_lim))
    return np.sqrt(1 / (ratio * (gamma + 1) - gamma))

@check
def m_from_critical_velocity_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Velocity Ratio U/U*.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Velocity Ratio U/U*. If float, list, tuple
        is given as input, a conversion will be attempted.
        Must be 0 < U/U* < (1 + gamma) / gamma.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    upper_lim = (1 + gamma) / gamma
    if np.any(ratio >= upper_lim) or np.any(ratio <= 0):
        raise ValueError("It must be 0 < U/U* < {}.".format(upper_lim))
    return -np.sqrt(-(ratio * gamma - 1 - gamma) * ratio) / (ratio * gamma - 1 - gamma)

@check
def m_from_critical_entropy(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Entropy (s*-s)/R.

    Parameters
    ----------
    ratio : array_like
        Rayleigh's Critical Critical Entropy (s*-s)/R. If float, list, tuple
        is given as input, a conversion will be attempted.
        Must be (s*-s)/R >= 0.
    flag : string, optional
        Can be either ``'sub'`` (subsonic) or ``'super'`` (supersonic). Default to ``'sub'``.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    if np.any(ratio < 0):
        raise ValueError("It must be (s*-s)/R >= 0.")

    # func = lambda M, r: r - Critical_Entropy_Parameter.__no_check(M, gamma)
    func = lambda M, r: r - (-gamma /(gamma - 1) * np.log(M**2 * ((gamma + 1) / (1 + gamma * M**2))**((gamma + 1) / gamma)))

    # TODO: need to adjust the extreme of the range where to apply bisection.
    # If I try M_From_Critical_Entropy(1000) I get:
    # ValueError: f(a) and f(b) must have different signs
    return apply_bisection(ratio, func, flag=flag)

@check
def get_ratios_from_mach(M, gamma):
    """
    Compute all Rayleigh ratios given the Mach number.

    Parameters
    ----------
    M : array_like
        Mach number
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    prs : array_like
        Critical Pressure Ratio P/P*
    drs : array_like
        Critical Density Ratio rho/rho*
    trs : array_like
        Critical Temperature Ratio T/T*
    tprs : array_like
        Critical Total Pressure Ratio P0/P0*
    tprs : array_like
        Critical Total Temperature Ratio T0/T0*
    urs : array_like
        Critical Velocity Ratio U/U*
    eps : array_like
        Critical Entropy Ratio (s*-s)/R
    """
    prs = critical_pressure_ratio.__no_check(M, gamma)
    drs = critical_density_ratio.__no_check(M, gamma)
    trs = critical_temperature_ratio.__no_check(M, gamma)
    tprs = critical_total_pressure_ratio.__no_check(M, gamma)
    ttrs = critical_total_temperature_ratio.__no_check(M, gamma)
    urs = critical_velocity_ratio.__no_check(M, gamma)
    eps = critical_entropy_parameter.__no_check(M, gamma)

    return prs, drs, trs, tprs, ttrs, urs, eps
