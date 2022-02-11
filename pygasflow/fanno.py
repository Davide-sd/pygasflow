import numpy as np

from pygasflow.utils.roots import apply_bisection
from pygasflow.utils.decorators import check

# TODO:
# PRIORITY: take a look at the TODO in M_From_Critical_Friction.
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
def critical_temperature_ratio(M, gamma=1.4):
    """
    Compute the Fanno's Critical Temperature Ratio T/T*.

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
        Critical Temperature Ratio T/T*.
    """
    return ((gamma + 1) / 2) / (1 + ((gamma - 1) / 2) * M**2)

@check
def critical_pressure_ratio(M, gamma=1.4):
    """
    Compute the Fanno's Critical Pressure Ratio P/P*.

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
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return (1 / M) * np.sqrt(((gamma + 1) / 2) / (1 + ((gamma - 1) / 2) * M**2))

@check
def critical_density_ratio(M, gamma=1.4):
    """
    Compute the Fanno's Critical Density Ratio rho/rho*.

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
    return (1 / M) * np.sqrt((2 + (gamma - 1) * M**2) / (gamma + 1))

@check	
def critical_total_pressure_ratio(M, gamma=1.4):
    """
    Compute the Fanno's Critical Total Pressure Ratio P0/P0*.

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
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return (1 / M) * ((1 + ((gamma - 1) / 2) * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))

@check
def critical_velocity_ratio(M, gamma=1.4):
    """
    Compute the Fanno's Critical Velocity Ratio U/U*.

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
    return M * np.sqrt(((gamma + 1) / 2) / (1 + ((gamma - 1) / 2) * M**2))

# Fanno's Maximum Limit Friction Parameter (4 f Lmax / D)
@check
def critical_friction_parameter(M, gamma=1.4):
    """
    Compute the Fanno's Critical Friction Parameter 4fL*/D.

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
        Critical Friction Parameter 4fL*/D.
    """
    # need to deal with division by 0
    f = np.zeros_like(M)
    f[M == 0] = np.inf
    f[M != 0] = ((gamma + 1) / (2 * gamma)) * np.log(((gamma + 1) / 2) * M[M != 0]**2 / (1 + ((gamma - 1) / 2) * M[M != 0]**2)) + (1 / gamma) * (1 / M[M != 0]**2 - 1)
    return f

@check		
def critical_entropy_parameter(M, gamma=1.4):
    """
    Compute the Fanno's Critical Entropy Parameter (s*-s)/R.

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
        Critical Entropy Parameter (s*-s)/R.
    """
    # TODO: here, division by M=0 produce the correct results, infinity.
    # Do I need to suppress the warning???
    return np.log((1 / M) * ((1 + ((gamma - 1) / 2) * M**2) / (1 + ((gamma - 1) / 2)))**((gamma + 1) / (2 * (gamma - 1))))

@check
def m_from_critical_temperature_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Temperature Ratio T/T*.

    Parameters
    ----------
    ratio : array_like
        Fanno's Critical Temperature Ratio T/T*. If float, list, tuple is
        given as input, a conversion will be attempted.
        Must be 0 < T/T* < Critical_Temperature_Ratio(0, gamma).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    upper_lim = critical_temperature_ratio(0, gamma)
    if np.any(ratio <= 0) or np.any(ratio >= upper_lim):
        raise ValueError("It must be 0 < T/T* < {}.".format(upper_lim))
    return np.sqrt(-ratio * (gamma - 1) * (2 * ratio - gamma - 1)) / (ratio * gamma - ratio)

@check
def m_from_critical_pressure_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Pressure Ratio P/P*.

    Parameters
    ----------
    ratio : array_like
        Fanno's Critical Pressure Ratio P/P*. If float, list, tuple is given
        as input, a conversion will be attempted.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    return np.sqrt(-ratio * (gamma - 1) * (ratio - np.sqrt(ratio**2 + gamma**2 - 1))) / (ratio * gamma - ratio)

@check
def m_from_critical_density_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Density Ratio rho/rho*.

    Parameters
    ----------
    ratio : array_like
        Fanno's Critical Density Ratio rho/rho*. If float, list, tuple is
        given as input, a conversion will be attempted.
        Must be: rho/rho* > np.sqrt((gamma - 1) / (gamma + 1))
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    lower_lim = np.sqrt((gamma - 1) / (gamma + 1))
    if np.any(ratio < lower_lim):
        raise ValueError(
            "The critical density ratio must be >= {}.".format(lower_lim))
    return np.sqrt(2 / ((gamma + 1) * ratio**2 - (gamma - 1)))


@check
def m_from_critical_total_pressure_ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Total Pressure Ratio P0/P0*.

    Parameters
    ----------
    ratio : array_like
        Fanno's Critical Total Pressure Ratio P0/P0*. If float, list, tuple
        is given as input, a conversion will be attempted. Must be P0/P0* > 1.
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
    if np.any(ratio < 1):
        raise ValueError("It must be P/P* > 1.")

    # func = lambda M, r: r - Critical_Total_Pressure_Ratio.__no_check(M, gamma)
    func = lambda M, r: r - (1 / M) * ((1 + ((gamma - 1) / 2) * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))

    return apply_bisection(ratio, func, flag=flag)

@check
def m_from_critical_velocity_ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Velocity Ratio U/U*.

    Parameters
    ----------
    ratio : array_like
        Fanno's Critical Velocity Ratio U/U*. If float, list, tuple is given
        as input, a conversion will be attempted.
        Must be 0 <= U/U* < (1 / np.sqrt((gamma - 1) / (gamma + 1))).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : ndarray
        Mach Number.
    """
    # Critical Velocity Ratio and Critical Density Ratio are reciprocal
    # of each other. Therefore, I can use the lower limit of CDR to
    # compute the upper limit of CVR
    lower_lim = np.sqrt((gamma - 1) / (gamma + 1))
    upper_lim = 1 / lower_lim
    if np.any(ratio < 0) or np.any(ratio >= upper_lim):
        raise ValueError("It must be 0 <= U/U* < {}.".format(upper_lim))
    return 2 * ratio / np.sqrt(2 * gamma + 2 - 2 * ratio**2 * gamma + 2 * ratio**2)

@check
def m_from_critical_friction(fp, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Friction Parameter 4fL*/D.

    Parameters
    ----------
    fp : array_like
        Fanno's Critical Friction Parameter 4fL*/D. If float, list, tuple
        is given as input, a conversion will be attempted.
        If flag="sub", it must be fp >= 0.
        Else, 0 <= fp <= ((gamma + 1) * np.log((gamma + 1) / (gamma - 1)) - 2) / (2 * gamma)
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
        if np.any(fp < 0):
            raise ValueError('It must be fp >= 0.')
    else:
        upper_lim = ((gamma + 1) * np.log((gamma + 1) / (gamma - 1)) - 2) / (2 * gamma)
        if np.any(fp < 0) or np.any(fp > upper_lim):
            raise ValueError('It must be 0 <= fp <= {}'.format(upper_lim))

    # TODO: when solving the supersonic case, and ratio -> upper limit,
    # I get: ValueError: f(a) and f(b) must have different signs
    # need to be dealt with!

    # func = lambda M, r: r - Critical_Friction_Parameter.__no_check(M, gamma)
    func = lambda M, r: r - (((gamma + 1) / (2 * gamma)) * np.log(((gamma + 1) / 2) * M**2 / (1 + ((gamma - 1) / 2) * M**2)) + (1 / gamma) * (1 / M**2 - 1))

    return apply_bisection(fp, func, flag=flag)

@check
def m_from_critical_entropy(ep, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Fanno's Critical Entropy Parameter (s*-s)/R.

    Parameters
    ----------
    ep : array_like
        Fanno's Critical Entropy Parameter (s*-s)/R. If float, list, tuple
        is given as input, a conversion will be attempted.
        Must be (s*-s)/R >= 0.
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
    if np.any(ep < 0):
        raise ValueError("It must be (s* - s) / R >= 0.")

    # func = lambda M, r: r - Critical_Entropy_Parameter.__no_check(M, gamma)
    func = lambda M, r: r - np.log((1 / M) * ((1 + ((gamma - 1) / 2) * M**2) / (1 + ((gamma - 1) / 2)))**((gamma + 1) / (2 * (gamma - 1))))

    return apply_bisection(ep, func, flag=flag)

@check
def get_ratios_from_mach(M, gamma):
    """
    Compute all fanno ratios given the Mach number.

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
    urs : array_like
        Critical Velocity Ratio U/U*
    fps : array_like
        Critical Friction Parameter 4fL*/D
    eps : array_like
        Critical Entropy Ratio (s*-s)/R
    """
    prs = critical_pressure_ratio.__no_check(M, gamma)
    drs = critical_density_ratio.__no_check(M, gamma)
    trs = critical_temperature_ratio.__no_check(M, gamma)
    tprs = critical_total_pressure_ratio.__no_check(M, gamma)
    urs = critical_velocity_ratio.__no_check(M, gamma)
    fps = critical_friction_parameter.__no_check(M, gamma)
    eps = critical_entropy_parameter.__no_check(M, gamma)

    return prs, drs, trs, tprs, urs, fps, eps
