import numpy as np
from decorators import check_M_gamma, check_ratio_gamma
from roots import Apply_Bisection

# TODO:
# 1. In the functions where the bisection is used, evaluate if the initial mach range
#   for supersonic case is sufficient for all gamma and ratios.
# 2. Evaluate if scipy.brentq and scipy.brenth performs better than scipy.bisect
# 3. Provide tolerance and maxiter arguments to the function where bisect is used.
# 4. Look at the possibility to change the argument flag="sub" (or "sup") to a Boolean.

# NOTE: certain function could very well be computed by using other functions.
#   In doing so, a performance penalty is introduced, that could become significant when
#   computing millions of data points. This is mostly the case with functions that get
#   called by root finding methods.
#   Therefore, I use plain formulas as much as possible.

@check_M_gamma	
def Critical_Total_Temperature_Ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Total Temperature Ratio T0/T0*.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Total Temperature Ratio T0/T0*.
    """
    return 2 * (1 + gamma) * M**2 / (1 + gamma * M**2)**2 * (1 + ((gamma - 1) / 2) * M**2)

@check_M_gamma	
def Critical_Temperature_Ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Temperature Ratio T/T*.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Total Temperature Ratio T/T*.
    """
    return M**2 * (1 + gamma)**2 / (1 + gamma * M**2)**2

@check_M_gamma	
def Critical_Pressure_Ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Pressure Ratio P/P*.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Pressure Ratio P/P*.
    """
    return (1 + gamma) / (1 + gamma * M**2)

@check_M_gamma	
def Critical_Total_Pressure_Ratio(M, gamma=1.4):
    """
    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Total Pressure Ratio P0/P0*.
    """
    return (1 + gamma) / (1 + gamma * M**2) * ((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1 ) / 2))**(gamma / (gamma - 1))

@check_M_gamma	
def Critical_Velocity_Ratio(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Velocity Ratio U/U*.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Velocity Ratio U/U*.
    """
    return (1 + gamma) * M**2 / (1 + gamma * M**2)

@check_M_gamma	
def Critical_Entropy_Parameter(M, gamma=1.4):
    """
    Compute the Rayleigh's Critical Entropy parameter (s*-s)/R.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Critical Entropy parameter (s*-s)/R.
    """
    return -gamma /(gamma - 1) * np.log(M**2 * ((gamma + 1) / (1 + gamma * M**2))**((gamma + 1) / gamma))

@check_ratio_gamma
def M_From_Critical_Total_Temperature_Ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Total Temperature Ratio T0/T0*.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Total Temperature Ratio T0/T0*. If float, list, tuple is
            given as input, a conversion will be attempted. Must be 0 < T0/T0* < 1.
        flag : string
            Can be either 'sub' (subsonic) or 'sup' (supersonic). Default to 'sub'.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio > 0) and np.all(ratio < 1), "It must be 0 < T0/T0* < 1."
    flag = flag.lower()
    assert flag in ["sub", "sup"], "Flag can be either 'sub' or 'sup'."
    if flag == "sub":
        return np.sqrt(-(ratio * gamma**2 + 1 - gamma**2) * (ratio * gamma - 1 - gamma + np.sqrt(-2 * ratio * gamma - ratio * gamma**2 + 1 + 2 * gamma + gamma**2 - ratio))) / (ratio * gamma**2 + 1 - gamma**2)
    else:
        return np.sqrt(-(ratio * gamma**2 + 1 - gamma**2) * (ratio * gamma - 1 - gamma - np.sqrt(-2 * ratio * gamma - ratio * gamma**2 + 1 + 2 * gamma + gamma**2 - ratio))) / (ratio * gamma**2 + 1 - gamma**2)

@check_ratio_gamma
def M_From_Critical_Temperature_Ratio(ratio, flag="below", gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Temperature Ratio T/T*.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Temperature Ratio T/T*. If float, list, tuple is
            given as input, a conversion will be attempted. Must be 0 < T/T* < 1.03.
        flag : string
            Can be either 'below' (below Tmax) or 'above' (Above Tmax). Default to 'sub'.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio > 0) and np.all(ratio < 1.03), "It must be 0 < T/T* < 1.03."
    flag = flag.lower()
    assert flag in ["below", "above"], "Flag can be either 'below' or 'above'."
    if flag == "below":
        return np.sqrt(-2 * ratio * (2 * ratio * gamma - 1 - 2 * gamma - gamma**2 + np.sqrt(1 - 4 * ratio * gamma - 8 * ratio * gamma**2 - 4 * ratio * gamma**3 + 4 * gamma + 6 * gamma**2 + 4 * gamma**3 + gamma**4))) / (2 * ratio * gamma)
    else:
        return np.sqrt(-2 * ratio * (2 * ratio * gamma - 1 - 2 * gamma - gamma**2 - np.sqrt(1 - 4 * ratio * gamma - 8 * ratio * gamma**2 - 4 * ratio * gamma**3 + 4 * gamma + 6 * gamma**2 + 4 * gamma**3 + gamma**4))) / (2 * ratio * gamma)

@check_ratio_gamma
def M_From_Critical_Pressure_Ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Pressure Ratio P/P*.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Pressure Ratio P/P*. If float, list, tuple 
            is given as input, a conversion will be attempted. Must be 0 < P/P* < 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio <= 2.4), "It must be 0 < P/P* < 1."
    return np.sqrt(ratio * gamma * (1 + gamma - ratio)) / (ratio * gamma)

# @execution_time
@check_ratio_gamma
def M_From_Critical_Total_Pressure_Ratio(ratio, flag="sub", gamma=1.4, tol=1e-08):
    """
    Compute the Mach number given Rayleigh's Critical Total Pressure Ratio P0/P0*.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Total Pressure Ratio P0/P0*. If float, list, tuple 
            is given as input, a conversion will be attempted. Must be 1 < P0/P0* < P0/P0*(M=0).
        flag : string
            Can be either 'sub' (subsonic) or 'sup' (supersonic). Default to 'sub'.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    upper_lim = (gamma + 1) * (2 / (gamma + 1))**(gamma / (gamma - 1))
    assert np.all(ratio > 1) and np.all(ratio < upper_lim), "It must be 1 < P0/P0* < P0/P0*(M=0)={}".format(upper_lim)
    flag = flag.lower()
    assert flag in ["sub", "sup"], "Flag can be either 'sub' or 'sup'."

    # func = lambda M, r: r - Critical_Total_Pressure_Ratio.__bypass_decorator(M, gamma)
    func = lambda M, r: r - (1 + gamma) / (1 + gamma * M**2) * ((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1 ) / 2))**(gamma / (gamma - 1))

    return Apply_Bisection(ratio, func, flag=flag)

@check_ratio_gamma
def M_From_Critical_Velocity_Ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given Rayleigh's Critical Velocity Ratio U/U*.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Velocity Ratio U/U*. If float, list, tuple 
            is given as input, a conversion will be attempted. Must be 0 < U/U* < 1.7144.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio <= 1.7144), "It must be 0 < U/U* < 1.7144."
    return -np.sqrt(-(ratio * gamma - 1 - gamma) * ratio) / (ratio * gamma - 1 - gamma)

# @execution_time
@check_ratio_gamma
def M_From_Critical_Entropy(ratio, flag="sub", gamma=1.4, tol=1e-08):
    """
    Compute the Mach number given Rayleigh's Critical Entropy (s*-s)/R.

    Parameters
    ----------
        ratio : array_like
            Rayleigh's Critical Critical Entropy (s*-s)/R. If float, list, tuple 
            is given as input, a conversion will be attempted. Must be (s*-s)/R >= 0.
        flag : string
            Can be either 'sub' (subsonic) or 'sup' (supersonic). Default to 'sub'.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio >= 0), "It must be (s*-s)/R >= 0."
    flag = flag.lower()
    assert flag in ["sub", "sup"], "Flag can be either 'sub' or 'sup'."

    # func = lambda M, r: r - Critical_Entropy_Parameter.__bypass_decorator(M, gamma)
    func = lambda M, r: r - (-gamma /(gamma - 1) * np.log(M**2 * ((gamma + 1) / (1 + gamma * M**2))**((gamma + 1) / gamma)))

    return Apply_Bisection(ratio, func, flag=flag)
