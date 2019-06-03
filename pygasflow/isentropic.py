import numpy as np
from pygasflow.utils.decorators import check_M_gamma, check_ratio_gamma, do_nothing, convert_first_argument
from pygasflow.utils.roots import Apply_Bisection
from scipy.optimize import bisect

# TODO:
# 1. In the functions where the bisection is used, evaluate if the initial mach range
#   for supersonic case is sufficient for all gamma and ratios.
# 2. Evaluate if scipy.brentq and scipy.brenth performs better than scipy.bisect
# 3. Provide tolerance and maxiter arguments to the function where bisect is used.
# 4. Look at the possibility to change the argument flag="sub" (or "sup") to a Boolean.

# NOTE: certain function could very well be computed by using other functions.
#   For instance, Critical_Temperature_Ratio could be computed using Temperature_Ratio.
#   In doing so, a performance penalty is introduced, that could become significant when
#   computing millions of data points. This is mostly the case with functions that get
#   called by root finding methods.
#   Therefore, I use plain formulas as much as possible.

@check_M_gamma
def Critical_Velocity_Ratio(M, gamma=1.4):
    """
    Compute the critical velocity ratio U/U*. 

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
            Critical velocity ratio U/U*. 
    """
    return 1 / ((2 / (gamma + 1))**(1 / (gamma - 1)) / M * (1 + (gamma - 1) / 2 * M**2)**0.5)

@check_M_gamma
def Critical_Temperature_Ratio(M, gamma=1.4):
    """
    Compute the critical temperature ratio T/T*. 

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
            Critical Temperature ratio T/T*. 
    """
    # return Temperature_Ratio.__bypass_decorator(M, gamma) * 0.5 * (gamma + 1)
    return ((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2)

@check_M_gamma
def Critical_Pressure_Ratio(M, gamma=1.4):
    """
    Compute the critical pressure ratio P/P*. 

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
            Critical Pressure ratio P/P*. 
    """
    # return Pressure_Ratio.__bypass_decorator(M, gamma) * (0.5 * (gamma + 1))**(gamma / (gamma - 1))
    return (((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2))**(gamma / (gamma - 1))

@check_M_gamma
def Critical_Density_Ratio(M, gamma=1.4):
    """
    Compute the critical density ratio rho*/rho. 

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
            Critical density ratio rho*/rho. 
    """
    # this first version appears to be faster!
    return Density_Ratio.__bypass_decorator(M, gamma) * (0.5 * (gamma + 1))**(1 / (gamma - 1))
    # return (((gamma + 1) / 2) / (1 + (gamma - 1) / 2 * M**2))**(1 / (gamma - 1))

@check_M_gamma
def Critical_Area_Ratio(M, gamma=1.4):
    """
    Compute the critical area ratio A/A*. 

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
            Critical area ratio A/A*. 
    """
    # return 1  / Critical_Density_Ratio.__bypass_decorator(M, gamma) * np.sqrt(1 / Critical_Temperature_Ratio.__bypass_decorator(M, gamma)) / M
    return (((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))) / M


@check_M_gamma
def Pressure_Ratio(M, gamma=1.4):
    """
    Compute the pressure ratio P/P0. 

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
            Pressure ratio P/P0. 
    """
    return (1 + (gamma - 1) / 2 * M**2)**(-gamma / (gamma - 1))

@check_M_gamma
def Temperature_Ratio(M, gamma=1.4):
    """
    Compute the temperature ratio T/T0. 

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
            Temperature ratio T/T0. 
    """
    return 1 / (1 + (gamma - 1) / 2 * M**2)

@check_M_gamma
def Density_Ratio(M, gamma=1.4):
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
            Density ratio rho/rho0. 
    """
    return (1 + (gamma - 1) / 2 * M**2)**(-1 / (gamma - 1))

@convert_first_argument
def Sound_Speed(T, R=287.058, gamma=1.4):
    """
    Compute the sound speed.

    Parameters
    ----------
        T : array_like
            Temperature. If float, list, tuple is given as input, a conversion
            will be attempted. Must be T >= 0.
        R : float
            Specific Gas Constant. Default is air, R=287.058
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Sound Speed. 
    """
    assert np.all(T >= 0), "Temperature must be >= 0."
    assert R > 0, "Specific gas constant must be >= 0."
    assert gamma > 1, "Specific heats ratio must be > 1."
    return np.sqrt(gamma * R * T)

# TODO: no check is done over a.
@do_nothing
def Mach_Number(U, a):
    """
    Compute the Mach number.

    Parameters
    ----------
        U : array_like
            Velocity. If float, list, tuple is given as input, a conversion
            will be attempted. Must be U >= 0.
        a : array_like
            Sound Speed. If float, list, tuple is given as input, a conversion
            will be attempted. If array_like, must be U.shape == a.shape.
    
    Returns
    -------
        out : ndarray
            Mach Number. 
    """
    U = np.asarray(U)
    a = np.asarray(a)
    if a.size > 1:
        assert U.shape == a.shape, "U and a must have the same shape."
    return U / a

@check_ratio_gamma
def M_From_Temperature_Ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Temperature Ratio T/T0.

    Parameters
    ----------
        ratio : array_like
            Isentropic Temperature Ratio T/T0. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= T/T0 <= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio >= 0) and np.all(ratio <= 1), "Temperature ratio must be 0 <= T/T0 <= 1."
    return np.sqrt(2 * (1 / ratio - 1) / (gamma - 1))

@check_ratio_gamma
def M_From_Pressure_Ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Pressure Ratio P/P0.

    Parameters
    ----------
        ratio : array_like
            Isentropic Pressure Ratio P/P0. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= P/P0 <= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio >= 0) and np.all(ratio <= 1), "Pressure ratio must be 0 <= P/P0 <= 1."
    return np.sqrt(2 / (gamma - 1) * (1 / ratio**((gamma - 1) / gamma) - 1))

@check_ratio_gamma
def M_From_Density_Ratio(ratio, gamma=1.4):
    """
    Compute the Mach number given the Isentropic Density Ratio rho/rho0.

    Parameters
    ----------
        ratio : array_like
            Isentropic Density Ratio rho/rho0. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= rho/rho0 <= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio >= 0) and np.all(ratio <= 1), "Density ratio must be 0 <= rho/rho0 <= 1."
    return np.sqrt(2 / (gamma - 1) * (1 / ratio**(gamma - 1) - 1))

@check_ratio_gamma
def M_From_Critical_Area_Ratio(ratio, flag="sub", gamma=1.4):
    """
    Compute the Mach number given the Isentropic Critical Area Ratio A/A*.

    Parameters
    ----------
        ratio : array_like
            Isentropic Critical Area Ratio A/A*. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be A/A* >= 1.
        flag : string
            Can be either 'sub' (subsonic) or 'sup' (supersonic). Default to 'sub'.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(ratio >= 1), "Area ratio must be A/A* >= 1."
    flag = flag.lower()
    assert flag in ["sub", "sup"], "flag can be either 'sub' or 'sup'."
    
    func = lambda M, r: r - (((1 + (gamma - 1) / 2 * M**2) / ((gamma + 1) / 2))**((gamma + 1) / (2 * (gamma - 1)))) / M
    # func = lambda M, r: r - Critical_Area_Ratio.__bypass_decorator(M, gamma)
    return Apply_Bisection(ratio, func, flag=flag)


@check_ratio_gamma
def M_From_Critical_Area_Ratio_And_Pressure_Ratio(a_ratio, p_ratio, gamma=1.4):
    """
    Compute the Mach number given the Critical Area Ratio (A/A*) and
    the Pressure Ratio (P/P0).

    Parameters
    ----------
        a_ratio : array_like
            Isentropic Critical Area Ratio A/A*. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be A/A* >= 1.
        p_ratio : array_like
            Isentropic Pressure Ratio (P/P0). If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= P/P0 <= 1.
            If array_like, must be a_ratio.shape == p_ratio.shape.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    p_ratio = np.asarray(p_ratio)
    assert np.all(a_ratio >= 1), "Area ratio must be A/A* >= 1."
    assert np.all(p_ratio >= 0) and np.all(p_ratio <= 1), "Pressure ratio must be 0 <= P/P0 <= 1."
    assert a_ratio.shape == p_ratio.shape, "The Critical Area Ratio and Pressure Ratio must have the same number of elements and the same shape."
    # eq. 5.28, Modern Compressible Flow, 3rd Edition, John D. Anderson
    return np.sqrt(-1 / (gamma - 1) + np.sqrt(1 / (gamma - 1)**2 + 2 / (gamma - 1) * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)) / a_ratio**2 / p_ratio**2))

@check_ratio_gamma
def M_From_Mach_Angle(angle, gamma=1.4):
    """
    Compute the Mach number given the Mach Angle.

    Parameters
    ----------
        ratio : array_like
            Mach Angle [degrees]. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= Mach Angle <= 90.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Mach Number.
    """
    assert np.all(angle >= 0) and np.all(angle <= 90), "Mach angle must be between 0° and 90°."
    return 1 / np.sin(np.deg2rad(angle))

@check_M_gamma
def Mach_Angle(M, gamma=1.4):
    """
    Compute the Mach angle given the Mach number.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
            TODO, NOTE: in this function, gamma is not used!!!!
    
    Returns
    -------
        out : ndarray
            Critical area Mach_Angle [degrees].
    """
    # here I use dtype=np.float to be able to assign np.nan where necessary (np.nan is an IEEE 754 floating point representation of Not a Number)
    angle = np.zeros_like(M, dtype=np.float)
    angle[M > 1] = np.rad2deg(np.arcsin(1 / M[M > 1]))
    angle[M == 1] = 90
    angle[M < 1] = np.nan
    return angle

@convert_first_argument
def M_From_Prandtl_Meyer_Angle(angle, gamma=1.4):
    """
    Compute the Mach number given the Prandtl Meyer angle.

    Parameters
    ----------
        angle : array_like
            Prandtl Meyer angle [degrees].
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M : array_like
            Mach Number.
    """
    nu_max = (np.sqrt((gamma + 1) / (gamma - 1)) - 1) * 90
    assert np.all(angle >= 0) and np.all(angle <= nu_max), "Prandtl-Meyer angle must be between 0 and {}".format(nu_max)
    angle = np.deg2rad(angle)

    func = lambda M, a: a - (np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1)))

    return Apply_Bisection(angle, func, flag="sup")

@check_M_gamma
def Prandtl_Meyer_Angle(M, gamma=1.4):
    """
    Compute the Prandtl Meyer function given the Mach number.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        nu : array_like
            Prandtl Meyer angle [degrees].
    """
    # here I use dtype=np.float to be able to assign np.nan where necessary (np.nan is an IEEE 754 floating point representation of Not a Number)
    nu = np.zeros_like(M, dtype=np.float)
    # Equation (4.44), Anderson
    nu[M > 1] = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M[M > 1]**2 - 1)))
    nu[M > 1] = nu[M > 1] - np.arctan(np.sqrt(M[M > 1]**2 - 1))
    nu[M > 1] = np.rad2deg(nu[M > 1])
    nu[M == 1] = 0
    nu[M < 1] = np.nan
    return nu

@check_M_gamma
def Get_Ratios_From_Mach(M, gamma):
    """
    Compute all isentropic ratios given the Mach number.

    Parameters
    ----------
        M : array_like
            Mach number
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
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

    pr = Pressure_Ratio.__bypass_decorator(M, gamma)
    dr = Density_Ratio.__bypass_decorator(M, gamma)
    tr = Temperature_Ratio.__bypass_decorator(M, gamma)
    prs = Critical_Pressure_Ratio.__bypass_decorator(M, gamma)
    drs = Critical_Density_Ratio.__bypass_decorator(M, gamma)
    trs = Critical_Temperature_Ratio.__bypass_decorator(M, gamma)
    urs = Critical_Velocity_Ratio.__bypass_decorator(M, gamma)
    ar = Critical_Area_Ratio.__bypass_decorator(M, gamma)
    ma = Mach_Angle.__bypass_decorator(M, gamma)
    pm = Prandtl_Meyer_Angle.__bypass_decorator(M, gamma)

    return pr, dr, tr, prs, drs, trs, urs, ar, ma, pm