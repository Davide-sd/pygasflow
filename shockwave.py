import numpy as np
from numbers import Number
from decorators import convert_first_argument, check_M_gamma, check_M_gamma_shockwave, check_ratio_gamma
from roots import Apply_Bisection

#####################################################################################
############# The following methods are specific for normal shock waves. ############
##### They can also be used for calculation with oblique shock wave, just use #######
##################### the normal component of the Mach number. ######################
#####################################################################################

@check_M_gamma_shockwave
def Pressure_Ratio(M1, gamma=1.4):
    """Compute the static pressure ratio P2/P1. 

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Pressure Ratio P2/P1
    """
    return (2 * gamma * M1**2 - gamma + 1) / (gamma + 1)

@check_M_gamma_shockwave
def Temperature_Ratio(M1, gamma=1.4):
    """ Compute the static temperature ratio T2/T1.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Temperature Ratio T2/T1
    """
    return ((2 * gamma * M1**2 - gamma + 1) * (2 + (gamma - 1) * M1**2)
            / ((gamma + 1)**2 * M1**2))

@check_M_gamma_shockwave
def Density_Ratio(M1, gamma=1.4):
    """ Compute the density ratio rho2/rho1.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Density Ratio rho2/rho1
    """
    return ((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2)

@check_M_gamma_shockwave
def Total_Pressure_Ratio(M1, gamma=1.4):
    """ Compute the total pressure ratio P02/P01.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Total Pressure Ratio p02/p01
    """
    a = (gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2)
    b = (gamma + 1) / (2 * gamma * M1**2 - gamma + 1)
    return a**(gamma / (gamma - 1)) * b**(1 / (gamma - 1))

@check_M_gamma_shockwave
def Total_Temperature_Ratio(M1, gamma=1.4):
    """ Compute the total temperature ratio T02/T01.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Total Temperature Ratio T02/T01 (spoiler: always equal to 1 :P )
    """
    return np.ones_like(M1)

# here I use check_M_gamma and not check_M_gamma_shockwave because this function
# can also be used to compute M1 given M2 (which is M2 < 1)
@check_M_gamma
def M2(M1, gamma=1.4):
    """ Compute the downstream Mach number M2.
    Note that this function can also be used to compute M1 given M2.
    
    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Because this function can be used to compute M1 given
            M2, it will not perform a check wheter M1 >= 1. Be careful on your use!
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Downstream Mach Number M2
    """
    return ((1 + (gamma - 1) / 2 * M1**2) / (gamma * M1**2 -
            (gamma - 1) / 2))**(0.5)

@check_ratio_gamma
def M1_From_Pressure_Ratio(ratio, gamma=1.4):
    """ Compute M1 from the pressure ratio. 

    Parameters
    ----------
        ratio : array_like
            Pressure Ratio P2/P1. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be P2/P1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    assert np.all(ratio >= 1), ("The pressure ratio must be P2/P1 >= 1")

    return np.sqrt((ratio * (gamma + 1) + gamma - 1) / (2 * gamma))

@check_ratio_gamma
def M1_From_Temperature_Ratio(ratio, gamma=1.4):
    """ Compute M1 from the temperature ratio. 

    Parameters
    ----------
        ratio : array_like
            Temperature Ratio T2/T1. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be T2/T1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    assert np.all(ratio >= 1), ("The temperature ratio must be T2/T1 >= 1")

    a = 2.0 * gamma * (gamma - 1)
    b = 4.0 * gamma - (gamma - 1)**2 - ratio * (gamma + 1)**2
    c = -2.0 * (gamma - 1)
    return np.sqrt((-b + np.sqrt(b**2 - 4.0 * a * c)) / 2.0 / a)

@check_ratio_gamma
def M1_From_Density_Ratio(ratio, gamma=1.4):
    """ Compute M1 from the density ratio. 

    Parameters
    ----------
        ratio : array_like
            Density Ratio rho2/rho1. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 1 <= rho2/rho1 < (gamma + 1) / (gamma - 1).
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    gr = (gamma + 1) / (gamma - 1)
    assert np.all(ratio >= 1) and np.all(ratio <= gr), ("The " +
            "density ratio must be 1 < rho2/rho1 < " + str(gr))

    return np.sqrt(2.0 * ratio / (gamma + 1 - ratio * (gamma - 1)))

@check_ratio_gamma
def M1_From_Total_Pressure_Ratio(ratio, gamma=1.4, tol=1e-5):
    """ Compute M1 from the total pressure ratio. 

    Parameters
    ----------
        ratio : array_like
            Total Pressure Ratio. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= P02/P01 <= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    assert np.all(ratio >= 0) and np.all(ratio <= 1), ("The total pressure ratio must be " +
            "0 <= P02/P01 <= 1")

    func = lambda M1, r: r - ((gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2))**(gamma / (gamma - 1)) * ((gamma + 1) / (2 * gamma * M1**2 - gamma + 1))**(1 / (gamma - 1))

    return Apply_Bisection(ratio, func, "sup")

@convert_first_argument
def M1_From_M2(M_2, gamma=1.4):
    """ Compute M1 from the downstream Mach number M2. 

    Parameters
    ----------
        M_2 : array_like
            Downstream Mach Number. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be ((gamma - 1) / 2 / gamma) < M_2 < 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    gr = (gamma - 1) / 2 / gamma
    assert np.all(M_2 >= gr) and np.all(M_2 < 1), ("The downstream M2 must be " +
            str(gr) + " < M2 < 1")

    return M2(M_2, gamma)

#######################################################################################
############## The following methods are specific for oblique shock waves #############
#######################################################################################

@check_M_gamma_shockwave
def Theta_From_Mach_Beta(M1, beta, gamma=1.4):
    """ Compute the flow turning angle Theta accordingly to the input
    parameters.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        beta : float
            Shock wave angle in degrees.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Flow angle Theta [degrees]
    """
    beta = np.deg2rad(beta)

    theta = np.arctan((M1**2 * np.sin(2 * beta) - 2 / np.tan(beta)) / \
        (2 + M1**2 * (gamma + np.cos(2 * beta))))

    return np.rad2deg(theta)

@convert_first_argument
def Beta_From_Mach_Theta(M1, theta, gamma=1.4):
    """ Compute the shock angle Beta accordingly to the input parameters.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        theta : float
            Flow turning angle in degrees.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Dictionary of Shock angle beta [degrees] if it exists, else NaN: 
            {"weak": beta_weak, "strong": beta_strong}.
    """

    # This code has been adapted from: http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
    theta = np.deg2rad(theta)

    p = -(M1**2 + 2) / M1**2 - gamma * np.sin(theta) * np.sin(theta)
    q = (2 * M1**2 + 1) / M1**4 + ((gamma + 1)**2 / 4 + \
        (gamma - 1) / M1**2) * np.sin(theta)**2
    r= -np.cos(theta)**2 / M1**4

    a = (3 * q - p**2) / 3
    b = (2 * p**3 - 9 * p * q + 27 * r) / 27

    test = b**2 / 4 + a**3 / 27

    def compute_beta(_t, _a, _b, _p):
        """Return beta_weak, beta_strong"""
        if _t > 0:
            # return None, None
            # need to
            # return -1, -1
            return np.nan, np.nan
        else:
            if _t == 0:
                x1 = np.sqrt(-_a / 3)
                x2 = x1
                x3 = 2 * x1
                if _b > 0:
                    x1 *= -1
                    x2 *= -1
                    x3 *= -1
            if _t < 0:
                phi = np.arccos(np.sqrt(-27 * _b**2 / 4 / _a**3))
                x1 = 2 * np.sqrt(-_a / 3) * np.cos(phi / 3)
                x2 = 2 * np.sqrt(-_a / 3) * np.cos(phi / 3 + np.pi * 2 / 3)
                x3 = 2 * np.sqrt(-_a / 3) * np.cos(phi / 3 + np.pi * 4 / 3)
                if _b > 0:
                    x1 *= -1
                    x2 *= -1
                    x3 *= -1

            s1 = x1 - _p / 3
            s2 = x2 - _p / 3
            s3 = x3 - _p / 3

            if (s1 < s2 and s1 < s3):
                t1 = s2
                t2 = s3
            elif (s2 < s1 and s2 < s3):
                t1 = s1
                t2 = s3
            else:
                t1 = s1
                t2 = s2

            b1 = np.arcsin(np.sqrt(t1))
            b2 = np.arcsin(np.sqrt(t2))

            if b2 > b1:
                return b1, b2
            return b2, b1
    
    beta_weak, beta_strong = np.zeros_like(M1), np.zeros_like(M1)

    if M1.size == 1:    # scalar case
        beta_weak, beta_strong = compute_beta(test, a, b, p)
    else:
        for i, _t in np.ndenumerate(test):
            beta_weak[i], beta_strong[i] = compute_beta(_t, a[i], b[i], p[i])

    return { "weak": np.rad2deg(beta_weak), "strong": np.rad2deg(beta_strong) }

@convert_first_argument
def Beta_From_Upstream_Mach(M1, MN1):
    """ Compute the shock wave angle beta from the upstream Mach number and
    its normal component.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        MN1 : array_like
            Normal Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be MN1.shape == M1.shape.
    
    Returns
    -------
        out : ndarray
            Shock angle Beta [degrees]
    """
    MN1 = np.asarray(MN1)
    assert np.all(M1 >= 1), "The upstream Mach number must be > 1."
    assert M1.shape == MN1.shape, "M1 and MN1 must have the same number of elements and the same shape."
    return np.rad2deg(np.arcsin(MN1 / M1))

@convert_first_argument
def Normal_Mach_Upstream(M1, beta=None, theta=None, flag="weak"):
    """ Compute the upstream normal Mach Number, which can then be used
    to evaluate all other ratios.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        beta : float
            The shock wave angle in degrees. If beta=None you must give in theta.
        theta : float
            The flow deflection angle in degrees. If theta=None you must give in beta.
        flag : string
            Can be either 'weak' or 'strong'. Default to 'weak'. Chose what value to 
            compute if theta is provided.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Normal Mach number upstream of the shock wave.
            If theta is given, and flag="both" it returns a dictionary of Normal Mach
            numbers: {"weak":weak_MN1, "strong":strong_MN1}.
    """
    assert np.all(M1 >= 1), "The upstream Mach number must be M1 >= 1."
    assert beta != None or theta != None, ("To compute the normal " +
        "component of the upstream Mach number, you have to provide " +
        "either theta or beta.")
    flag = flag.lower()
    assert flag in ["weak", "strong", "both"], "Flag must be either 'weak' or 'strong' or 'both'."

    MN1 = -1
    if beta != None:
        beta = np.deg2rad(beta)
        MN1 = M1 * np.sin(beta)
    elif theta != None:
        beta = Beta_From_Mach_Theta(M1, theta)
        MN1 = dict()
        for k,v in beta.items():
            beta[k] = np.deg2rad(v)
            MN1[k] = M1 * np.sin(beta[k])
        if flag != "both":
            MN1 = MN1[flag]

    return MN1


def Get_M1_From_Ratio(ratioName, ratio, gamma=1.4):
    """ Compute the upstream Mach number given a ratio as an argument.

    Parameters
    ----------
        ratioName : string
            Name of the ratio given in input. Can be either one of:
                    ['p2/p1', 't2/t1', 'rho2/rho1', 'p02/p01', 'm2']
        ratio : array_like
            Actual value of the ratio. If float, list, tuple is given as input, 
            a conversion will be attempted.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            The upstream Mach number.
    """
    ratioName = ratioName.lower()
    
    ratios = {
        "p2/p1": M1_From_Pressure_Ratio,
        "t2/t1": M1_From_Temperature_Ratio,
        "rho2/rho1": M1_From_Density_Ratio,
        "p02/p01": M1_From_Total_Pressure_Ratio,
        "m2": M1_From_M2,
    }

    assert ratioName in ratios.keys(), "Unrecognized ratio '{}'".format(ratioName)

    return ratios[ratioName](ratio, gamma)

def Solve_Normal_Shockwave(name, value, gamma=1.4):
    """ Compute all the ratios and the Mach number downstream the shockwave.
    This method first compute the upstream Mach number M1 with the method
    Get_M1_From_Ratio, then it computes all the ratios.

    Parameters
    ----------
        name : string
            Name of the parameter given in input. Can be either one of:
                    ['p2/p1', 't2/t1', 'rho2/rho1', 'p02/p01', 'm2']
        value : array_like
            Actual value of the ratio. If float, list, tuple is given as input, 
            a conversion will be attempted.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M1 : array_like
            Mach number upstream of the shock wave.
        M2 : array_like
            Mach number downstream of the shock wave.
        pr : array_like
            Pressure ratio across the shock wave.
        dr : array_like
            Density ratio across the shock wave.
        tr : array_like
            Temperature ratio across the shock wave.
        ttr : array_like
            Total Temperature ratio across the shock wave.
        tpr : array_like
            Total Pressure ratio across the shock wave.
    """

    M1 = Get_M1_From_Ratio(name, value, gamma)

    if name == "p2/p1":
        pr = value
    else:
        pr = Pressure_Ratio.__bypass_decorator(M1, gamma)
    
    if name == "rho2/rho1":
        dr = value
    else:
        dr = Density_Ratio.__bypass_decorator(M1, gamma)
    
    if name == "t2/t1":
        tr = value
    else:
        tr = Temperature_Ratio.__bypass_decorator(M1, gamma)
    
    if name == "p02/p01":
        tpr = value
    else:
        tpr = Total_Pressure_Ratio.__bypass_decorator(M1, gamma)
    
    if name == "m2":
        M_2 = value
    else:
        M_2 = M2.__bypass_decorator(M1, gamma)

    ttr = Total_Temperature_Ratio.__bypass_decorator(M1, gamma)

    return M1, M_2, pr, dr, tr, ttr, tpr
