import numpy as np
from scipy.optimize import bisect, minimize_scalar
from scipy.integrate import solve_ivp

from pygasflow.utils.roots import Apply_Bisection
from pygasflow.generic import Characteristic_Mach_Number
from pygasflow.utils.decorators import Check_Shockwave

# NOTE:
# In the following module:
#   beta: shock wave angle.
#   theta: flow deflection angle.

#####################################################################################
############# The following methods are specific for normal shock waves. ############
##### They can also be used for calculation with oblique shock wave, just use #######
##################### the normal component of the Mach number. ######################
#####################################################################################

@Check_Shockwave
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

@Check_Shockwave
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

@Check_Shockwave
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

@Check_Shockwave
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

@Check_Shockwave
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

# here I use Check_Mach_Specific_Heat_Ratio and not 
# Check_Mach_Specific_Heat_Ratio(1) because this function
# can also be used to compute M1 given M2 (which is M2 < 1)
@Check_Shockwave
def Mach_Downstream(M1, gamma=1.4):
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

@Check_Shockwave
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

@Check_Shockwave
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

@Check_Shockwave
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

@Check_Shockwave
def M1_From_Total_Pressure_Ratio(ratio, gamma=1.4):
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

@Check_Shockwave
def M1_From_M2(M2, gamma=1.4):
    """ Compute M1 from the downstream Mach number M2. 

    Parameters
    ----------
        M2 : array_like
            Downstream Mach Number. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be ((gamma - 1) / 2 / gamma) < M_2 < 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        out : ndarray
            Upstream Mach number M1.
    """
    lower_lim = (gamma - 1) / 2 / gamma
    assert np.all(M2 >= lower_lim) and np.all(M2 < 1), ("The downstream M2 must be " +
            str(lower_lim) + " < M2 < 1")

    return Mach_Downstream(M2, gamma)

#######################################################################################
############## The following methods are specific for oblique shock waves #############
#######################################################################################

@Check_Shockwave
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

    num = M1**2 * np.sin(beta)**2 - 1
    den = M1**2 * (gamma + np.cos(2 * beta)) + 2
    theta = np.asarray(np.arctan(2 / np.tan(beta) * num / den))
    # need to take into account that we are considering only positive
    # values of the Flow Angle Theta.
    theta[theta < 0] = np.nan

    return np.rad2deg(theta)


@Check_Shockwave
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

        # beta_weak = np.arctan(np.sqrt(Xw / (1 - Xw)))
        # beta_strong = np.arctan(np.sqrt(Xs / (1 - Xs)))

        return beta_weak, beta_strong
    
    beta_weak, beta_strong = np.zeros_like(M1), np.zeros_like(M1)

    if M1.size == 1:    # scalar case
        # print(Q, R, D)
        if M1 == 1:
            beta_weak, beta_strong = np.pi / 2, np.pi / 2
        else:
            beta_weak, beta_strong = func(Q, R, D, b)
    else:
        for i, _d in np.ndenumerate(D):
            # print(Q[i], R[i], _d)
            beta_weak[i], beta_strong[i] = func(Q[i], R[i], _d, b[i])
        # idx = M1 == 1
        # beta_weak[idx], beta_strong[idx] = 90, 90

    return { "weak": np.rad2deg(beta_weak), "strong": np.rad2deg(beta_strong) }


@Check_Shockwave([0, 1])
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
    assert np.all(M1 - MN1 >= 0), "Upstream Mach number must be >= of the normal upstream Mach number."
    return np.rad2deg(np.arcsin(MN1 / M1))

@Check_Shockwave
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
        print("asd", beta)
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


def Get_Upstream_Normal_Mach_From_Ratio(ratioName, ratio, gamma=1.4):
    """
    Compute the upstream Mach number given a ratio as an argument.

    Parameters
    ----------
        ratioName : string
            Name of the ratio given in input. Can be either one of:
                    ['pressure', 'temperature', 'density', 'total_pressure', 'mn2']
            This ratio is in the form downstream/upstream, therefore:
                'pressure': p2/p1
                'temperature': t2/t1
                'density': rho2/rho1
                'total_pressure': p02/p01
                'mn2': Normal Mach downstream of the shock wave
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
        "pressure": M1_From_Pressure_Ratio,
        "temperature": M1_From_Temperature_Ratio,
        "density": M1_From_Density_Ratio,
        "total_pressure": M1_From_Total_Pressure_Ratio,
        "mn2": M1_From_M2,
    }

    assert ratioName in ratios.keys(), "Unrecognized ratio '{}'".format(ratioName)

    return ratios[ratioName](ratio, gamma)

@Check_Shockwave
def Get_Ratios_From_Normal_Mach_Upstream(Mn, gamma=1.4):
    """
    Compute the ratios of the quantities across a Shock Wave given the Normal Mach number.

    Parameters
    ----------
        Mn : array_like
            Normal Mach number upstream of the shock wave. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
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
    # TODO: use __no_check
    pr = Pressure_Ratio(Mn, gamma)
    dr = Density_Ratio(Mn, gamma)
    tr = Temperature_Ratio(Mn, gamma)
    tpr = Total_Pressure_Ratio(Mn, gamma)
    mn2 = Mach_Downstream(Mn, gamma)

    return pr, dr, tr, tpr, mn2

@Check_Shockwave
def Maximum_Mach_From_Deflection_Angle(theta, gamma=1.4):
    """
    Compute the maximum Mach number from a given Deflection angle theta.

    Parameters
    ----------
        theta : float
            Deflection angle in degrees. Must be 0 <= theta <= 90.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M : float
            The maximum Mach number for the specified theta.
    """
    upper_lim = Max_Theta_From_Mach(np.finfo(np.float32).max, gamma)
    assert theta < upper_lim, "The flow deflection angle must be < {}, which correspond to Mach=infinity.".format(upper_lim)
    
    def function(t):
        def func(M):
            theta_max = Max_Theta_From_Mach(M, gamma)
            return theta_max - t
        
        # TODO: 
        # 0. Do I really need to do a = 1 + 1e-08 ????
        # 1. what if the actual M is > 1000???
        # 2. this is a slow procedure, can it be done faster, differently?
        a = 1 + 1e-08        
        b = 1000
        return bisect(func, a, b)

    if theta.shape:
        Max_M = np.zeros_like(theta)
        for i, t in enumerate(theta):
            Max_M[i] = function(t)
        return Max_M
    return function(theta)

@Check_Shockwave
def Mimimum_Beta_From_Mach(M1):
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

@Check_Shockwave
def Max_Theta_From_Mach(M1, gamma=1.4):
# def Max_Deflection_Angle_From_Mach(M, gamma=1.4):
    """
    Compute the maximum deflection angle for a given upstream Mach number.

    Parameters
    ----------
        M1 : array_like
            Upstream Mach number. If float, list, tuple is given as input,
            a conversion will be attempted. Must be M1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        Theta_max : ndarray
            Maximum deflection angle theta in degrees
    """
    # http://www.pdas.com/maxwedge.xml

    # Rewrite Anderson's eq. 4.17 to have cotan(theta) = tan(beta)/2 * ....
    # We minimize the right hand side, which is a function of Beta. This means we find 
    # the minimum of cotan(theta), which corresponds to the maximum deflection angle theta.

    def function(M):
        # Right hand side function of beta
        func = lambda beta: np.tan(beta) * ((M**2 * (gamma + 1)) / (2 * (M**2 * np.sin(beta)**2 - 1)) - 1)
        # bound a correspond to the discontinuity of func
        a = np.arcsin(1 / M)
        b = np.pi / 2

        # result of the minimization. res.x correspond to the value of Beta where function is
        # minimized
        res = minimize_scalar(func, bounds=(a,b), method='bounded')
        # cotan(theta_max) = func(beta_min)
        # Therefore theta_max = arctan(1 / func(beta_min))
        return np.rad2deg(np.arctan(1 / func(res.x)))

    if M1.shape:
        theta_max = np.zeros_like(M1)
        for i, m in enumerate(M1):
            theta_max[i] = function(m)
        return theta_max
    return function(M1)

@Check_Shockwave
def Beta_From_Mach_Max_Theta(M1, gamma=1.4):
    """
    Compute the shock wave angle beta corresponding to the maximum deflection angle theta
    given an upstream Mach number.

    Parameters
    ----------
        M1 : array_like
            Mach number. If float, list, tuple is given as input, a conversion
            will be attempted. Must be M1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        Beta : array_like
            The shock angle in degrees.
    """

    theta_max = Max_Theta_From_Mach.__no_check(M1, gamma)

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            # here I chose 'weak', but in this case it's the same as 'strong'!
            beta[i] = Beta_From_Mach_Theta(m, t)["weak"]
        return beta
    return Beta_From_Mach_Theta(M1, theta_max)["weak"]

@Check_Shockwave
def Beta_Theta_Max_For_Unit_Mach_Downstream(M1, gamma=1.4):
    """
    Compute the shock maximum deflection angle, theta_max, as well as the
    wave angle beta corresponding to the unitary downstream Mach number, M2 = 1.

    Parameters
    ----------
        M1 : array_like
            Upstream Mach number. If float, list, tuple is given as input,
            a conversion will be attempted. Must be M1 >= 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        Beta : array_like
            The shock angle in degrees corresponding to M2=1.
        Theta max : array_like
            The maximum deflection angle in degrees corresponding to M2=1.
    """
    # assert np.all(M >= 1), "Upstream Mach number must be M >= 1."

    def func(b, M, t):
        return ((1 + (gamma - 1) / 2 * (M * np.sin(b))**2) / (gamma * (M * np.sin(b))**2 - (gamma - 1) / 2)) - np.sin(b - t)**2
    
    theta_max = np.deg2rad(Max_Theta_From_Mach.__no_check(M1, gamma))

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            a = np.arcsin(1 / m)
            b = np.deg2rad(Beta_From_Mach_Max_Theta.__no_check(m, gamma))
            beta[i] = bisect(func, a, b, args=(m, t))
        return np.rad2deg(beta), np.rad2deg(theta_max)

    a = np.arcsin(1 / M1)
    b = np.deg2rad(Beta_From_Mach_Max_Theta.__no_check(M1, gamma))
    return np.rad2deg(bisect(func, a, b, args=(M1, theta_max))), np.rad2deg(theta_max)

@Check_Shockwave([0, 1])
def Mach_From_Theta_Beta(theta, beta, gamma=1.4):
    """
    Compute the upstream Mach number given the flow deflection angle and the shock wave angle.
    
    Parameters
    ----------
        theta : array_like
            Flow deflection angle in degrees. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= theta <= 90.
        beta : array_like
            Shock wave angle in degrees. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be 0 <= beta <= 90.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        Mach : ndarray
            The upstream Mach number.
    """
    assert beta.shape == theta.shape, "Flow deflection angle and Shock wave angle must have the same shape."

    # case beta == 90 and theta == 0, which leaves M to be indeterminate, NaN
    idx0 = np.bitwise_and(beta == 90, theta == 0)
    # if beta == 0 and theta == 0, mach goes to infinity. But out num and den both
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

    mach = np.zeros_like(beta, dtype=np.float)
    mach[den > 0] = np.sqrt(num[den > 0] / den[den > 0])
    mach[den <= 0] = np.nan

    mach[idx0] = np.nan
    mach[idx1] = np.inf
    return mach

@Check_Shockwave
def Shock_Polar(M1, gamma=1.4, N=100):
    """
    Compute the ratios (Vx/a*), (Vy/a*) for plotting a Shock Polar.

    Parameters
    ----------
        M1 : float
            Upstream Mach number of the shock wave. Must be > 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        N : int
            Number of discretization steps in the range [2, 2*pi]. Must be > 1.
    
    Returns
    -------
        (Vx/a*) : ndarray [1 x N]
            x-coordinate for the shock polar plot
        (Vy/a*) : ndarray [1 x N]
            y-coordinate for the shock polar plot
    """
    assert isinstance(N, (int)) and N > 1, "The number of discretization steps must be integer and > 1."

    M1s = Characteristic_Mach_Number(M1, gamma)
    # downstream Mach number to a normal shock wave
    M_2 = Mach_Downstream(M1)
    M2s = Characteristic_Mach_Number(M_2, gamma)

    def _shock_polar(Vx_as_ratio, M1s):
        # equation 4.22 (Anderson)
        num = (M1s - Vx_as_ratio)**2 * (Vx_as_ratio * M1s - 1)
        # account for numerical errors leading (num) to be proportional to a very small 
        # negative value
        num[num < 0] = 0
        den = (2 / (gamma + 1)) * M1s**2 - Vx_as_ratio * M1s + 1
        return np.sqrt(num / den)

    # polar coordinate
    alpha = np.linspace(0, np.pi, 100)
    r = (M1s - M2s) / 2
    Vx_as = M2s + r + r * np.cos(alpha)
    Vy_as = _shock_polar(Vx_as, M1s)

    Vx_as = np.append(Vx_as, Vx_as[::-1])
    Vy_as = np.append(Vy_as, -Vy_as[::-1])

    return Vx_as, Vy_as

@Check_Shockwave
def Pressure_Deflection(M1, gamma=1.4, N=100):
    """
    Helper function to build Pressure-Deflection plots.

    Parameters
    ----------
        M1 : float
            Upstream Mach number. Must be > 1.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        N : int
            Half number of points discretizing the range [0, theta_max]. 
            This function compute N points in the range [0, theta_max] for the 'weak'
            solution, then compute N points in the range [theta_max, 0] for the 
            'strong' solution.
    
    Returns
    -------
        theta : array_like
            Deflection angles
        pr : array_like
            Pressure ratios computed for the given Mach and the above deflection angles.
    """    
    theta_max = Max_Theta_From_Mach.__no_check(M1, gamma)

    theta = np.linspace(0, theta_max, N)
    theta = np.append(theta, theta[::-1])
    beta = np.zeros_like(theta)

    for i in range(N):
        betas = Beta_From_Mach_Theta.__no_check(M1, theta[i], gamma)
        beta[i] = betas["weak"]
        beta[len(theta) -1 - i] = betas["strong"]
    
    # TODO:
    # it may happend to get a NaN, especially for theta=0, in that case, manual correction
    # idx = np.where(np.isnan(beta))
    # beta[idx] = 1

    Mn = M1 * np.sin(np.deg2rad(beta))

    return theta, Pressure_Ratio.__no_check(Mn, gamma)


###########################################################################################
################################# Conical Flow Relations ##################################
###########################################################################################

def Taylor_Maccoll(theta, V, gamma=1.4):
    """
    Taylor-Maccoll differential equation for conical shock wave.

    Parameters
    ----------
        theta : float
            Polar coordinate, angle in radians.
        V : list
            Velocity Vector with components:
            V_r: velocity along the radial direction
            V_theta: velocity along the polar direction
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        dV_dtheta : list
            Taylor-Maccoll differential equation.
    """
    # Reorganizing Anderson's equation 10.15:
    V_r, V_theta = V

    dV_dtheta = [
        V_theta,
        (V_r * V_theta**2 - (gamma - 1) / 2 * (1 - V_r**2 - V_theta**2) * (2 * V_r + V_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - V_r**2 - V_theta**2) - V_theta**2)
    ]

    return dV_dtheta

@Check_Shockwave
def Nondimensional_Velocity(M, gamma=1.4):
    """
    Compute the Nondimensional Velocity given the Mach number.

    Parameters
    ----------
        M : array_like
            Mach number. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be M >= 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        V : array_like
            Nondimensional Velocity
    """
    # Anderson's equation 10.16
    return np.sqrt((gamma - 1) * M**2 / (2 + (gamma - 1) * M**2))

@Check_Shockwave
def Mach_From_Nondimensional_Velocity(V, gamma=1.4):
    """
    Compute the Mach number given the Nondimensional Velocity.

    Parameters
    ----------
        V : array_like
            Nondimensional Velocity. If float, list, tuple is given as input, 
            a conversion will be attempted. Must be V > 0.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M : array_like
            Mach number
    """
    assert np.all(V > 0), "Nondimensional velocity must be V > 0."
    # inverse Anderson's equation 10.16
    return np.sqrt(2 * V**2 / ((gamma - 1) * (1 - V**2)))

# @Check
def Mach_Cone_Angle_From_Shock_Angle(M, beta, gamma=1.4):
    """
    Compute the half-cone angle and the Mach number at the surface of the cone.

    Parameters
    ----------
        M : float
            Upstream Mach number. Must be > 1.
        beta : float
            Shock Angle in degrees. Must be 0 <= beta <= 90
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        Mc : float
            Mach number at the surface of the cone
        theta_c : float
            Half-cone angle in degrees. 
    """

    # TODO:
    # should I create a new decorator to avoid the assertion checks during
    # iterative procedures????

    # # TODO: is this condition correct: 0 <= beta <= 90 ???
    # assert isinstance(beta, (int, float)) and bet >= 0 and beta <= 90, "The shock wave angle must be 0 <= beta <= 90." 

    # TODO:
    # WARNING: when using this function and beta has been assumed to be the Mach angle,
    # the numerator in Theta_From_Mach_Beta should go to zero, hence theta = 0.
    # Rounding error may happens that prevents this, resulting in very very small values
    # of thetas, which may also happens to be negative.
    # NEED TO TAKE CARE OF THIS

    # flow deflection angle
    delta = Theta_From_Mach_Beta.__no_check(M, beta, gamma)

    # TODO:
    # this check need to be done correctly
    # check for detachment
    if delta < 0 or np.isnan(delta):
        return None, None
    
    # Mach downstream of the shock wave
    MN1 = M * np.sin(np.deg2rad(beta))
    MN2 = Mach_Downstream(MN1, gamma)
    M_2 = MN2 / np.sin(np.deg2rad(beta - delta))

    # compute the nondimensional velocity components
    V0 = Nondimensional_Velocity.__no_check(M_2, gamma)
    V0_r = V0 * np.cos(np.deg2rad(beta - delta))
    V0_theta = -V0 * np.sin(np.deg2rad(beta - delta))

    # range of integration
    thetas = [np.deg2rad(beta), 0]
    # initial condition
    V0_ic = [V0_r, V0_theta]

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
    result = solve_ivp(lambda theta, V: Taylor_Maccoll(theta, V, gamma), thetas, V0_ic, events=event)

    if not result.success:
        raise Exception("Could not successfully integrate Taylor-Maccoll equation.\n" +
            "Here some useful data for debug:\n" + 
            "\tInput Mach number: {}\n".format(M) + 
            "\tInput Shock Wave angle [degrees]: {}\n".format(beta) +
            "\tRange of integration [degrees]: [{}, 0]\n".format(beta) + 
            "\tInitial Conditions: [Vr = {}, V_theta = {}]\n".format(V0_r, V0_theta) 
            )

    # the cone angle is the angle where V_theta = 0.
    theta_c = np.rad2deg(result.t[-1])
    # at the cone surface, V_theta = 0, therefore V = V_r
    Vc = result.y[0, -1]
    # Mach number at the cone surface
    Mc = Mach_From_Nondimensional_Velocity.__no_check(Vc, gamma)

    return Mc, theta_c

@Check_Shockwave
def Shock_Angle_From_Mach_Cone_Angle(M, theta_c, gamma=1.4, step=0.0025):
    """
    Compute the shock wave angle.

    Parameters
    ----------
        M : float
            Upstream Mach number. Must be > 1.
        theta_c : float
            Half cone angle in degrees. Must be 0 < theta_c < 90
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        step : float
            Angle-Increment used on the shock wave angle iteration. Default to 0.025 deg.
    
    Returns
    -------
        Mc : float
            Mach number at the surface of the cone, computed.
        theta_c : float
            Half cone angle of the cone in degrees.
        beta : float
            Shock wave angle in degrees. 
    """

    # assume a shock wave angle beta.
    # use the mach angle since mu < beta < 90 deg
    beta_hp = np.rad2deg(np.arcsin(1 / M))

    Mc_hp, theta_c_hp = Mach_Cone_Angle_From_Shock_Angle(M, beta_hp, gamma)

    # TODO:
    # the while loop is slow, really really slow.
    # evaluate if it's possible to do a bisection, minimization problem: the issue
    # will be the detachment detection! 

    while theta_c_hp < theta_c:
        beta_hp += step
        Mc_hp, theta_c_hp = Mach_Cone_Angle_From_Shock_Angle(M, beta_hp, gamma)
        
        # check for detachment
        if Mc_hp == None and theta_c_hp == None:
            return Mc_hp, theta_c_hp

    return Mc_hp, theta_c_hp, beta_hp

@Check_Shockwave
def Shock_Angle_From_Machs(M1, Mc, gamma=1.4, step=0.0025):
    """
    Compute the shock wave angle.

    Parameters
    ----------
        M1 : float
            Upstream Mach number. Must be > 1.
        Mc : float
            Mach number at the surface of the cone.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        step : float
            Angle-Increment used on the shock wave angle iteration. Default to 0.0025 deg.
    
    Returns
    -------
        Mc : float
            Mach number at the surface of the cone, computed.
        theta_c : float
            Half cone angle of the cone in degrees.
        beta : float
            Shock wave angle in degrees. 
    """

    # assume a shock wave angle beta.
    # use the mach angle since mu < beta < 90 deg
    beta_hp = np.rad2deg(np.arcsin(1 / M1))
    
    Mc_hp, theta_c_hp = Mach_Cone_Angle_From_Shock_Angle(M1, beta_hp, gamma)

    # TODO:
    # the while loop is slow, really really slow.
    # evaluate if it's possible to do a bisection, minimization problem: the issue
    # will be the detachment detection! 

    while Mc_hp > Mc:
        beta_hp += step
        Mc_hp, theta_c_hp = Mach_Cone_Angle_From_Shock_Angle(M1, beta_hp, gamma)
        
        # check for detachment
        if Mc_hp == None and theta_c_hp == None:
            return Mc_hp, theta_c_hp

    return Mc_hp, theta_c_hp, beta_hp