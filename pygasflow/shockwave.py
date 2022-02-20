import numpy as np
from scipy.optimize import bisect, minimize_scalar
from scipy.integrate import solve_ivp

from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.roots import apply_bisection
from pygasflow.generic import characteristic_mach_number
from pygasflow.utils.decorators import check_shockwave, check
import warnings

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

    References
    ----------

    "Equations, Tables and Charts for compressible flow", NACA R-1135, 1953
    """
    return ((gamma + 1) * M1**2 / 2)**(gamma / (gamma - 1)) * ((gamma + 1) / (2 * gamma * M1**2 - (gamma - 1)))**(1 / (gamma - 1))

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
    if np.any(np.isnan(theta)):
        warnings.warn("WARNING: detachment detected in at least one element of " +
        "the flow turning angle theta array. Be careful!")

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

        # beta_weak = np.arctan(np.sqrt(Xw / (1 - Xw)))
        # beta_strong = np.arctan(np.sqrt(Xs / (1 - Xs)))

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
        Can be either ``'weak'`` or ``'strong'``. Default to ``'weak'``.
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
    # if np.any(M1 < 1):
    #     raise ValueError("The upstream Mach number must be M1 >= 1.")
    if (beta is None) and (theta is None):
        raise ValueError("To compute the normal " +
        "component of the upstream Mach number, you have to provide " +
        "either theta or beta.")
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
        beta = beta_from_mach_theta(M1, theta)
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
        * ``'mn2'``: Normal Mach downstream of the shock wave

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
    if isinstance(ratio, (list, tuple)):
        ratio = np.asarray(ratio)

    ratios = {
        "pressure": m1_from_pressure_ratio,
        "temperature": m1_from_temperature_ratio,
        "density": m1_from_density_ratio,
        "total_pressure": m1_from_total_pressure_ratio,
        "mn2": m1_from_m2.__no_check,
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
    # TODO: use __no_check
    pr = pressure_ratio(Mn, gamma)
    dr = density_ratio(Mn, gamma)
    tr = temperature_ratio(Mn, gamma)
    tpr = total_pressure_ratio(Mn, gamma)
    mn2 = mach_downstream(Mn, gamma)

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
    M : float
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

    theta_max = max_theta_from_mach.__no_check(M1, gamma)

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            # here I chose 'weak', but in this case it's the same as 'strong'!
            beta[i] = beta_from_mach_theta(m, t)["weak"]
        return beta
    return beta_from_mach_theta(M1, theta_max)["weak"]

@check_shockwave
def beta_theta_max_for_unit_mach_downstream(M1, gamma=1.4):
    """
    Compute the shock maximum deflection angle, theta_max, as well as the
    wave angle beta corresponding to the unitary downstream Mach
    number, M2 = 1.

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

    def func(b, M, t):
        return ((1 + (gamma - 1) / 2 * (M * np.sin(b))**2) / (gamma * (M * np.sin(b))**2 - (gamma - 1) / 2)) - np.sin(b - t)**2

    theta_max = np.deg2rad(max_theta_from_mach.__no_check(M1, gamma))

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            a = np.arcsin(1 / m)
            b = np.deg2rad(beta_from_mach_max_theta.__no_check(m, gamma))
            beta[i] = bisect(func, a, b, args=(m, t))
        return np.rad2deg(beta), np.rad2deg(theta_max)

    a = np.arcsin(1 / M1)
    b = np.deg2rad(beta_from_mach_max_theta.__no_check(M1, gamma))
    return np.rad2deg(bisect(func, a, b, args=(M1, theta_max))), np.rad2deg(theta_max)

@check_shockwave([0, 1])
def mach_from_theta_beta(theta, beta, gamma=1.4):
    """
    Compute the upstream Mach number given the flow deflection angle and the shock wave angle.

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
        raise ValueError("Flow deflection angle and Shock wave angle must have the same shape.")

    # make sure (beta, theta) is in the allowable space: consider for example
    # M1 = inf and the Oblique Shock Diagram. Then, there exis a theta_max
    # corresponding to the provided beta. If theta > theta_max, there is no
    # solution.
    theta_max = np.zeros_like(beta)
    for i, b in enumerate(beta):
        # I approximate Mach = inf with 100000.
        theta_max[i] = theta_from_mach_beta(100000, b, gamma)
    if np.any(theta > theta_max):
        raise ValueError("There is no solution for the current choice of" +
        " parameters. Please check the Oblique Shock diagram with the following"
        " parameters:\n" +
        "beta = {}\n".format(beta) +
        "theta = {}\n".format(theta))
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

    mach = np.zeros_like(beta, float)
    mach[den > 0] = np.sqrt(num[den > 0] / den[den > 0])
    mach[den <= 0] = np.nan

    mach[idx0] = np.nan
    mach[idx1] = np.inf
    return mach

@check_shockwave
def shock_polar(M1, gamma=1.4, N=100):
    """
    Compute the ratios (Vx/a*), (Vy/a*) for plotting a Shock Polar.

    Parameters
    ----------
    M1 : float
        Upstream Mach number of the shock wave. Must be > 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    N : int, optional
        Number of discretization steps in the range [2, 2*pi]. Must be > 1.

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
    M_2 = mach_downstream(M1)
    M2s = characteristic_mach_number(M_2, gamma)

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

@check_shockwave
def pressure_deflection(M1, gamma=1.4, N=100):
    """
    Helper function to build Pressure-Deflection plots.

    Parameters
    ----------
    M1 : float
        Upstream Mach number. Must be > 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    N : int, optional
        Half number of points discretizing the range [0, theta_max].
        This function compute N points in the range [0, theta_max] for
        the `'weak'` solution, then compute N points in the range
        [theta_max, 0] for the `'strong'` solution.

    Returns
    -------
    theta : array_like
        Deflection angles
    pr : array_like
        Pressure ratios computed for the given Mach and the above
        deflection angles.
    """
    if (not isinstance(N, int)) or (N <= 1):
        raise ValueError("The number of discretization steps must be integer and > 1.")

    theta_max = max_theta_from_mach.__no_check(M1, gamma)

    theta = np.linspace(0, theta_max, N)
    theta = np.append(theta, theta[::-1])
    beta = np.zeros_like(theta)

    for i in range(N):
        betas = beta_from_mach_theta.__no_check(M1, theta[i], gamma)
        beta[i] = betas["weak"]
        beta[len(theta) -1 - i] = betas["strong"]

    # TODO:
    # it may happend to get a NaN, especially for theta=0, in that case, manual correction
    # idx = np.where(np.isnan(beta))
    # beta[idx] = 1

    Mn = M1 * np.sin(np.deg2rad(beta))

    return theta, pressure_ratio.__no_check(Mn, gamma)


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
    if np.any(V <= 0):
        raise ValueError("Nondimensional velocity must be V > 0.")
    # inverse Anderson's equation 10.16
    return np.sqrt(2 * V**2 / ((gamma - 1) * (1 - V**2)))

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
    """
    # NOTE:
    # Here we implement the first 3-ish steps of Anderson's "Modern Compressible
    # Flow" procedure for solving the flow over a circular cone (Chapter 10,
    # Section 4). The 'ish' part indicates that this function only solve the
    # first part of step 3. The other part is left for other functions.

    # Step 1. Compute M2 and theta (delta, in the book), just behind the shock.
    # delta = flow deflection angle
    delta = theta_from_mach_beta.__no_check(M, beta, gamma)

    # check for detachment
    if delta < 0 or np.isnan(delta):
        print(delta)
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
    V0 = nondimensional_velocity.__no_check(M_2, gamma)
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
    Mc = mach_from_nondimensional_velocity.__no_check(Vc, gamma)

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
        Mc, tcmax, bmax = max_theta_c_from_mach(M)
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

    if M1.shape:
        theta_c_comp = np.zeros_like(M1)
        beta = np.zeros_like(M1)
        Mc = np.zeros_like(M1)
        for i, m in enumerate(M1):
            Mc[i], theta_c_comp[i], beta[i] = function(m)
        return ret_correct_vals(Mc), ret_correct_vals(theta_c_comp), ret_correct_vals(beta)
    return function(M1)

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
        _, tcmax, bmax = max_theta_c_from_mach(M)
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

    if M1.shape:
        theta_c_comp = np.zeros_like(M1)
        beta_comp = np.zeros_like(M1)
        Mc_comp = np.zeros_like(M1)
        for i, m1 in enumerate(M1):
            Mc_comp[i], theta_c_comp[i], beta_comp[i] = function(m1)
        return ret_correct_vals(Mc_comp), ret_correct_vals(theta_c_comp), ret_correct_vals(beta_comp)
    return function(M1)


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

    if M1.shape:
        theta_c_max = np.zeros_like(M1)
        beta = np.zeros_like(M1)
        Mc = np.zeros_like(M1)
        for i, m in enumerate(M1):
            Mc[i], theta_c_max[i], beta[i] = function(m)
        return ret_correct_vals(Mc), ret_correct_vals(theta_c_max), ret_correct_vals(beta)
    return function(M1)

@check_shockwave
def beta_theta_c_for_unit_mach_downstream(M1, gamma=1.4):
    """ Given an upstream Mach number, compute the point (beta, theta_c)
    where the downstream Mach number is sonic.
    WARNING: this procedure is really slow!

    Parameters
    ----------
    M1 : array_like
        Upstream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be M1 >= 1.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    theta_c : ndarray
        Cone angle theta_c in degrees.
    beta : ndarray
        Shockwave angle corresponding in degrees.
    """
    def function(M1):
        def func(theta_c):
            Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag="weak")
            MN1 = M1 * np.sin(np.deg2rad(beta))
            MN2 = mach_downstream(MN1, gamma)
            # delta is the flow deflection angle (Anderson's Figure 10.4)
            delta = theta_from_mach_beta(M1, beta, gamma)
            M2 = MN2 /  np.sin(np.deg2rad(beta - delta))
            return M2 - 1
        # theta_c search interval
        # TODO: is a=1 reasonable for very low M1???
        a = 1
        b = max_theta_c_from_mach(M1, gamma)[1]
        theta_c = bisect(func, a, b)
        Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag="weak")
        return beta, theta_c

    if M1.shape:
        theta_c = np.zeros_like(M1)
        beta = np.zeros_like(M1)
        for i, m in enumerate(M1):
            theta_c, beta = function(m)
        return theta_c, beta
    return function(M1)

def load_data(gamma=1.4):
    """ The ``beta_theta_c_for_unit_mach_downstream`` function is really slow
    in computing the data. Often, that data is needed in a plot. Here, a few
    precomputed tables has been provided.

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

    import csv
    import os
    # path of the folder containing this file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # path of the folder containing the data of the plot
    data_dir = os.path.join(current_dir, "data")
    filename = os.path.join(data_dir, "m-beta-theta_c-g" + str(gamma) + ".csv")
    if not os.path.exists(filename):
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        raise FileNotFoundError("Could not find the file: {}\n".format(filename) +
        "The following files are available: {}\n".format(files) +
        "Insert the interested specific heat ratio!")
    mach, beta, theta_c = [], [], []
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        count = 0
        for row in csv_reader:
            if count > 0:
                mach.append(float(row["M1"]))
                beta.append(float(row["beta"]))
                theta_c.append(float(row["theta_c"]))
            count += 1

    return mach, beta, theta_c

if __name__ == "__main__":
    # print(np.rad2deg(np.arcsin(1 / 1.1)))
    # print(mach_cone_angle_from_shock_angle(1.1, 65.3800226713429))
    # print(beta_theta_c_for_unit_mach_downstream(2))
    # print(max_theta_c_from_mach(2))
    # print(mach_cone_angle_from_shock_angle(1.05, ))
    # M1 = np.logspace(0, 1, 50)
    # print(M1)

    # print(load_data())

    # print(normal_mach_upstream(2, theta=30))
    print(mach_from_theta_beta(45, 50))


    # M1 = np.logspace(0, 1, 50) / 100 + 0.99
    # M1 = [1.1, 1.12, 1.14, 1.16, 1.18, 1.20, 1.23, 1.26, 1.29, 1.32, 1.35, 1.38,
    #     1.41, 1.44, 1.47, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10,
    #     100, 10000]
    # for i, m in enumerate(M1):
    #     # if i > 1:
    #     b, t = beta_theta_c_for_unit_mach_downstream(m)
    #     print(i, m, b, t)
