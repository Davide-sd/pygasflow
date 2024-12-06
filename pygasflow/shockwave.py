import numpy as np
import param
from scipy.optimize import bisect, minimize_scalar
from scipy.integrate import solve_ivp

from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.roots import apply_bisection
from pygasflow.generic import characteristic_mach_number
from pygasflow.utils.decorators import check_shockwave, check
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
    if np.any(M1 < 1):
        raise ValueError("The upstream Mach number must be M1 >= 1.")
    if (beta is None) and (theta is None):
        raise ValueError("To compute the normal " +
        "component of the upstream Mach number, you have to provide " +
        "either theta or beta.")
    flag = flag.lower()
    if flag not in ["weak", "strong", "both"]:
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
        "mn2": m1_from_m2.__no_check__,
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

    theta_max = max_theta_from_mach.__no_check__(M1, gamma)
    theta_max = np.atleast_1d(theta_max)

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            # here I chose 'weak', but in this case it's the same as 'strong'!
            beta[i] = beta_from_mach_theta(m, t, gamma=gamma)["weak"]
        return beta
    return beta_from_mach_theta(M1, theta_max, gamma=gamma)["weak"]

@check_shockwave
def beta_theta_max_for_unit_mach_downstream(M1, gamma=1.4):
    """
    Compute the shock maximum deflection angle, theta_max, as well as the
    wave angle beta corresponding to the unitary downstream Mach
    number, M2 = 1.

    Notes
    -----
    This function relies on root-finding algorithms. If a root can't be found,
    np.nan will be used as the result.

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

    theta_max = np.deg2rad(max_theta_from_mach.__no_check__(M1, gamma))

    if M1.shape:
        beta = np.zeros_like(M1)
        for i, (m, t) in enumerate(zip(M1, theta_max)):
            a = np.arcsin(1 / m)
            b = np.deg2rad(beta_from_mach_max_theta.__no_check__(m, gamma))
            try:
                beta[i] = bisect(func, a, b, args=(m, t))
            except ValueError:
                beta[i] = np.nan
        return np.rad2deg(beta), np.rad2deg(theta_max)

    a = np.arcsin(1 / M1)
    b = np.deg2rad(beta_from_mach_max_theta.__no_check__(M1, gamma))
    try:
        res = np.rad2deg(bisect(func, a, b, args=(M1, theta_max)))
    except ValueError:
        res = np.nan
    return res, np.rad2deg(theta_max)

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
    from a numerical standpoint.
    Take a look at
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
    pr_to_freestream = param.Number(1, bounds=(1, None), doc="""
        Pressure ratio multiplier between the pressure at the current state
        (M, theta_origin) and the free-stream pressure.""")
    theta_max = param.Number(constant=True, doc="""
        The maximum deflection angle possible with this pressure-deflection
        locus. Note that `theta_max` is considered relative to `theta_origin`
        along the left-running branch.""")
    func_of_theta = param.Parameter(constant=True, doc="""
        A numerical function whose only argument is the deflection angle (taken
        from a pressure-deflection diagram), returning two pressure ratios
        (weak and strong) associated with this pressure-deflection locus.""")
    upstream_locus = param.ClassSelector(
        class_=_BasePDLocus, constant=True, doc="""
        The locus associated with the upstream condition.""")

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
        from pygasflow.solvers import shockwave_solver

        res = shockwave_solver(
            "m1", self.M, "theta", abs(theta), gamma=self.gamma, to_dict=True)
        M2 = res["m2"]
        pr_to_freestream = res["pr"] * self.pr_to_freestream
        theta_origin = self.theta_origin + theta
        return type(self)(
            M=M2, gamma=self.gamma, label=label,
            theta_origin=theta_origin, pr_to_freestream=pr_to_freestream,
            upstream_locus=self)

    def __str__(self):
        s = f"Pressure-Deflection Locus of {self.label}\n"
        s += f"M\t\t{self.M}\n"
        s += f"p/p_inf\t\t{self.pr_to_freestream}\n"
        s += f"th_orig\t\t{self.theta_origin}\n"
        return s

    @param.depends(
        "M", "gamma", "theta_origin", "pr_to_freestream",
        watch=True, on_init=True
    )
    def update_func(self):
        theta_max = max_theta_from_mach(self.M, self.gamma)

        def func(theta):
            actual_theta = abs(self.theta_origin - theta)
            betas = beta_from_mach_theta(self.M, actual_theta, self.gamma)
            betas = [betas["weak"], betas["strong"]]
            Mn = self.M * np.sin(np.deg2rad(betas))
            # avoid rounding errors that would make Mn < 1
            idx = np.isclose(Mn, 1)
            Mn[idx] = 1
            pr = pressure_ratio(Mn, self.gamma) * self.pr_to_freestream
            return pr

        with param.edit_constant(self):
            self.theta_max = theta_max
            self.func_of_theta = func

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
            f1, f2 = self.func_of_theta, other.func_of_theta
        else:
            theta_origin1, theta_origin2 = other.theta_origin, self.theta_origin
            theta_max1, theta_max2 = other.theta_max, self.theta_max
            f1, f2 = other.func_of_theta, self.func_of_theta

        if theta_origin1 + theta_max1 < theta_origin2 - theta_max2:
            # no intersection found
            return None, None

        # TODO: This approach is not going to work if the intersection happens
        # between the weak region in one locus and the strong region in
        # the other. Maybe I should run a bisection with func(pressure_ratio)..
        def func(theta, region="weak"):
            pr1 = f1(theta)
            pr2 = f2(theta)
            if region == "weak":
                return pr1[0] - pr2[0]
            return pr1[1] - pr2[1]

        if a is None:
            a = min(theta_origin1 + theta_max1, theta_origin2 - theta_max2)
        if b is None:
            b = max(theta_origin1 + theta_max1, theta_origin2 - theta_max2)
        theta_intersection = bisect(func, a=a, b=b, args=(region,))
        pr_intersection = f1(theta_intersection)
        pr_intersection = pr_intersection[0] if region == "weak" else pr_intersection[1]
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
        pr *= self.pr_to_freestream

        return theta, pr

    def pressure_deflection_split_regions(self, N=100, include_mirror=False):
        """
        Helper function to build Pressure-Deflection plots. Computes the locus
        of all possible static pressure values behind oblique shock wave for
        given upstream conditions, and split them between weak and strong regions.

        Parameters are the same of
        :func:`~PressureDeflectionLocus.pressure_deflection`.

        Returns
        -------
        theta_weak : array_like
            Deflection angles for the weak region.
        pr_weak : array_like
            Pressure ratios for the weak region.
        theta_strong : array_like
            Deflection angles for the strong region.
        pr_strong : array_like
            Pressure ratios for the strong region.

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

            theta_1w, pr_1w, theta_1s, pr_1s = locus1.pressure_deflection_split_regions(
                include_mirror=False)
            theta_2w, pr_2w, theta_2s, pr_2s = locus2.pressure_deflection_split_regions(
                include_mirror=True)

            fig, ax = plt.subplots()
            ax.plot(theta_1w, pr_1w, label="M1 weak")
            ax.plot(theta_1s, pr_1s, label="M1 strong")
            ax.plot(theta_2w, pr_2w, label="M2 weak")
            ax.plot(theta_2s, pr_2s, label="M2 strong")
            ax.set_xlabel(r"Deflection Angle $\theta$ [deg]")
            ax.set_ylabel("Pressure Ratio to Freestream")
            ax.legend()
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', alpha=0.7)
            ax.grid(which='minor', linestyle=':', alpha=0.5)
            plt.show()

        """
        t, pr = self.pressure_deflection(N=N, include_mirror=False)

        idx = np.array(list(range(len(t))), dtype=int)
        n = len(t) / 2
        weak_idx = idx[idx < n]
        strong_idx = idx[idx >= n]
        theta_weak, pr_weak = t[weak_idx], pr[weak_idx]
        theta_strong, pr_strong = t[strong_idx], pr[strong_idx]

        if include_mirror:
            theta_weak -= self.theta_origin
            theta_strong -= self.theta_origin

            theta_weak = np.concatenate([-theta_weak[::-1], theta_weak])
            pr_weak = np.concatenate([pr_weak[::-1], pr_weak])
            theta_strong = np.concatenate([theta_strong, -theta_strong[::-1]])
            pr_strong = np.concatenate([pr_strong, pr_strong[::-1]])

            theta_weak += self.theta_origin
            theta_strong += self.theta_origin

        return theta_weak, pr_weak, theta_strong, pr_strong

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
        pr *= self.pr_to_freestream

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
    """ The :func:`beta_theta_c_for_unit_mach_downstream` function is really
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
                beta[i], theta_c[i] = beta_theta_c_for_unit_mach_downstream(m, g)
            except ValueError:
                beta[i], theta_c[i] = np.nan, np.nan
        df = pd.DataFrame.from_dict({
            "M1": M, "beta": beta, "theta_c": theta_c
        })
        filepath = os.path.join(folder, filename % g)
        df.to_csv(filepath, index=False, compression="zip")
