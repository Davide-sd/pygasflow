import numpy as np
import warnings

from pygasflow.isentropic import (
    pressure_ratio as ise_PR,
    density_ratio as ise_DR,
    temperature_ratio as ise_TR
)

from pygasflow.shockwave import (
    mach_from_theta_beta,
    get_upstream_normal_mach_from_ratio,
    get_ratios_from_normal_mach_upstream,
    normal_mach_upstream,
    theta_from_mach_beta,
    beta_from_mach_theta,
    shock_angle_from_machs,
    mach_cone_angle_from_shock_angle,
    shock_angle_from_mach_cone_angle,
    beta_from_upstream_mach,
    theta_from_mach_beta,
)

from pygasflow.utils.common import convert_to_ndarray
from pygasflow.utils.decorators import check_shockwave

# TODO:
# detachment detection, when provided theta > theta_max for the specified Mach
# number

@check_shockwave([1, 3])
def shockwave_solver(p1_name, p1_value, p2_name="beta", p2_value=90, gamma=1.4, flag="weak", to_dict=False):
    """
    Try to compute all the ratios, angles and mach numbers across the shock wave.

    Remember: a normal shock wave has a wave angle beta=90 deg.

    Parameters
    ----------
    p1_name : string
        Name of the first parameter given in input. Can be either one of:

        * ``'pressure'``: Pressure Ratio P2/P1
        * ``'temperature'``: Temperature Ratio T2/T1
        * ``'density'``: Density Ratio rho2/rho1
        * ``'total_pressure'``: Total Pressure Ratio P02/P01
        * ``'m1'``: Mach upstream of the shock wave
        * ``'mn1'``: Normal Mach upstream of the shock wave
        * ``'mn2'``: Normal Mach downstream of the shock wave
        * ``'beta'``: The shock wave angle [in degrees]. It can only be used
          if ``p2_name='theta'``.
        * ``'theta'``: The deflection angle [in degrees]. It can only be
          used if ``p2_name='beta'``.

        If the parameter is a ratio, it is in the form downstream/upstream:

    p1_value : float
        Actual value of the parameter.
    p2_name : string, optional
        Name of the second parameter. It could either be:

        * ``'beta'``: Shock wave angle.
        * ``'theta'``: Flow deflection angle.
        * ``'mn1'``: Input Normal Mach number.

        Default to ``'beta'``.
    p2_value : float, optional
        Value of the angle in degrees.
        Default to 90 degrees (normal shock wave).
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.
    flag : string, optional
        Chose what solution to compute if the angle 'theta' is provided.
        Can be either ``'weak'`` or ``'strong'``. Default to ``'weak'``.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    m1 : float
        Mach number upstream of the shock wave.
    mn1 : float
        Normal Mach number upstream of the shock wave.
    m2 : float
        Mach number downstream of the shock wave.
    mn2 : float
        Normal Mach number downstream of the shock wave.
    beta : float
        Shock wave angle in degrees.
    theta : float
        Flow deflection angle in degrees.
    pr : float
        Pressure ratio across the shock wave.
    dr : float
        Density ratio across the shock wave.
    tr : float
        Temperature ratio across the shock wave.
    tpr : float
        Total Pressure ratio across the shock wave.

    Examples
    --------

    Compute all ratios across a normal shockwave starting with the upstream
    Mach number:

    >>> from pygasflow import shockwave_solver
    >>> shockwave_solver("m1", 2)
    [2.0, 2.0, 0.5773502691896257, 0.5773502691896257, 90.0, 5.847257748779064e-15, 4.5, 2.666666666666667, 1.6874999999999998, 0.7208738614847455]

    Compute all ratios and parameters across an oblique shockwave starting
    from the shockwave angle and the deflection angle:

    >>> shockwave_solver("theta", 8, "beta", 80)
    [1.511670289641015, 1.4887046212366817, 0.7414131402857721, 0.7051257983356364, 80.0, 7.999999999999998, 2.418948357506694, 1.84271116608139, 1.312711618636739, 0.9333272472012358]

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers:

    >>> results = shockwave_solver("m1", [1.5, 3], "beta", [60, 60])
    >>> print(results[2])
    [1.04454822 1.12256381]

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers, returning a dictionary:

    >>> results = shockwave_solver("m1", [1.5, 3], "beta", [60, 60], to_dict=True)
    >>> print(results["m2"])
    [1.04454822 1.12256381]

    """

    beta, theta = None, None
    MN1, M1 = None, None

    p2_name = p2_name.lower()
    if p2_name not in ['beta', 'theta', 'mn1']:
        raise ValueError("p2_name must be either 'beta' or 'theta' or 'mn1.")
    if p2_value is None:
        raise ValueError("p2_value must be a real positive number.")
    if p2_name == 'beta':
        beta = p2_value
        if (not np.all(beta >= 0)) or (not np.all(beta <= 90)):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
    elif p2_name == 'theta':
        theta = p2_value
        # TODO: is this condition correct? The 0 and 90 part????
        if np.any(theta < 0) or np.any(theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
    else:
        MN1 = p2_value


    p1_name = p1_name.lower()
    available_p1names = ['beta', 'theta', 'pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2']
    if p1_name not in available_p1names:
        raise ValueError("p1_name must be either one of {}".format(available_p1names))
    if p1_name in ['pressure', 'temperature', 'density', 'total_pressure', 'mn2']:
        MN1 = get_upstream_normal_mach_from_ratio.__no_check(p1_name, p1_value, gamma)
    elif p1_name == "mn1":
        if p1_name == p2_name:
            raise ValueError("p1_name must be different than p2_name")
        MN1 = p1_value
    elif p1_name == "m1":
        M1 = p1_value
    elif p1_name == "theta":
        if beta is None:
            raise ValueError("If you provide p1_name='theta', it must be p2_name='beta'.")
        theta = p1_value
        if (theta < 0) or (theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
        if not isinstance(beta, np.ndarray):
            beta = beta * np.ones_like(theta)
        M1 = mach_from_theta_beta.__no_check(theta, beta)
    elif p1_name == "beta":
        if theta is None:
            raise ValueError("If you provide p1_name='beta', it must be p2_name='theta'.")
        beta = p1_value
        if (beta < 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
        if not isinstance(theta, np.ndarray):
            theta = theta * np.ones_like(beta)
        M1 = mach_from_theta_beta.__no_check(theta, beta)
    else:   # 'm2'
        # TODO:
        # Is it even possible to solve it knowing only M2, beta or M2, theta?????
        raise NotImplementedError("Solving a shock wave with a given M2 is not yet implemented.")

    if (M1 is not None) and (MN1 is not None):
        beta = beta_from_upstream_mach.__no_check(M1, MN1)
        theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    elif M1 is not None:
        # at this point, either beta or theta is set, not both!
        MN1 = normal_mach_upstream.__no_check(M1, beta, theta, gamma, flag)
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)

        if beta is not None:
            theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
        else:
            beta = beta_from_mach_theta.__no_check(M1, theta, gamma)[flag]

        if isinstance(M1, (list, tuple, np.ndarray)):
            beta *= np.ones_like(M1)
            theta *= np.ones_like(M1)

        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    else:
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)

        if beta is not None:
            M1 = MN1 / np.sin(np.deg2rad(beta))
            theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
            if isinstance(M1, (list, tuple, np.ndarray)):
                beta *= np.ones_like(M1)
        else:
            # TODO:
            # Is it even possible to uniquely determine M1 = f(MN1, beta)????

            # M1 = Upstream_Mach_From_Normal_Mach_Theta(MN1, theta, flag, gamma)
            # beta = Beta_From_Mach_Theta(M1, theta, gamma)[flag]
            M1 = np.nan * np.ones_like(MN2)
            beta = np.nan * np.ones_like(MN2)
            warnings.warn("Undetermined case. Setting M1 = beta = M2 = NaN")
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))

    # TODO
    # 1. What if p1_name is M2????
    #

    if to_dict:
        return {
            "m1": M1,
            "mn1": MN1,
            "m2": M2,
            "mn2": MN2,
            "beta": beta,
            "theta": theta,
            "pr": pr,
            "dr": dr,
            "tr": tr,
            "tpr": tpr
        }
    return M1, MN1, M2, MN2, beta, theta, pr, dr, tr, tpr

@check_shockwave
def conical_shockwave_solver(M1, param_name, param_value, gamma=1.4, flag="weak", to_dict=False):
    """
    Try to compute all the ratios, angles and mach numbers across the conical shock wave.

    Parameters
    ----------
    M1 : float
        Upstream Mach number. Must be M1 > 1.
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'mc'``: Mach number at the cone's surface.
        * ``'theta_c'``: Half cone angle.
        * ``'beta'``: shock wave angle.

    param_value : float
        Actual value of the parameter. Requirements:

        * Mc >= 0
        * 0 < beta <= 90
        * 0 < theta_c < 90

    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.
    flag : string, optional
        Can be either ``'weak'`` or ``'strong'``. Default to ``'weak'``
        (in conical shockwaves, the strong solution is rarely encountered).
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    m : float
        Upstream Mach number.
    mc : float
        Mach number at the surface of the cone.
    theta_c : float
        Half cone angle.
    beta : float
        Shock wave angle.
    delta : float
        Flow deflection angle.
    pr : float
        Pressure ratio across the shock wave.
    dr : float
        Density ratio across the shock wave.
    tr : float
        Temperature ratio across the shock wave.
    tpr : float
        Total Pressure ratio across the shock wave.
    pc_p1 : float
        Pressure ratio between the cone's surface and the upstream condition.
    rhoc_rho1 : float
        Density ratio between the cone's surface and the upstream condition.
    Tc_T1 : float
        Temperature ratio between the cone's surface and the upstream
        condition.

    Examples
    --------

    Compute all quantities across a conical shockwave starting from the
    upstream Mach number and the half cone angle:

    >>> from pygasflow import conical_shockwave_solver
    >>> conical_shockwave_solver(2.5, "theta_c", 15)
    [2.5, 2.1179295900668067, 15.0, 28.45459370447941, 6.229019180107892, 1.488659699248579, 1.3262664608044694, 1.122443900410184, 0.9936173734022627, 1.8051864085591218, 1.5220731269187135, 1.186005045771711]

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface:

    >>> results = conical_shockwave_solver([2.5, 5], "mc", 1.5)
    >>> print(results[5])
    [ 3.42459174 18.60172442]

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface,
    but returning a dictionary:

    >>> results = conical_shockwave_solver([2.5, 5], "mc", 1.5, to_dict=True)
    >>> print(results["pr"])
    [ 3.42459174 18.60172442]

    """
    param_name = param_name.lower()
    if param_name not in ["mc", "beta", "theta_c"]:
        raise ValueError(
            "param_name can be either 'beta' or 'mc' or 'theta_c'.")

    Mc, beta, theta_c = None, None, None
    if param_name == 'mc':
        Mc = param_value
        if np.any(M1 <= Mc):
            raise ValueError("It must be M1 > Mc.")
        if (not isinstance(Mc, (int, float))) or (Mc < 0):
            raise ValueError(
                "The Mach number at the cone's surface must be Mc >= 0.")
    elif param_name == 'beta':
        beta = param_value
        if (not isinstance(beta, (int, float))) or (beta <= 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 < beta <= 90.")
    else:
        theta_c = param_value
        if (not isinstance(theta_c, (int, float))) or (theta_c <= 0) or (theta_c > 90):
            raise ValueError("The half cone angle must be 0 < theta_c < 90.")

    if Mc:
        _, theta_c, beta = shock_angle_from_machs(M1, Mc, gamma, flag)
    elif beta:
        Mc, theta_c = mach_cone_angle_from_shock_angle(M1, beta, gamma)
    elif theta_c:
        Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag)

    # compute the ratios across the shockwave
    MN1 = M1 * np.sin(np.deg2rad(beta))
    pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)

    # delta is the flow deflection angle (Anderson's Figure 10.4)
    delta = theta_from_mach_beta(M1, beta, gamma)
    M2 = MN2 /  np.sin(np.deg2rad(beta - delta))

    # ratios between cone surface and upstream conditions. Note that
    # p0c/p01 = p02/p01, already computed
    pc_p1 = ise_PR(Mc) * tpr / ise_PR(M1)
    rhoc_rho1 = ise_DR(Mc) / ise_DR(M2) * dr
    Tc_T1 = ise_TR(Mc) / ise_TR(M2) * tr

    # set Mc, theta_c to have the same shape as M1 and the other ratios. This is
    # necessary because Mc or theta_c are parameters passed in by the user, in
    # that case they are scalars.
    theta_c = theta_c * np.ones_like(M1)
    if not isinstance(Mc, np.ndarray):
        Mc = Mc * np.ones_like(M1)

    if to_dict:
        return {
            "m": M1,
            "mc": Mc,
            "theta_c": theta_c,
            "beta": beta,
            "delta": delta,
            "pr": pr,
            "dr": dr,
            "tr": tr,
            "tpr": tpr,
            "pc_p1": pc_p1,
            "rhoc_rho1": rhoc_rho1,
            "Tc_T1": Tc_T1
        }
    return M1, Mc, theta_c, beta, delta, pr, dr, tr, tpr, pc_p1, rhoc_rho1, Tc_T1
