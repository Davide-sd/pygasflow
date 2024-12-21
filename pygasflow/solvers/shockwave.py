import numpy as np
import warnings
from numbers import Number
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
from pygasflow.utils.common import (
    convert_to_ndarray,
    ShockResults,
    _should_solver_return_dict,
    _print_results_helper,
)
from pygasflow.utils.decorators import check_shockwave


@check_shockwave([1, 3])
def oblique_shockwave_solver(
    p1_name, p1_value, p2_name="beta", p2_value=90, gamma=1.4,
    flag="weak", to_dict=None
):
    """
    Try to compute all the ratios, angles and mach numbers across the shock wave.

    An alias of this function is `shockwave_solver`.

    Parameters
    ----------
    p1_name : string
        Name of the first parameter given in input. Can be either one of:

        * ``'pressure'``: Pressure Ratio P2/P1
        * ``'temperature'``: Temperature Ratio T2/T1
        * ``'density'``: Density Ratio rho2/rho1
        * ``'total_pressure'``: Total Pressure Ratio P02/P01
        * ``'mu'``: upstream Mach number of the shock wave
        * ``'mnu'``: upstream normal Mach number of the shock wave
        * ``'mnd'``: downstream normal Mach number of the shock wave
        * ``'beta'``: The shock wave angle [in degrees]. It can only be used
          if ``p2_name='theta'``.
        * ``'theta'``: The deflection angle [in degrees]. It can only be
          used if ``p2_name='beta'``.

        If the parameter is a ratio, it is in the form downstream/upstream.

    p1_value : float
        Actual value of the parameter.
    p2_name : string, optional
        Name of the second parameter. It could either be:

        * ``'beta'``: Shock wave angle.
        * ``'theta'``: Flow deflection angle.
        * ``'mnu'``: upstream normal Mach number.

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
    mu : float
        Mach number upstream of the shock wave.
    mnu : float
        Normal Mach number upstream of the shock wave.
    md : float
        Mach number downstream of the shock wave.
    mnd : float
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

    See Also
    --------
    :class:`~pygasflow.interactive.diagrams.oblique_shock.ObliqueShockDiagram`,
    print_oblique_shockwave_results

    Examples
    --------

    Compute all ratios across a weak oblique shockwave starting with the
    upstream Mach number and the deflection angle, using air:

    >>> from pygasflow.solvers import (
    ...     oblique_shockwave_solver,
    ...     print_oblique_shockwave_results
    ... )
    >>> res = oblique_shockwave_solver("mu", 2, "theta", 15, flag="weak")
    >>> type(res)
    <class 'list'>
    >>> res
    [np.float64(2.0), np.float64(1.42266946274781), np.float64(1.4457163651405158), np.float64(0.7303538499327245), np.float64(45.343616761854385), np.float64(15.0), np.float64(2.1946531336076665), np.float64(1.7289223315067423), np.float64(1.2693763586794804), np.float64(0.9523563236996431)]
    >>> print_oblique_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          2.0
    Mnu         1.42266946274781
    Md          1.4457163651405158
    Mnd         0.7303538499327245
    beta        45.343616761854385
    theta       15.0
    pd/pu       2.1946531336076665
    rhod/rhou   1.7289223315067423
    Td/Tu       1.2693763586794804
    p0d/p0u     0.9523563236996431

    Using the same parameters, but computing the solution across a
    strong oblique shock wave:

    >>> res = oblique_shockwave_solver("mu", 2, "theta", 15, flag="strong")
    >>> print_oblique_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          2.0
    Mnu         1.9685867877725771
    Md          0.643970917523609
    Mnd         0.5828338581186796
    beta        79.83168734049043
    theta       15.0
    pd/pu       4.354556264491546
    rhod/rhou   2.6198454954536237
    Td/Tu       1.6621423942932017
    p0d/p0u     0.7355379974194783

    Compute all ratios and parameters across an oblique shockwave starting
    from the shock wave angle and the deflection angle, using methane at 20°C
    as the fluid:

    >>> res = oblique_shockwave_solver("theta", 8, "beta", 80, gamma=1.32)
    >>> print_oblique_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          1.4819290082790446
    Mnu         1.4594151767668957
    Md          0.7477053570397926
    Mnd         0.711110052081489
    beta        80.0
    theta       8.000000000000002
    pd/pu       2.2857399213744523
    rhod/rhou   1.8427111660813902
    Td/Tu       1.240422244922509
    p0d/p0u     0.9398367786738993

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers:

    >>> res = oblique_shockwave_solver("mu", [1.5, 3], "beta", 60)
    >>> print(res[2])
    [1.04454822 1.12256381]

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers, returning a dictionary:

    >>> res = oblique_shockwave_solver("mu", [1.5, 3], "beta", 60, to_dict=True)
    >>> type(res)
    <class 'pygasflow.utils.common.ShockResults'>
    >>> print(res["md"])
    [1.04454822 1.12256381]

    This function is capable of detecting detachment:

    >>> oblique_shockwave_solver("mu", 2, 'theta', 30)
    Traceback (most recent call last):
     ...
    ValueError: Detachment detected: can't solve the flow when theta > theta_max.
    M1 = [2.]
    theta_max(M1) = 22.97353176093536
    theta = [30.]

    """
    to_dict = _should_solver_return_dict(to_dict)

    if not isinstance(gamma, Number):
        raise ValueError("The specific heats ratio must be > 1.")
    beta, theta = None, None
    MN1, M1 = None, None

    def _check_name(name):
        # deprecate m1,mn1,m2,mn2
        if name in ["m1", "mn1", "m2", "mn2"]:
            warnings.warn(
                f"Key '{name}' is deprecated and will be removed in the future."
                f" Use '{ShockResults.deprecation_map[name]}' instead.",
                stacklevel=1
            )
            return ShockResults.deprecation_map[name]
        return name

    p1_name = _check_name(p1_name.lower())
    p2_name = _check_name(p2_name.lower())

    if p2_name not in ['beta', 'theta', 'mnu']:
        raise ValueError("p2_name must be either 'beta' or 'theta' or 'mnu'.")
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
    available_p1names = [
        'beta', 'theta', 'pressure', 'temperature', 'density',
        'total_pressure', 'mu', 'mnu', 'mnd']
    if p1_name not in available_p1names:
        raise ValueError(
            f"p1_name must be either one of {available_p1names}."
            f" Instead, '{p1_name}' was received."
        )
    if p1_name in ['pressure', 'temperature', 'density', 'total_pressure', 'mnd']:
        MN1 = get_upstream_normal_mach_from_ratio.__no_check__(p1_name, p1_value, gamma)
    elif p1_name == "mnu":
        if p1_name == p2_name:
            raise ValueError("p1_name must be different than p2_name")
        MN1 = p1_value
    elif p1_name == "mu":
        M1 = p1_value
    elif p1_name == "theta":
        if beta is None:
            raise ValueError("If you provide p1_name='theta', it must be p2_name='beta'.")
        theta = p1_value
        if (theta < 0) or (theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
        if not isinstance(beta, np.ndarray):
            beta = beta * np.ones_like(theta)
        M1 = mach_from_theta_beta.__no_check__(theta, beta, gamma)
    elif p1_name == "beta":
        if theta is None:
            raise ValueError("If you provide p1_name='beta', it must be p2_name='theta'.")
        beta = p1_value
        if (beta < 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
        if not isinstance(theta, np.ndarray):
            theta = theta * np.ones_like(beta)
        M1 = mach_from_theta_beta.__no_check__(theta, beta, gamma)
    else:   # 'm2'
        # TODO:
        # Is it even possible to solve it knowing only M2, beta or M2, theta?????
        raise NotImplementedError("Solving a shock wave with a given M2 is not yet implemented.")

    if (M1 is not None) and (MN1 is not None):
        beta = beta_from_upstream_mach.__no_check__(M1, MN1)
        theta = theta_from_mach_beta.__no_check__(M1, beta, gamma)
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check__(MN1, gamma)
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    elif M1 is not None:
        # at this point, either beta or theta is set, not both!
        MN1 = normal_mach_upstream.__no_check__(M1, beta, theta, gamma, flag)
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check__(MN1, gamma)

        if beta is not None:
            theta = theta_from_mach_beta.__no_check__(M1, beta, gamma)
        else:
            beta = beta_from_mach_theta.__no_check__(M1, theta, gamma)[flag]

        if isinstance(M1, (list, tuple, np.ndarray)):
            if hasattr(beta, "__iter__") and (len(beta) == 1):
                beta = beta[0]
            if hasattr(theta, "__iter__") and (len(theta) == 1):
                theta = theta[0]
            beta *= np.ones_like(M1)
            theta *= np.ones_like(M1)

        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    else:
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check__(MN1, gamma)

        if beta is not None:
            M1 = MN1 / np.sin(np.deg2rad(beta))
            theta = theta_from_mach_beta.__no_check__(M1, beta, gamma)
            if isinstance(M1, (list, tuple, np.ndarray)):
                beta = beta * np.ones_like(M1)
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
        return ShockResults(
            mu=M1,
            mnu=MN1,
            md=M2,
            mnd=MN2,
            beta=beta,
            theta=theta,
            pr=pr,
            dr=dr,
            tr=tr,
            tpr=tpr
        )
    return M1, MN1, M2, MN2, beta, theta, pr, dr, tr, tpr


shockwave_solver = oblique_shockwave_solver


def normal_shockwave_solver(param_name, param_value, gamma=1.4, to_dict=None):
    """
    Compute all the ratios across a normal shock wave.

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'pressure'``: Pressure Ratio P2/P1
        * ``'temperature'``: Temperature Ratio T2/T1
        * ``'density'``: Density Ratio rho2/rho1
        * ``'total_pressure'``: Total Pressure Ratio P02/P01
        * ``'mu'``: upstream Mach number of the shock wave
        * ``'md'``: downstream Mach number of the shock wave

        If the parameter is a ratio, it is in the form downstream/upstream.

    param_value : float
        Actual value of the parameter.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    mu : float
        Mach number upstream of the shock wave.
    md : float
        Mach number downstream of the shock wave.
    pr : float
        Pressure ratio across the shock wave.
    dr : float
        Density ratio across the shock wave.
    tr : float
        Temperature ratio across the shock wave.
    tpr : float
        Total Pressure ratio across the shock wave.

    See Also
    --------
    :class:`~pygasflow.interactive.diagrams.normal_shock.NormalShockDiagram`,
    print_normal_shockwave_results

    Examples
    --------

    Compute all ratios across a normal shockwave starting with the upstream
    Mach number, using air:

    >>> from pygasflow.solvers import (
    ...     normal_shockwave_solver,
    ...     print_normal_shockwave_results
    ... )
    >>> res = normal_shockwave_solver("mu", 2)
    >>> type(res)
    <class 'list'>
    >>> print_normal_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          2.0
    Md          0.5773502691896257
    pd/pu       4.5
    rhod/rhou   2.666666666666667
    Td/Tu       1.6874999999999998
    p0d/p0u     0.7208738614847455

    Compute all ratios and parameters across a normal shockwave starting
    from the downstream Mach, using methane at 20°C:

    >>> res = normal_shockwave_solver("md", 0.4, gamma=1.32)
    >>> print_normal_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          4.4756284474920385
    Md          0.4000000000000001
    pd/pu       22.65624999999999
    rhod/rhou   5.525862068965516
    Td/Tu       4.100039001560062
    p0d/p0u     0.06721056989701328

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers, returning a dictionary:

    >>> res = normal_shockwave_solver("mu", [1.5, 3], to_dict=True)
    >>> type(res)
    <class 'pygasflow.utils.common.ShockResults'>
    >>> print(res["md"])
    [0.70108874 0.47519096]

    """
    to_dict = _should_solver_return_dict(to_dict)
    if param_name in ["m2", "M2", "md", "Md"]:
        param_name = "mnd"
    results = oblique_shockwave_solver(param_name, param_value, "beta", 90,
        gamma=gamma, to_dict=to_dict)
    if not to_dict:
        idx_to_exclude = [1, 3, 4, 5]
        return [r for i, r in enumerate(results) if i not in idx_to_exclude]
    for k in ["mnu", "mnd", "beta", "theta"]:
        results.pop(k)
    return results


@check_shockwave
def conical_shockwave_solver(Mu, param_name, param_value, gamma=1.4, flag="weak", to_dict=None):
    """
    Try to compute all the ratios, angles and mach numbers across the conical shock wave.

    Parameters
    ----------
    Mu : float
        Upstream Mach number. Must be Mu > 1.
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
    mu : float
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
    pc_pu : float
        Pressure ratio between the cone's surface and the upstream condition.
    rhoc_rhou : float
        Density ratio between the cone's surface and the upstream condition.
    Tc_Tu : float
        Temperature ratio between the cone's surface and the upstream
        condition.

    See Also
    --------
    :class:`~pygasflow.interactive.diagrams.conical_shock.ConicalShockDiagram`,
    print_conical_shockwave_results

    Examples
    --------

    Compute all quantities across a conical shockwave starting from the
    upstream Mach number and the half cone angle:

    >>> from pygasflow.solvers import (
    ...     conical_shockwave_solver,
    ...     print_conical_shockwave_results
    ... )
    >>> res = conical_shockwave_solver(2.5, "theta_c", 15)
    >>> type(res)
    <class 'list'>
    >>> print_conical_shockwave_results(res)    # doctest: +NORMALIZE_WHITESPACE
    Mu          2.5
    Mc          2.1179295900667547
    theta_c     15.0
    beta        28.454593704480512
    delta       6.2290191801091845
    pd/pu       1.488659699248697
    rhod/rhou   1.326266460804543
    Td/Tu       1.122443900410211
    p0d/p0u     0.9936173734022588
    pc/pu       1.8051864085592608
    rho_c/rhou  1.5220731269187957
    Tc/Tu       1.186005045771739

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface:

    >>> res = conical_shockwave_solver([2.5, 5], "mc", 1.5)
    >>> print(res[5])
    [ 3.42459174 18.60172442]

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface,
    but returning a dictionary:

    >>> res = conical_shockwave_solver([2.5, 5], "mc", 1.5, to_dict=True)
    >>> type(res)
    <class 'pygasflow.utils.common.ShockResults'>
    >>> print(res["pr"])
    [ 3.42459174 18.60172442]

    This function is capable of detecting detachment:

    >>> conical_shockwave_solver(2, 'theta_c', 45)
    Traceback (most recent call last):
     ...
    ValueError: Detachment detected: can't solve the flow when theta_c > theta_c_max.
    M1 = [2.]
    theta_c_max(M1) = 40.68847689093214
    theta_c = 45

    """
    to_dict = _should_solver_return_dict(to_dict)
    if not isinstance(gamma, Number):
        raise ValueError("The specific heats ratio must be > 1.")

    param_name = param_name.lower()
    if param_name not in ["mc", "beta", "theta_c"]:
        raise ValueError(
            "param_name can be either 'beta' or 'mc' or 'theta_c'.")

    Mc, beta, theta_c = None, None, None
    if param_name == 'mc':
        Mc = param_value
        if np.any(Mu <= Mc):
            raise ValueError("It must be Mu > Mc.")
        if (not isinstance(Mc, Number)) or (Mc < 0):
            raise ValueError(
                "The Mach number at the cone's surface must be Mc >= 0.")
    elif param_name == 'beta':
        beta = param_value
        if (not isinstance(beta, Number)) or (beta <= 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 < beta <= 90.")
    else:
        theta_c = param_value
        if (not isinstance(theta_c, Number)) or (theta_c <= 0) or (theta_c > 90):
            raise ValueError("The half cone angle must be 0 < theta_c < 90.")

    if Mc:
        _, theta_c, beta = shock_angle_from_machs(Mu, Mc, gamma, flag)
    elif beta:
        Mc, theta_c = mach_cone_angle_from_shock_angle(Mu, beta, gamma)
    elif theta_c:
        Mc, _, beta = shock_angle_from_mach_cone_angle(Mu, theta_c, gamma, flag)

    # compute the ratios across the shockwave
    MN1 = Mu * np.sin(np.deg2rad(beta))
    pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)

    # delta is the flow deflection angle (Anderson's Figure 10.4)
    delta = theta_from_mach_beta(Mu, beta, gamma)
    M2 = MN2 /  np.sin(np.deg2rad(beta - delta))

    # ratios between cone surface and upstream conditions. Note that
    # p0c/p01 = p02/p01, already computed
    pc_p1 = ise_PR(Mc) * tpr / ise_PR(Mu)
    rhoc_rho1 = ise_DR(Mc) / ise_DR(M2) * dr
    Tc_T1 = ise_TR(Mc) / ise_TR(M2) * tr

    # set Mc, theta_c to have the same shape as Mu and the other ratios. This is
    # necessary because Mc or theta_c are parameters passed in by the user, in
    # that case they are scalars.
    theta_c = theta_c * np.ones_like(Mu)
    if not isinstance(Mc, np.ndarray):
        Mc = Mc * np.ones_like(Mu)

    if to_dict:
        return ShockResults(
            mu=Mu,
            mc=Mc,
            theta_c=theta_c,
            beta=beta,
            delta=delta,
            pr=pr,
            dr=dr,
            tr=tr,
            tpr=tpr,
            pc_pu=pc_p1,
            rhoc_rhou=rhoc_rho1,
            Tc_Tu=Tc_T1
        )
    return Mu, Mc, theta_c, beta, delta, pr, dr, tr, tpr, pc_p1, rhoc_rho1, Tc_T1


def print_normal_shockwave_results(results, formatter="{}", blank_line=False):
    """
    Parameters
    ----------
    results : list or dict
    formatter : str
        A formatter to properly show floating point numbers. For example,
        ``"{:.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    normal_shockwave_solver
    """
    data = results.values() if isinstance(results, dict) else results
    labels = ["Mu", "Md", "pd/pu", "rhod/rhou", "Td/Tu", "p0d/p0u"]
    _print_results_helper(data, labels, "{:12}", formatter, blank_line)


def print_oblique_shockwave_results(results, formatter="{}", blank_line=False):
    """
    Parameters
    ----------
    results : list or dict
    formatter : str
        A formatter to properly show floating point numbers. For example,
        ``"{:.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    oblique_shockwave_solver
    """
    data = results.values() if isinstance(results, dict) else results
    labels = ["Mu", "Mnu", "Md", "Mnd", "beta", "theta", "pd/pu",
        "rhod/rhou", "Td/Tu", "p0d/p0u"]
    _print_results_helper(data, labels, "{:12}", formatter, blank_line)


def print_conical_shockwave_results(results, formatter="{}", blank_line=False):
    """
    Parameters
    ----------
    results : list or dict
    formatter : str
        A formatter to properly show floating point numbers. For example,
        ``"{:.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    conical_shockwave_solver
    """
    data = results.values() if isinstance(results, dict) else results
    labels = ["Mu", "Mc", "theta_c", "beta", "delta", "pd/pu", "rhod/rhou",
        "Td/Tu", "p0d/p0u", "pc/pu", "rho_c/rhou", "Tc/Tu"]
    _print_results_helper(data, labels, "{:12}", formatter, blank_line)
