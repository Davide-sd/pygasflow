import numpy as np
import warnings
from numbers import Number
import pygasflow
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
    mach_beta_from_theta_ratio,
    mach_downstream,
)
from pygasflow.utils.common import (
    ShockResults,
    ShockResultsList,
    _should_solver_return_dict,
    _print_results_helper,
    _is_pint_quantity
)
from pygasflow.utils.decorators import check_shockwave


@check_shockwave([1, 3])
def oblique_shockwave_solver(
    p1_name, p1_value, p2_name="beta", p2_value=90, gamma=1.4,
    flag="weak", to_dict=None
):
    """
    Attempt to compute all the ratios, angles and mach numbers across the 
    shock wave.

    An alias of this function is `shockwave_solver`.

    Parameters
    ----------
    p1_name : string
        Name of the first parameter given in input. In the following
        ratios, `d` stands for downstream of the shockwave while `u` stand 
        for upstream of the shockwave. Can be either one of:

        * ``'pressure'``: Pressure Ratio Pd / Pu
        * ``'temperature'``: Temperature Ratio Td / Tu
        * ``'density'``: Density Ratio rhod / rhou
        * ``'total_pressure'``: Total Pressure Ratio P0d / P0u
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

    >>> from pygasflow.solvers import oblique_shockwave_solver
    >>> res = oblique_shockwave_solver("mu", 2, "theta", 15, flag="weak")
    >>> isinstance(res, list)
    True
    >>> res
    [np.float64(2.0), np.float64(1.42266946274781), np.float64(1.4457163651405158), np.float64(0.7303538499327245), np.float64(45.343616761854385), np.float64(15.0), np.float64(2.1946531336076665), np.float64(1.7289223315067423), np.float64(1.2693763586794804), np.float64(0.9523563236996431)]
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                2.00000000
    1     Mnu               1.42266946
    2     Md                1.44571637
    3     Mnd               0.73035385
    4     beta             45.34361676
    5     theta            15.00000000
    6     pd/pu             2.19465313
    7     rhod/rhou         1.72892233
    8     Td/Tu             1.26937636
    9     p0d/p0u           0.95235632

    Using the same parameters, but computing the solution across a
    strong oblique shock wave:

    >>> res = oblique_shockwave_solver("mu", 2, "theta", 15, flag="strong")
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                2.00000000
    1     Mnu               1.96858679
    2     Md                0.64397092
    3     Mnd               0.58283386
    4     beta             79.83168734
    5     theta            15.00000000
    6     pd/pu             4.35455626
    7     rhod/rhou         2.61984550
    8     Td/Tu             1.66214239
    9     p0d/p0u           0.73553800

    Compute all ratios and parameters across an oblique shockwave starting
    from the shock wave angle and the deflection angle, using methane at 20°C
    as the fluid:

    >>> res = oblique_shockwave_solver("theta", 8, "beta", 80, gamma=1.32)
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                1.48192901
    1     Mnu               1.45941518
    2     Md                0.74770536
    3     Mnd               0.71111005
    4     beta             80.00000000
    5     theta             8.00000000
    6     pd/pu             2.28573992
    7     rhod/rhou         1.84271117
    8     Td/Tu             1.24042224
    9     p0d/p0u           0.93983678

    Compute all ratios and parameters across an oblique shockwave starting
    from some ratio and the deflection angle. This mode of operation
    computes two different solutions: depending on the parameters, one solution
    could be in the strong region, the other in the weak region.
    Other times, both solutions could be in the weak region.
    Hence, the ``flag`` keyword argument is not used by this mode of operation:

    >>> res = oblique_shockwave_solver("pressure", 4.5, "theta", 20, gamma=1.4)
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                2.06488358     3.53991435
    1     Mnu               2.00000000     2.00000000
    2     Md                0.69973294     2.32136532
    3     Mnd               0.57735027     0.57735027
    4     beta             75.59872102    34.40127898
    5     theta            20.00000000    20.00000000
    6     pd/pu             4.50000000     4.50000000
    7     rhod/rhou         2.66666667     2.66666667
    8     Td/Tu             1.68750000     1.68750000
    9     p0d/p0u           0.72087386     0.72087386

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers:

    >>> res = oblique_shockwave_solver("mu", [1.5, 3], "beta", 60)
    >>> res.show()
    idx   quantity     
    -------------------
    0     Mu                1.50000000     3.00000000
    1     Mnu               1.29903811     2.59807621
    2     Md                1.04454822     1.12256381
    3     Mnd               0.78644587     0.50403775
    4     beta             60.00000000    60.00000000
    5     theta            11.15732683    33.32007997
    6     pd/pu             1.80208333     7.70833333
    7     rhod/rhou         1.51401869     3.44680851
    8     Td/Tu             1.19026492     2.23636831
    9     p0d/p0u           0.97953995     0.46084919
    >>> print(res[2])
    [1.04454822 1.12256381]

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers, returning a dictionary:

    >>> res = oblique_shockwave_solver("mu", [1.5, 3], "beta", 60, to_dict=True)
    >>> isinstance(res, dict)
    True
    >>> res.show()
    key     quantity     
    ---------------------
    mu      Mu                1.50000000     3.00000000
    mnu     Mnu               1.29903811     2.59807621
    md      Md                1.04454822     1.12256381
    mnd     Mnd               0.78644587     0.50403775
    beta    beta             60.00000000    60.00000000
    theta   theta            11.15732683    33.32007997
    pr      pd/pu             1.80208333     7.70833333
    dr      rhod/rhou         1.51401869     3.44680851
    tr      Td/Tu             1.19026492     2.23636831
    tpr     p0d/p0u           0.97953995     0.46084919
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
    is_pint = False

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
        if _is_pint_quantity(beta):
            is_pint = True
            beta = beta.to("deg").magnitude
        if (not np.all(beta >= 0)) or (not np.all(beta <= 90)):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
    elif p2_name == 'theta':
        theta = p2_value
        if _is_pint_quantity(theta):
            is_pint = True
            theta = theta.to("deg").magnitude
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
    if p1_name not in ["beta", "theta"]:
        if _is_pint_quantity(p1_value):
            is_pint = True
            p1_value = p1_value.magnitude
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
        if _is_pint_quantity(theta):
            is_pint = True
            theta = theta.to("deg").magnitude
        if (theta < 0) or (theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
        if not isinstance(beta, np.ndarray):
            beta = beta * np.ones_like(theta)
        M1 = mach_from_theta_beta.__no_check__(theta, beta, gamma)
    elif p1_name == "beta":
        if theta is None:
            raise ValueError("If you provide p1_name='beta', it must be p2_name='theta'.")
        beta = p1_value
        if _is_pint_quantity(beta):
            is_pint = True
            beta = beta.to("deg").magnitude
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
            pr, dr, tr, tpr, MN2, theta = [
                np.atleast_1d(t) for t in [pr, dr, tr, tpr, MN2, theta]]
            if (len(pr) > 1) and (len(theta) == 1):
                theta = theta[0] * np.ones_like(pr)
            elif (len(pr) == 1) and (len(theta) > 1):
                pr, dr, tr, tpr, MN2 = [
                    t[0] * np.ones_like(theta) for t in [pr, dr, tr, tpr, MN2]
                ]

            M1_list, Mn1_list, Mn2_list = [], [], []
            beta_list, theta_list = [], []
            pr_list, dr_list, tr_list, tpr_list = [], [], [], []

            for i in range(len(pr)):
                m1, beta = mach_beta_from_theta_ratio(
                    theta[i], "pressure", pr[i], gamma)
                pr_list += [pr[i]] * len(m1)
                dr_list += [dr[i]] * len(m1)
                tr_list += [tr[i]] * len(m1)
                tpr_list += [tpr[i]] * len(m1)
                theta_list += [theta[i]] * len(m1)
                Mn1 = [
                    mu * np.sin(np.deg2rad(b)) for mu, b in zip(m1, beta)]
                Mn1_list += Mn1
                Mn2_list += [
                    mach_downstream.__no_check__(mn, gamma) for mn in Mn1
                ]
                M1_list += m1
                beta_list += beta

            pr, dr, tr, tpr, theta, beta, M1, MN1, MN2 = [
                np.asarray(t) for t in [
                    pr_list, dr_list, tr_list, tpr_list,
                    theta_list, beta_list, M1_list, Mn1_list, Mn2_list]]

        M2 = MN2 / np.sin(np.deg2rad(beta - theta))

    # TODO
    # 1. What if p1_name is M2????
    #
    if is_pint:
        deg = pygasflow.defaults.pint_ureg.deg
        beta *= deg
        theta *= deg

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
            tpr=tpr,
            printer=print_oblique_shockwave_results
        )
    return ShockResultsList(
        [M1, MN1, M2, MN2, beta, theta, pr, dr, tr, tpr],
        printer=print_oblique_shockwave_results
    )


shockwave_solver = oblique_shockwave_solver


def normal_shockwave_solver(param_name, param_value, gamma=1.4, to_dict=None):
    """
    Compute all the ratios across a normal shock wave.

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. In the following
        ratios, `d` stands for downstream of the shockwave while `u` stand 
        for upstream of the shockwave. Can be either one of:

        * ``'pressure'``: Pressure Ratio Pd / Pu
        * ``'temperature'``: Temperature Ratio Td / Tu
        * ``'density'``: Density Ratio rhod / rhou
        * ``'total_pressure'``: Total Pressure Ratio P0d / P0u
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

    >>> from pygasflow.solvers import normal_shockwave_solver
    >>> res = normal_shockwave_solver("mu", 2)
    >>> isinstance(res, list)
    True
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                2.00000000
    1     Md                0.57735027
    2     pd/pu             4.50000000
    3     rhod/rhou         2.66666667
    4     Td/Tu             1.68750000
    5     p0d/p0u           0.72087386

    Compute all ratios and parameters across a normal shockwave starting
    from the downstream Mach number, using methane at 20°C:

    >>> res = normal_shockwave_solver("md", 0.4, gamma=1.32)
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                4.47562845
    1     Md                0.40000000
    2     pd/pu            22.65625000
    3     rhod/rhou         5.52586207
    4     Td/Tu             4.10003900
    5     p0d/p0u           0.06721057

    Compute the Mach number downstream of an oblique shockwave starting with
    multiple upstream Mach numbers, returning a dictionary:

    >>> res = normal_shockwave_solver("mu", [1.5, 3], to_dict=True)
    >>> isinstance(res, dict)
    True
    >>> res.show()
    key     quantity     
    ---------------------
    mu      Mu                1.50000000     3.00000000
    md      Md                0.70108874     0.47519096
    pr      pd/pu             2.45833333    10.33333333
    dr      rhod/rhou         1.86206897     3.85714286
    tr      Td/Tu             1.32021605     2.67901235
    tpr     p0d/p0u           0.92978651     0.32834389
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
        short_results = ShockResultsList(
            [r for i, r in enumerate(results) if i not in idx_to_exclude],
            printer=print_normal_shockwave_results
        )
        return short_results
    for k in ["mnu", "mnd", "beta", "theta"]:
        results.pop(k)
    results.printer = print_normal_shockwave_results
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
        Pressure ratio across the shock wave, Pd / Pu.
    dr : float
        Density ratio across the shock wave, rhod / rhou.
    tr : float
        Temperature ratio across the shock wave, Td / Tu.
    tpr : float
        Total Pressure ratio across the shock wave, P0d / P0u.
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

    >>> from pygasflow.solvers import conical_shockwave_solver
    >>> res = conical_shockwave_solver(2.5, "theta_c", 15)
    >>> isinstance(res, list)
    True
    >>> res.show()    # doctest: +NORMALIZE_WHITESPACE
    idx   quantity     
    -------------------
    0     Mu                2.50000000
    1     Mc                2.11792959
    2     theta_c          15.00000000
    3     beta             28.45459370
    4     delta             6.22901918
    5     pd/pu             1.48865970
    6     rhod/rhou         1.32626646
    7     Td/Tu             1.12244390
    8     p0d/p0u           0.99361737
    9     pc/pu             1.80518641
    10    rho_c/rhou        1.52207313
    11    Tc/Tu             1.18600505

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface:

    >>> res = conical_shockwave_solver([2.5, 5], "mc", 1.5)
    >>> res.show()
    idx   quantity     
    -------------------
    0     Mu                2.50000000     5.00000000
    1     Mc                1.50000000     1.50000000
    2     theta_c          31.69943179    44.78663038
    3     beta             44.57115023    53.33848167
    4     delta            21.26058010    36.98108963
    5     pd/pu             3.42459174    18.60172442
    6     rhod/rhou         2.28631129     4.57733550
    7     Td/Tu             1.49786766     4.06387612
    8     p0d/p0u           0.83262941     0.13748699
    9     pc/pu             3.87527523    19.81540542
    10    rho_c/rhou        2.49739959     4.78872298
    11    Tc/Tu             1.55172414     4.13793103
    >>> print(res[5])
    [ 3.42459174 18.60172442]

    Compute the pressure ratio across a conical shockwave starting with
    multiple upstream Mach numbers and Mach numbers at the cone surface,
    but returning a dictionary:

    >>> res = conical_shockwave_solver([2.5, 5], "mc", 1.5, to_dict=True)
    >>> isinstance(res, dict)
    True
    >>> res.show()
    key        quantity     
    ------------------------
    mu         Mu                2.50000000     5.00000000
    mc         Mc                1.50000000     1.50000000
    theta_c    theta_c          31.69943179    44.78663038
    beta       beta             44.57115023    53.33848167
    delta      delta            21.26058010    36.98108963
    pr         pd/pu             3.42459174    18.60172442
    dr         rhod/rhou         2.28631129     4.57733550
    tr         Td/Tu             1.49786766     4.06387612
    tpr        p0d/p0u           0.83262941     0.13748699
    pc_pu      pc/pu             3.87527523    19.81540542
    rhoc_rhou  rho_c/rhou        2.49739959     4.78872298
    Tc_Tu      Tc/Tu             1.55172414     4.13793103
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
    is_pint = False
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
        if _is_pint_quantity(beta):
            is_pint = True
            beta = beta.to("deg").magnitude
        if (not isinstance(beta, Number)) or (beta <= 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 < beta <= 90.")
    else:
        theta_c = param_value
        if _is_pint_quantity(theta_c):
            is_pint = True
            theta_c = theta_c.to("deg").magnitude
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

    if is_pint:
        deg = pygasflow.defaults.pint_ureg.deg
        beta *= deg
        theta_c *= deg
        delta *= deg

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
            Tc_Tu=Tc_T1,
            printer=print_conical_shockwave_results
        )
    return ShockResultsList(
        [Mu, Mc, theta_c, beta, delta, pr, dr, tr, tpr, pc_p1, rhoc_rho1, Tc_T1],
        printer=print_conical_shockwave_results
    )


def print_normal_shockwave_results(
    results, number_formatter=None, blank_line=False
):
    """
    Parameters
    ----------
    results : list or dict
    number_formatter : str or None
        A formatter to properly show floating point numbers. For example,
        ``"{:>8.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    normal_shockwave_solver
    """
    labels = ["Mu", "Md", "pd/pu", "rhod/rhou", "Td/Tu", "p0d/p0u"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def print_oblique_shockwave_results(
    results, number_formatter=None, blank_line=False
):
    """
    Parameters
    ----------
    results : list or dict
    number_formatter : str or None
        A formatter to properly show floating point numbers. For example,
        ``"{:>8.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    oblique_shockwave_solver
    """
    labels = ["Mu", "Mnu", "Md", "Mnd", "beta", "theta", "pd/pu",
        "rhod/rhou", "Td/Tu", "p0d/p0u"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def print_conical_shockwave_results(
    results, number_formatter=None, blank_line=False
):
    """
    Parameters
    ----------
    results : list or dict
    number_formatter : str or None
        A formatter to properly show floating point numbers. For example,
        ``"{:>8.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    conical_shockwave_solver
    """
    labels = ["Mu", "Mc", "theta_c", "beta", "delta", "pd/pu", "rhod/rhou",
        "Td/Tu", "p0d/p0u", "pc/pu", "rho_c/rhou", "Tc/Tu"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def shock_compression(pr=None, dr=None, gamma=1.4, to_dict=None):
    """
    Solve the Hugoniot equation for a calorically perfect gas in order to
    compute either the pressure ratio P2/P1 or the density ratio rho2/rho1
    across the schock wave.

    Parameters
    ----------
    pr : None, float or array_like
        Pressure ration Pd / Pu across the shock wave. If None, ``dr`` must be
        provided instead.
    dr : None, float or array_like
        Density ration rhod / rhou across the shock wave. If None, ``pr`` must be
        provided instead.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    pr : float or ndarray
        Pressure ratio Pd / Pu across the shock wave.
    dr : float or ndarray
        Density ratio rhod / rhou across the shock wave.
    
    Examples
    --------

    >>> from pygasflow import shock_compression
    >>> rho1 = 1   # kg / m**3
    >>> res = shock_compression(pr=2, to_dict=True)
    >>> res.show()
    key     quantity            
    ----------------------------
    pr      P2 / P1                  2.00000000
    dr      rho2 / rho1              1.62500000
    >>> rho2_rho1 = res["dr"]
    >>> rho2 = rho2_rho1 * rho1
    >>> rho2
    np.float64(1.6250000000000002)

    See Also
    --------
    :func:`~pygasflow.solvers.isentropic.isentropic_compression`

    References
    ----------
    Anderson's, last equation of section 3.7.
    """
    to_dict = _should_solver_return_dict(to_dict)

    a = (gamma + 1) / (gamma - 1)
    if pr is not None:
        is_scalar = isinstance(pr, Number)
        pr = np.atleast_1d(pr)
        v1_v2 = (pr * a + 1) / (pr + a)
        dr = v1_v2
    elif dr is not None:
        is_scalar = isinstance(dr, Number)
        dr = np.atleast_1d(dr)
        v1_v2 = dr
        pr = (a * v1_v2 - 1) / (a - v1_v2)
        pr[a <= v1_v2] = np.nan
    else:
        raise ValueError("Either `pr` or `dr` must be numerical values.")

    if is_scalar:
        pr, dr = pr[0], dr[0]

    if to_dict:
        return ShockResults(
            pr=pr, dr=dr,
            printer=print_shock_compression_results
        )
    return ShockResultsList(
        [pr, dr],
        printer=print_shock_compression_results
    )


def print_shock_compression_results(results, number_formatter=None, blank_line=False):
    """
    Parameters
    ----------
    results : list or dict
    number_formatter : str or None
        A formatter to properly show floating point numbers. For example,
        ``"{:>8.3f}"`` to show numbers with 3 decimal places.
    blank_line : bool
        If True, a blank line will be printed after the results.

    See Also
    --------
    shock_compression
    """
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["P2 / P1", "rho2 / rho1"]
    _print_results_helper(results, labels, "{:20}", number_formatter, blank_line)
