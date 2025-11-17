from pygasflow.isentropic import (
    sonic_density_ratio,
    sonic_pressure_ratio,
    sonic_sound_speed_ratio,
    sonic_temperature_ratio
)
from pygasflow.utils.common import (
    _should_solver_return_dict,
    _print_results_helper,
    FlowResultsDict,
    FlowResultsList,
)
from pygasflow.utils.decorators import check


def gas_solver(p1_name, p1_value, p2_name="gamma", p2_value=1.4, to_dict=None):
    """
    For a thermally perfect gas, compute quantities like the ratio of specific 
    heats, the heat capacities and the mass-specific gas constant.

    Parameters
    ----------
    p1_name : str
        Name of the first parameter given in input. Can be either one of:

        * ``"cp"``: heat capacity at constant pressure.
        * ``"cv"``: heat capacity at constant volume.
        * ``"gamma"``: ration of specific heats.
        * ``"r"``: mass-specific gas constant.
    p1_value : float
        Actual value of the first parameter.
    p2_name : str
        Name of the second parameter given in input. Can be either one of:

        * ``"cp"``: heat capacity at constant pressure.
        * ``"cv"``: heat capacity at constant volume.
        * ``"gamma"``: ration of specific heats.
        * ``"r"``: mass-specific gas constant.

        It must be different from ``p1_name``.
    p2_value : float
        Actual value of the second parameter.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    gamma : float
    r : float
    Cp : float
    Cv : float

    See Also
    --------
    print_gas_results,
    :class:`~pygasflow.interactive.diagrams.gas.GasDiagram`

    Examples
    --------

    Compute the specific heats for air:

    >>> from pygasflow.solvers.gas import gas_solver
    >>> res1 = gas_solver("r", 287.05, "gamma", 1.4)
    >>> res1
    [1.4, 287.05, 1004.6750000000002, 717.6250000000002]
    >>> res1.show()
    idx   quantity     
    -------------------
    0     gamma             1.40000000
    1     R               287.05000000
    2     Cp             1004.67500000
    3     Cv              717.62500000

    Compute the specific heats for methane, using pint quantities:

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> res2 = gas_solver(
    ...     "r", 518.28 * ureg.J / (ureg.kg * ureg.K), "gamma", 1.32, 
    ...     to_dict=True)
    >>> res2
    {'gamma': 1.32, 'R': <Quantity(518.28, 'joule / kilogram / kelvin')>, 'Cp': <Quantity(2137.905, 'joule / kilogram / kelvin')>, 'Cv': <Quantity(1619.625, 'joule / kilogram / kelvin')>}
    >>> res2.show()
    key     quantity         
    -------------------------
    gamma   gamma                 1.32000000
    R       R [J / K / kg]      518.28000000
    Cp      Cp [J / K / kg]    2137.90500000
    Cv      Cv [J / K / kg]    1619.62500000

    """
    to_dict = _should_solver_return_dict(to_dict)
    allowed_names = ["cp", "cv", "gamma", "r"]
    p1_name = p1_name.lower()
    p2_name = p2_name.lower()
    if p1_name == p2_name:
        raise ValueError("`p1_name` must be different from `p2_name`.")
    if (p1_name not in allowed_names) or (p2_name not in allowed_names):
        raise ValueError(
            "Wrong `p1_name` or `p2_name`. Possible values are:"
            f" {allowed_names}. Instead, this was received: "
            f" p1_name={p1_name}, p2_name={p2_name}."
        )

    def from_Cv_r(Cv, r):
        Cp = Cv + r
        gamma = (Cv + r) / Cv
        return Cp, gamma

    def from_Cv_gamma(Cv, gamma):
        Cp = Cv * gamma
        r = Cv * gamma - Cv
        return Cp, r

    def from_gamma_r(gamma, r):
        Cp = r * gamma / (gamma - 1)
        Cv = r / (gamma - 1)
        return Cp, Cv

    def from_Cp_r(Cp, r):
        Cv = Cp - r
        gamma = Cp / (Cp - r)
        return Cv, gamma

    def from_Cp_gamma(Cp, gamma):
        Cv = Cp / gamma
        r = (Cp * gamma - Cp) / gamma
        return Cv, r

    def from_Cp_Cv(Cp, Cv):
        r = Cp - Cv
        gamma = Cp / Cv
        return gamma, r

    Cp, Cv, gamma, r = [None] * 4
    if (p1_name == "cv") and (p2_name == "r"):
        Cv = p1_value
        r = p2_value
        Cp, gamma = from_Cv_r(Cv, r)
    elif (p1_name == "r") and (p2_name == "cv"):
        Cv = p2_value
        r = p1_value
        Cp, gamma = from_Cv_r(Cv, r)
    elif (p1_name == "cv") and (p2_name == "gamma"):
        Cv = p1_value
        gamma = p2_value
        Cp, r = from_Cv_gamma(Cv, gamma)
    elif (p1_name == "gamma") and (p2_name == "cv"):
        Cv = p2_value
        gamma = p1_value
        Cp, r = from_Cv_gamma(Cv, gamma)
    elif (p1_name == "gamma") and (p2_name == "r"):
        gamma = p1_value
        r = p2_value
        Cp, Cv = from_gamma_r(gamma, r)
    elif (p1_name == "r") and (p2_name == "gamma"):
        gamma = p2_value
        r = p1_value
        Cp, Cv = from_gamma_r(gamma, r)
    elif (p1_name == "cp") and (p2_name == "r"):
        Cp = p1_value
        r = p2_value
        Cv, gamma = from_Cp_r(Cp, r)
    elif (p1_name == "r") and (p2_name == "cp"):
        Cp = p2_value
        r = p1_value
        Cv, gamma = from_Cp_r(Cp, r)
    elif (p1_name == "cp") and (p2_name == "gamma"):
        Cp = p1_value
        gamma = p2_value
        Cv, r = from_Cp_gamma(Cp, gamma)
    elif (p1_name == "gamma") and (p2_name == "cp"):
        Cp = p2_value
        gamma = p1_value
        Cv, r = from_Cp_gamma(Cp, gamma)
    elif (p1_name == "cv") and (p2_name == "cp"):
        Cv = p1_value
        Cp = p2_value
        gamma, r = from_Cp_Cv(Cp, Cv)
    elif (p1_name == "cp") and (p2_name == "cv"):
        Cv = p2_value
        Cp = p1_value
        gamma, r = from_Cp_Cv(Cp, Cv)
    else:
        raise ValueError(
            f"Unknown configuration: {p1_name}, {p2_name}"
        )

    if to_dict:
        return FlowResultsDict(
            gamma=gamma,
            R=r,
            Cp=Cp,
            Cv=Cv,
            printer=print_gas_results
        )
    return FlowResultsList([gamma, r, Cp, Cv], printer=print_gas_results)


def ideal_gas_solver(wanted, p=None, rho=None, R=None, T=None, to_dict=None):
    """Solve for quantities of the ideal gas law: p/rho = R*T

    Note: 3 numerical parameters are needed to compute the wanted quantity.

    Parameters
    ----------
    wanted : str
        Name of the parameter to compute. Can be either one of:

        * ``"p"``: static pressure [Pa].
        * ``"rho"``: density [Kg / m**3].
        * ``"R"``: mass-specific gas constant [J / (Kg K)].
        * ``"T"``: temperature [K].

    p : float or None
        Static pressure [Pa].
    rho : float or None
        Density [Kg / m**3].
    R : float or None
        Mass-specific gas constant [J / (Kg K)].
    T : float or None
        Temperature [K].
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False.

    Returns
    -------
    p : float
    rho : float
    R : float
    T : float

    See Also
    --------
    print_ideal_gas_results,
    :class:`~pygasflow.interactive.diagrams.gas.IdealGasDiagram`

    Examples
    --------

    >>> from pygasflow.solvers.gas import ideal_gas_solver
    >>> res1 = ideal_gas_solver("p", R=287.05, T=288, rho=1.2259)
    >>> res1
    [101345.64336000002, 1.2259, 287.05, 288]
    >>> res1.show()
    idx   quantity     
    -------------------
    0     P            101345.64336000
    1     rho               1.22590000
    2     R               287.05000000
    3     T               288.00000000
    >>> res2 = ideal_gas_solver("p", R=287.05, T=288, rho=1.2259, to_dict=True)
    >>> res2
    {'p': 101345.64336000002, 'rho': 1.2259, 'R': 287.05, 'T': 288}
    >>> res2.show()
    key     quantity     
    ---------------------
    p       P            101345.64336000
    rho     rho               1.22590000
    R       R               287.05000000
    T       T               288.00000000
    """
    to_dict = _should_solver_return_dict(to_dict)
    wanted = wanted.lower()
    allowed_wanted = ["p", "rho", "t", "r"]
    if wanted not in allowed_wanted:
        raise ValueError(
            f"`wanted` must be one of the following: {allowed_wanted}."
            f" Instead, '{wanted}' was received."
        )
    d = [("p", p), ("t", T), ("rho", rho), ("r", R)]
    not_None = [e for e in d if e[1] is not None]
    if len(not_None) < 3:
        not_None = ", ".join([e[0] + f"={e[1]}" for e in not_None])
        msg = f" Instead, only these parameters were received: {not_None}."
        if not not_None:
            msg = " Instead, no parameters were provided."
        raise ValueError(
            "To solve the ideal gas law, 3 parameters must be provided."
            + msg
        )
    d = {k: v for (k, v) in d}
    if (d[wanted] is not None) and (len(not_None) == 3):
        raise ValueError(
            "`wanted` is also a parameter:"
            f" {wanted}={d[wanted]}. However, there are not enough"
            " parameters to allow the computation."
        )

    if wanted == "p":
        p = rho * R * T
    elif wanted == "rho":
        rho = p / (R * T)
    elif wanted == "r":
        R = p / (rho * T)
    else:
        T = p / (rho * R)

    if to_dict:
        return FlowResultsDict(
            p=p,
            rho=rho,
            R=R,
            T=T,
            printer=print_ideal_gas_results
        )
    return FlowResultsList([p, rho, R, T], printer=print_ideal_gas_results)


@check([0], skip_gamma_check=True)
def sonic_condition(gamma=1.4, to_dict=False):
    """Compute the sonic condition for a gas.

    Parameters
    ----------
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    drs : float
        Sonic density ratio rho0/rho*
    prs : float
        Sonic pressure ratio P0/P*
    ars : float
        Sonic Temperature ratio a0/a*
    trs : float
        Sonic Temperature ratio T0/T*

    See Also
    --------
    print_sonic_condition_results,
    :class:`~pygasflow.interactive.diagrams.gas.SonicDiagram`

    Examples
    --------

    >>> from pygasflow.solvers import sonic_condition
    >>> res = sonic_condition(1.4, to_dict=True)
    >>> res
    {'drs': np.float64(1.5774409656148785), 'prs': np.float64(1.892929158737854), 'ars': np.float64(1.0954451150103321), 'trs': np.float64(1.2)}
    >>> res.show()
    key     quantity     
    ---------------------
    drs     rho0/rho*         1.57744097
    prs     P0/P*             1.89292916
    ars     a0/T*             1.09544512
    trs     T0/T*             1.20000000

    """
    drs = sonic_density_ratio.__no_check__(gamma)
    prs = sonic_pressure_ratio.__no_check__(gamma)
    ars = sonic_sound_speed_ratio.__no_check__(gamma)
    trs = sonic_temperature_ratio.__no_check__(gamma)
    if to_dict:
        return FlowResultsDict(
            drs=drs,
            prs=prs,
            ars=ars,
            trs=trs,
            printer=print_sonic_condition_results
        )
    return FlowResultsList(
        [drs, prs, ars, trs],
        printer=print_sonic_condition_results
    )


def print_gas_results(results, number_formatter=None, blank_line=False):
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
    gas_solver
    """
    labels = ["gamma", "R", "Cp", "Cv"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def print_ideal_gas_results(results, number_formatter=None, blank_line=False):
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
    ideal_gas_solver
    """
    labels = ["P", "rho", "R", "T"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def print_sonic_condition_results(results, number_formatter=None, blank_line=False):
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
    sonic_condition
    """
    labels = ["rho0/rho*", "P0/P*", "a0/T*", "T0/T*"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)
