import numpy as np
import pygasflow.rayleigh as ray
from pygasflow.utils.common import (
    ret_correct_vals,
    _should_solver_return_dict,
    _print_results_helper,
    _is_pint_quantity,
    FlowResultsDict,
    FlowResultsList,
)
from pygasflow.utils.decorators import check
from numbers import Number


@check([1])
def rayleigh_solver(param_name, param_value, gamma=1.4, to_dict=None):
    """
    Given an input parameter, compute all ratios and Mach number 
    of 1-D flow with heat addition (Rayleigh flow).

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'m'``: Mach number
        * ``'pressure'``: Critical Pressure Ratio P/P*
        * ``'density'``: Critical Density Ratio rho/rho*
        * ``'velocity'``: Critical Velocity Ratio U/U*.
        * ``'temperature_sub'``: Critical Temperature Ratio T/T* for
          subsonic case.
        * ``'temperature_super'``: Critical Temperature Ratio T/T* for
          supersonic case.
        * ``'total_pressure_sub'``: Critical Total Pressure Ratio P0/P0*
          for subsonic case.
        * ``'total_pressure_super'``: Critical Total Pressure Ratio P0/P0*
          for supersonic case.
        * ``'total_temperature_sub'``: Critical Total Temperature Ratio T0/T0*
          for subsonic case.
        * ``'total_temperature_super'``: Critical Total Temperature Ratio
          T0/T0* for supersonic case.
        * ``'entropy_sub'``: Entropy parameter (s*-s)/R for subsonic case.
        * ``'entropy_super'``: Entropy parameter (s*-s)/R for supersonic case.

    param_value : float/list/array_like
        Actual value of the parameter. If float, list, tuple is given as
        input, a conversion will be attempted.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    m : array_like
        Mach number
    prs : array_like
        Critical Pressure Ratio P/P*
    drs : array_like
        Critical Density Ratio rho/rho*
    trs : array_like
        Critical Temperature Ratio T/T*
    tprs : array_like
        Critical Total Pressure Ratio P0/P0*
    ttrs : array_like
        Critical Total Temperature Ratio T0/T0*
    urs : array_like
        Critical Velocity Ratio U/U*
    eps : array_like
        Critical Entropy Ratio (s*-s)/R

    See Also
    --------
    print_rayleigh_results,
    :class:`~pygasflow.interactive.diagrams.rayleigh.RayleighDiagram`

    Examples
    --------

    Compute all ratios starting from a single Mach number:

    >>> from pygasflow.solvers import rayleigh_solver
    >>> res = rayleigh_solver("m", 2)
    >>> res
    [np.float64(2.0), np.float64(0.36363636363636365), np.float64(0.6875), np.float64(0.5289256198347108), np.float64(1.5030959785260414), np.float64(0.793388429752066), np.float64(1.4545454545454546), np.float64(1.2175752061512626)]
    >>> res.show()
    idx   quantity     
    -------------------
    0     M                 2.00000000
    1     P / P*            0.36363636
    2     rho / rho*        0.68750000
    3     T / T*            0.52892562
    4     P0 / P0*          1.50309598
    5     T0 / T0*          0.79338843
    6     U / U*            1.45454545
    7     (s*-s) / R        1.21757521

    Compute the subsonic Mach number starting from the critical entropy ratio:

    >>> res = rayleigh_solver("entropy_sub", 0.5)
    >>> res.show()
    idx   quantity     
    -------------------
    0     M                 0.66341885
    1     P / P*            1.48498826
    2     rho / rho*        1.53003501
    3     T / T*            0.97055835
    4     P0 / P0*          1.05396114
    5     T0 / T0*          0.87999306
    6     U / U*            0.65357982
    7     (s*-s) / R        0.50000000
    >>> print(res[0])
    0.6634188478510624

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2:

    >>> res = rayleigh_solver("m", [0.5, 1.5], 1.2)
    >>> res.show()
    idx   quantity     
    -------------------
    0     M                 0.50000000     1.50000000
    1     P / P*            1.69230769     0.59459459
    2     rho / rho*        2.36363636     0.74747475
    3     T / T*            0.71597633     0.79547115
    4     P0 / P0*          1.10781288     1.13417842
    5     T0 / T0*          0.66715976     0.88586560
    6     U / U*            0.42307692     1.33783784
    7     (s*-s) / R        2.53074211     0.85304875
    >>> print(res[3])
    [0.71597633 0.79547115]

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2, returning a dictionary:

    >>> res = rayleigh_solver("m", [0.5, 1.5], 1.2, to_dict=True)
    >>> res.show()
    key     quantity     
    ---------------------
    m       M                 0.50000000     1.50000000
    prs     P / P*            1.69230769     0.59459459
    drs     rho / rho*        2.36363636     0.74747475
    trs     T / T*            0.71597633     0.79547115
    tprs    P0 / P0*          1.10781288     1.13417842
    ttrs    T0 / T0*          0.66715976     0.88586560
    urs     U / U*            0.42307692     1.33783784
    eps     (s*-s) / R        2.53074211     0.85304875
    >>> print(res["trs"])
    [0.71597633 0.79547115]

    """
    to_dict = _should_solver_return_dict(to_dict)
    if not isinstance(param_name, str):
        raise ValueError("param_name must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'density', 'velocity',
                        'temperature_sub', 'temperature_super',
                        'total_pressure_sub', 'total_pressure_super',
                        'total_temperature_sub', 'total_temperature_super',
                        'entropy_sub', 'entropy_super']
    if param_name not in available_pnames:
        raise ValueError("param_name not recognized. Must be one of the following:\n{}".format(available_pnames))
    if not isinstance(gamma, Number):
        raise ValueError("The specific heats ratio must be > 1.")

    if _is_pint_quantity(param_value):
        param_value = param_value.magnitude

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if there is only one mach number, doesn't make any sense to keep it
        # into a numpy array. Let's extract it.
        M = ret_correct_vals(M)
    elif param_name == 'total_pressure_sub':
        M = ray.m_from_critical_total_pressure_ratio.__no_check__(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = ray.m_from_critical_total_pressure_ratio.__no_check__(param_value, "super", gamma)
    elif param_name == 'total_temperature_sub':
        M = ray.m_from_critical_total_temperature_ratio.__no_check__(param_value, "sub", gamma)
    elif param_name == 'total_temperature_super':
        M = ray.m_from_critical_total_temperature_ratio.__no_check__(param_value, "super", gamma)
    elif param_name == 'temperature_sub':
        M = ray.m_from_critical_temperature_ratio.__no_check__(param_value, "sub", gamma)
    elif param_name == 'temperature_super':
        M = ray.m_from_critical_temperature_ratio.__no_check__(param_value, "super", gamma)
    elif param_name == 'entropy_sub':
        M = ray.m_from_critical_entropy.__no_check__(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = ray.m_from_critical_entropy.__no_check__(param_value, "super", gamma)

    func_dict = {
        'pressure': ray.m_from_critical_pressure_ratio,
        'density': ray.m_from_critical_density_ratio,
        'velocity': ray.m_from_critical_velocity_ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check__(param_value, gamma)

    # compute the different ratios
    M = np.atleast_1d(M)
    prs, drs, trs, tprs, ttrs, urs, eps = ray.get_ratios_from_mach.__no_check__(M, gamma)

    if to_dict:
        return FlowResultsDict(
            m=M,
            prs=prs,
            drs=drs,
            trs=trs,
            tprs=tprs,
            ttrs=ttrs,
            urs=urs,
            eps=eps,
            printer=print_rayleigh_results
        )
    return FlowResultsList(
        [M, prs, drs, trs, tprs, ttrs, urs, eps],
        printer=print_rayleigh_results
    )


def print_rayleigh_results(results, number_formatter=None, blank_line=False):
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
    rayleigh_solver
    """
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["M", "P / P*", "rho / rho*", "T / T*", "P0 / P0*",
        "T0 / T0*", "U / U*", "(s*-s) / R"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)


def specific_heat_solver(q=None, Cp=None, T01=None, T02=None, DeltaT0=None, q_Cp=None):
    """
    Compute the missing quantities of this equation: `q = Cp * (T02 - T01)`,
    which is valid for a calorically perfect gas.

    Note that if one quantity can't be resolved, its result will be set 
    to None.

    Parameters
    ----------
    q : float or array, optional
        Specific heat per unit mass.
    Cp : float or array, optional
        Specific heat at constant-pressure.
    T01 : float or array, optional
        Total temperature at the initial state of the flow.
    T02 : float or array, optional
        Total temperature at the final state of the flow.
    DeltaT0 : float or array, optional
        Total temperature difference, T02 - T01, between two
        states of the flow.
    q_Cp : float or array, optional
        The ratio q / Cp.

    Returns
    -------
    q : float or array
        Specific heat per unit mass.
    Cp : float or array, optional
        Specific heat at constant-pressure.
    T01 : float or array, optional
        Total temperature at the initial state of the flow.
    T02 : float or array, optional
        Total temperature at the final state of the flow.
    DeltaT0 : float or array, optional
        Total temperature difference, T02 - T01, between two
        states of the flow.
    q_Cp : float or array, optional
        The ratio q / Cp.
    """
    if q_Cp and q and Cp:
        raise ValueError(
            "Too many related parameters: q, Cp, q_Cp. Please,"
            " provide at most two of them."
        )
    if DeltaT0 and T01 and T02:
        raise ValueError(
            "Too many related parameters: T01, T02, DeltaT0. Please,"
            " provide at most two of them."
        )

    if DeltaT0:
        if T01:
            T02 = DeltaT0 + T01
        elif T02:
            T01 = T02 - DeltaT0
    elif T02 and T01:
        DeltaT0 = T02 - T01

    if q_Cp:
        if q:
            Cp = (1 / q_Cp) * q
        elif Cp:
            q = q_Cp * Cp
    elif q and Cp:
        q_Cp = q / Cp

    if Cp and DeltaT0:
        q = Cp * DeltaT0
    elif q_Cp and T02:
        T01 = T02 - q_Cp
        DeltaT0 = T02 - T01
    elif q_Cp and T01:
        T02 = q_Cp + T01
        DeltaT0 = T02 - T01
    elif q_Cp:
        DeltaT0 = q_Cp
    elif q and DeltaT0:
        Cp = q / DeltaT0

    if not q_Cp:
        q_Cp = q / Cp if (q and Cp) else DeltaT0

    return FlowResultsDict(
        q=q, Cp=Cp, T01=T01, T02=T02, DeltaT0=DeltaT0, q_Cp=q_Cp,
        printer=print_specific_heat_results
    )


def print_specific_heat_results(results, number_formatter=None, blank_line=False):
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
    rayleigh_solver
    """
    labels = ["q", "Cp", "T01", "T02", "ΔT0", "q / Cp"]
    _print_results_helper(results, labels, None, number_formatter, blank_line)
