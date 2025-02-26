import numpy as np
import pygasflow.isentropic as ise
from pygasflow.utils.common import (
    ret_correct_vals,
    _should_solver_return_dict,
    _print_results_helper,
)
from pygasflow.utils.decorators import check
from numbers import Number


@check([1])
def isentropic_solver(param_name, param_value, gamma=1.4, to_dict=None):
    """
    Compute all isentropic ratios and Mach number given an input parameter.

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'m'``: Mach number
        * ``'pressure'``: Pressure Ratio P/P0
        * ``'density'``: Density Ratio rho/rho0
        * ``'temperature'``: Temperature Ratio T/T0
        * ``'crit_area_sub'``: Critical Area Ratio A/A* for subsonic case.
        * ``'crit_area_super'``: Critical Area Ratio A/A* for supersonic case.
        * ``'mach_angle'``: Mach Angle in degrees.
        * ``'prandtl_meyer'``: Prandtl-Meyer Angle in degrees.

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
    ars : array_like
        Critical Area Ratio A/A*
    ma : array_like
        Mach Angle
    pm : array_like
        Prandtl-Meyer Angle

    See Also
    --------
    print_isentropic_results,
    :class:`~pygasflow.interactive.diagrams.isentropic.IsentropicDiagram`

    Examples
    --------

    Compute all ratios starting from a single Mach number:

    >>> from pygasflow.solvers import isentropic_solver, print_isentropic_results
    >>> res = isentropic_solver("m", 2)
    >>> res
    [np.float64(2.0), np.float64(0.12780452546295096), np.float64(0.23004814583331168), np.float64(0.5555555555555556), np.float64(0.24192491286747442), np.float64(0.36288736930121157), np.float64(0.6666666666666667), np.float64(2.3515101530718505), np.float64(1.6875000000000002), np.float64(30.000000000000004), np.float64(26.379760813416457)]
    >>> print_isentropic_results(res)
    M                      2.00000000
    P / P0                 0.12780453
    rho / rho0             0.23004815
    T / T0                 0.55555556
    P / P*                 0.24192491
    rho / rho*             0.36288737
    T / T*                 0.66666667
    U / U*                 2.35151015
    A / A*                 1.68750000
    Mach Angle            30.00000000
    Prandtl-Meyer         26.37976081

    Compute all parameters starting from the pressure ratio:

    >>> res = isentropic_solver("pressure", 0.12780452546295096)
    >>> print_isentropic_results(res)
    M                      2.00000000
    P / P0                 0.12780453
    rho / rho0             0.23004815
    T / T0                 0.55555556
    P / P*                 0.24192491
    rho / rho*             0.36288737
    T / T*                 0.66666667
    U / U*                 2.35151015
    A / A*                 1.68750000
    Mach Angle            30.00000000
    Prandtl-Meyer         26.37976081

    Compute the Mach number starting from the Mach Angle:

    >>> results = isentropic_solver("mach_angle", 25)
    >>> print_isentropic_results(results)
    M                      2.36620158
    P / P0                 0.07210756
    rho / rho0             0.15285231
    T / T0                 0.47174663
    P / P*                 0.13649451
    rho / rho*             0.24111550
    T / T*                 0.56609595
    U / U*                 2.56365309
    A / A*                 2.32958260
    Mach Angle            25.00000000
    Prandtl-Meyer         35.92354277
    >>> print(results[0])
    2.3662015831524985

    Compute the pressure ratios starting from two Mach numbers:

    >>> results = isentropic_solver("m", [2, 3])
    >>> print_isentropic_results(results)
    M                      2.00000000     3.00000000
    P / P0                 0.12780453     0.02722368
    rho / rho0             0.23004815     0.07622631
    T / T0                 0.55555556     0.35714286
    P / P*                 0.24192491     0.05153250
    rho / rho*             0.36288737     0.12024251
    T / T*                 0.66666667     0.42857143
    U / U*                 2.35151015     2.82810386
    A / A*                 1.68750000     4.23456790
    Mach Angle            30.00000000    19.47122063
    Prandtl-Meyer         26.37976081    49.75734674
    >>> print(results[1])
    [0.12780453 0.02722368]

    Compute the pressure ratios starting from two Mach numbers, returning a
    dictionary:

    >>> results = isentropic_solver("m", [2, 3], to_dict=True)
    >>> print(results["pr"])
    [0.12780453 0.02722368]

    """
    to_dict = _should_solver_return_dict(to_dict)
    if not isinstance(param_name, str):
        raise ValueError("`param_name` must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'density', 'temperature', 'crit_area_sub', 'crit_area_super', 'mach_angle', 'prandtl_meyer']
    if param_name not in available_pnames:
        raise ValueError("`param_name` not recognized. Must be one of the following:\n{}".format(available_pnames))
    if not isinstance(gamma, Number):
        raise ValueError("The specific heats ratio must be > 1.")

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if a single mach number was provided, convert it to scalar so that
        # we have all scalar values in the output
        M = ret_correct_vals(M)
    elif param_name == "crit_area_sub":
        M = ise.m_from_critical_area_ratio.__no_check__(param_value, "sub", gamma)
    elif param_name == "crit_area_super":
        M = ise.m_from_critical_area_ratio.__no_check__(param_value, "super", gamma)

    func_dict = {
        'pressure': ise.m_from_pressure_ratio,
        'density': ise.m_from_density_ratio,
        'temperature': ise.m_from_temperature_ratio,
        'mach_angle': ise.m_from_mach_angle,
        'prandtl_meyer': ise.m_from_prandtl_meyer_angle,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check__(param_value, gamma)
    # compute the different ratios
    M = np.atleast_1d(M)
    pr, dr, tr, prs, drs, trs, urs, ar, ma, pm = ise.get_ratios_from_mach.__no_check__(M, gamma)

    if to_dict:
        return {
            "m": M,
            "pr": pr,
            "dr": dr,
            "tr": tr,
            "prs": prs,
            "drs": drs,
            "trs": trs,
            "urs": urs,
            "ars": ar,
            "ma": ma,
            "pm": pm
        }
    return M, pr, dr, tr, prs, drs, trs, urs, ar, ma, pm


def print_isentropic_results(results, number_formatter=None, blank_line=False):
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
    isentropic_solver
    """
    data = results.values() if isinstance(results, dict) else results
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["M", "P / P0", "rho / rho0", "T / T0",
        "P / P*", "rho / rho*", "T / T*", "U / U*", "A / A*",
        "Mach Angle", "Prandtl-Meyer"]
    _print_results_helper(data, labels, "{:18}", number_formatter, blank_line)
