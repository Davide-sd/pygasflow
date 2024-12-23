import numpy as np
import pygasflow.fanno as fanno
from pygasflow.utils.common import (
    ret_correct_vals,
    _should_solver_return_dict,
    _print_results_helper,
)
from pygasflow.utils.decorators import check
from numbers import Number


@check([1])
def fanno_solver(param_name, param_value, gamma=1.4, to_dict=None):
    """
    Compute all Fanno ratios and Mach number given an input parameter.

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'m'``: Mach number
        * ``'pressure'``: Critical Pressure Ratio P/P*
        * ``'density'``: Critical Density Ratio rho/rho*
        * ``'temperature'``: Critical Temperature Ratio T/T*
        * ``'total_pressure_sub'``: Critical Total Pressure Ratio P0/P0*
          for subsonic case.
        * ``'total_pressure_super'``: Critical Total Pressure Ratio P0/P0*
          for supersonic case.
        * ``'velocity'``: Critical Velocity Ratio U/U*.
        * ``'friction_sub'``: Critical Friction parameter 4fL*/D for
          subsonic case.
        * ``'friction_super'``: Critical Friction parameter 4fL*/D for
          supersonic case.
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
    urs : array_like
        Critical Velocity Ratio U/U*
    fps : array_like
        Critical Friction Parameter 4fL*/D
    eps : array_like
        Critical Entropy Ratio (s*-s)/R

    See Also
    --------
    print_fanno_results,
    :class:`~pygasflow.interactive.diagrams.fanno.FannoDiagram`

    Examples
    --------

    Compute all ratios starting from a Mach number:

    >>> from pygasflow.solvers import fanno_solver, print_fanno_results
    >>> res = fanno_solver("m", 2)
    >>> res
    [np.float64(2.0), np.float64(0.408248290463863), np.float64(0.6123724356957945), np.float64(0.6666666666666667), np.float64(1.6875000000000002), np.float64(1.632993161855452), np.float64(0.3049965025814798), np.float64(0.523248143764548)]
    >>> print_fanno_results(res)
    M                2.00000000
    P / P*           0.40824829
    rho / rho*       0.61237244
    T / T*           0.66666667
    P0 / P0*         1.68750000
    U / U*           1.63299316
    4fL* / D         0.30499650
    (s*-s) / R       0.52324814

    Compute the subsonic Mach number starting from the critical friction
    parameter:

    >>> results = fanno_solver("friction_sub", 0.3049965025814798)
    >>> print_fanno_results(results)
    M                0.65725799
    P / P*           1.59904374
    rho / rho*       1.44766442
    T / T*           1.10456796
    P0 / P0*         1.12898142
    U / U*           0.69076782
    4fL* / D         0.30499650
    (s*-s) / R       0.12131583
    >>> print(results[0])
    0.6572579935727846

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2:

    >>> results = fanno_solver("m", [0.5, 1.5], 1.2)
    >>> print_fanno_results(results)
    M                0.50000000     1.50000000
    P / P*           2.07187908     0.63173806
    rho / rho*       1.93061460     0.70352647
    T / T*           1.07317073     0.89795918
    P0 / P0*         1.35628665     1.20502889
    U / U*           0.51796977     1.42141062
    4fL* / D         1.29396294     0.18172829
    (s*-s) / R       0.30475056     0.18650354
    >>> print(results[3])
    [1.07317073 0.89795918]

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2, returning a dictionary:

    >>> results = fanno_solver("m", [0.5, 1.5], 1.2, to_dict=True)
    >>> results
    {'m': array([0.5, 1.5]), 'prs': array([2.07187908, 0.63173806]), 'drs': array([1.9306146 , 0.70352647]), 'trs': array([1.07317073, 0.89795918]), 'tprs': array([1.35628665, 1.20502889]), 'urs': array([0.51796977, 1.42141062]), 'fps': array([1.29396294, 0.18172829]), 'eps': array([0.30475056, 0.18650354])}
    >>> print(results["trs"])
    [1.07317073 0.89795918]

    """
    to_dict = _should_solver_return_dict(to_dict)
    if not isinstance(param_name, str):
        raise ValueError("param_name must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'density', 'temperature', 'total_pressure_sub', 'total_pressure_super', 'velocity', 'friction_sub', 'friction_super', 'entropy_sub', 'entropy_super']
    if param_name not in available_pnames:
        raise ValueError("param_name not recognized. Must be one of the following:\n{}".format(available_pnames))
    if not isinstance(gamma, Number):
        raise ValueError("The specific heats ratio must be > 1.")

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if there is only one mach number, doesn't make any sense to keep it
        # into a numpy array. Let's extract it.
        M = ret_correct_vals(M)
    elif param_name == 'total_pressure_sub':
        M = fanno.m_from_critical_total_pressure_ratio.__no_check__(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = fanno.m_from_critical_total_pressure_ratio.__no_check__(param_value, "super", gamma)
    elif param_name == 'friction_sub':
        M = fanno.m_from_critical_friction.__no_check__(param_value, "sub", gamma)
    elif param_name == 'friction_super':
        M = fanno.m_from_critical_friction.__no_check__(param_value, "super", gamma)
    elif param_name == 'entropy_sub':
        M = fanno.m_from_critical_entropy.__no_check__(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = fanno.m_from_critical_entropy.__no_check__(param_value, "super", gamma)

    func_dict = {
        'pressure': fanno.m_from_critical_pressure_ratio,
        'density': fanno.m_from_critical_density_ratio,
        'temperature': fanno.m_from_critical_temperature_ratio,
        'velocity': fanno.m_from_critical_velocity_ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check__(param_value, gamma)

    # compute the different ratios
    M = np.atleast_1d(M)
    prs, drs, trs, tprs, urs, fps, eps = fanno.get_ratios_from_mach.__no_check__(M, gamma)

    if to_dict:
        return {
            "m": M,
            "prs": prs,
            "drs": drs,
            "trs": trs,
            "tprs": tprs,
            "urs": urs,
            "fps": fps,
            "eps": eps
        }

    return M, prs, drs, trs, tprs, urs, fps, eps


def print_fanno_results(results, number_formatter=None, blank_line=False):
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
    fanno_solver
    """
    data = results.values() if isinstance(results, dict) else results
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["M", "P / P*", "rho / rho*", "T / T*", "P0 / P0*",
        "U / U*", "4fL* / D", "(s*-s) / R"]
    _print_results_helper(data, labels, None, number_formatter, blank_line)
