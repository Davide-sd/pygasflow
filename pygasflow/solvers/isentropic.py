import numpy as np
import pygasflow
import pygasflow.isentropic as ise
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

    >>> from pygasflow.solvers import isentropic_solver
    >>> res = isentropic_solver("m", 2)
    >>> res
    [np.float64(2.0), np.float64(0.12780452546295096), np.float64(0.23004814583331168), np.float64(0.5555555555555556), np.float64(0.24192491286747442), np.float64(0.36288736930121157), np.float64(0.6666666666666667), np.float64(1.632993161855452), np.float64(1.6875000000000002), np.float64(30.000000000000004), np.float64(26.379760813416457)]
    >>> res.show()
    idx   quantity            
    --------------------------
    0     M                        2.00000000
    1     P / P0                   0.12780453
    2     rho / rho0               0.23004815
    3     T / T0                   0.55555556
    4     P / P*                   0.24192491
    5     rho / rho*               0.36288737
    6     T / T*                   0.66666667
    7     U / U*                   1.63299316
    8     A / A*                   1.68750000
    9     Mach Angle              30.00000000
    10    Prandtl-Meyer           26.37976081

    Compute all parameters starting from the pressure ratio:

    >>> res = isentropic_solver("pressure", 0.12780452546295096)
    >>> res.show()
    idx   quantity            
    --------------------------
    0     M                        2.00000000
    1     P / P0                   0.12780453
    2     rho / rho0               0.23004815
    3     T / T0                   0.55555556
    4     P / P*                   0.24192491
    5     rho / rho*               0.36288737
    6     T / T*                   0.66666667
    7     U / U*                   1.63299316
    8     A / A*                   1.68750000
    9     Mach Angle              30.00000000
    10    Prandtl-Meyer           26.37976081

    Compute the Mach number starting from the Mach Angle:

    >>> results = isentropic_solver("mach_angle", 25)
    >>> results.show()
    idx   quantity            
    --------------------------
    0     M                        2.36620158
    1     P / P0                   0.07210756
    2     rho / rho0               0.15285231
    3     T / T0                   0.47174663
    4     P / P*                   0.13649451
    5     rho / rho*               0.24111550
    6     T / T*                   0.56609595
    7     U / U*                   1.78031465
    8     A / A*                   2.32958260
    9     Mach Angle              25.00000000
    10    Prandtl-Meyer           35.92354277
    >>> print(results[0])
    2.3662015831524985

    Compute the pressure ratios starting from two Mach numbers:

    >>> results = isentropic_solver("m", [2, 3])
    >>> results.show()
    idx   quantity            
    --------------------------
    0     M                        2.00000000     3.00000000
    1     P / P0                   0.12780453     0.02722368
    2     rho / rho0               0.23004815     0.07622631
    3     T / T0                   0.55555556     0.35714286
    4     P / P*                   0.24192491     0.05153250
    5     rho / rho*               0.36288737     0.12024251
    6     T / T*                   0.66666667     0.42857143
    7     U / U*                   1.63299316     1.96396101
    8     A / A*                   1.68750000     4.23456790
    9     Mach Angle              30.00000000    19.47122063
    10    Prandtl-Meyer           26.37976081    49.75734674
    >>> print(results[1])
    [0.12780453 0.02722368]

    Compute the pressure ratios starting from two Mach numbers, returning a
    dictionary:

    >>> results = isentropic_solver("m", [2, 3], to_dict=True)
    >>> results.show()
    key     quantity            
    ----------------------------
    m       M                        2.00000000     3.00000000
    pr      P / P0                   0.12780453     0.02722368
    dr      rho / rho0               0.23004815     0.07622631
    tr      T / T0                   0.55555556     0.35714286
    prs     P / P*                   0.24192491     0.05153250
    drs     rho / rho*               0.36288737     0.12024251
    trs     T / T*                   0.66666667     0.42857143
    urs     U / U*                   1.63299316     1.96396101
    ars     A / A*                   1.68750000     4.23456790
    ma      Mach Angle              30.00000000    19.47122063
    pm      Prandtl-Meyer           26.37976081    49.75734674
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

    is_pint = _is_pint_quantity(param_value)
    if is_pint:
        if param_name in ["mach_angle", "prandtl_meyer"]:
            param_value = param_value.to("deg").magnitude
        else:
            param_value = param_value.magnitude

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

    if is_pint:
        deg = pygasflow.defaults.pint_ureg.deg
        ma *= deg
        pm *= deg

    if to_dict:
        return FlowResultsDict(
            m=M,
            pr=pr,
            dr=dr,
            tr=tr,
            prs=prs,
            drs=drs,
            trs=trs,
            urs=urs,
            ars=ar,
            ma=ma,
            pm=pm,
            printer=print_isentropic_results
        )
    return FlowResultsList(
        [M, pr, dr, tr, prs, drs, trs, urs, ar, ma, pm],
        printer=print_isentropic_results
    )


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
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["M", "P / P0", "rho / rho0", "T / T0",
        "P / P*", "rho / rho*", "T / T*", "U / U*", "A / A*",
        "Mach Angle", "Prandtl-Meyer"]
    _print_results_helper(results, labels, "{:20}", number_formatter, blank_line)



def isentropic_compression(pr=None, dr=None, tr=None, gamma=1.4, to_dict=None):
    """
    Solve the isentropic compression (or expansion) by providing a ratio.

    Parameters
    ----------
    pr : array_like, optional
        Pressure ratio p2/p1
    dr : array_like, optional
        Density ratio rho2/rho1
    tr : array_like, optional
        Temperature ratio T2/T1
    gamma : float, optional
        Specific heats ratio. Must be gamma > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).
    
    Returns
    -------
    pr : array_like
        Pressure ratio p2/p1
    dr : array_like
        Density ratio rho2/rho1
    tr : array_like
        Temperature ratio T2/T1
    
    Examples
    --------

    >>> from pygasflow import isentropic_compression
    >>> T1 = 290    # K
    >>> res = isentropic_compression(pr=2, to_dict=True)
    >>> res.show()
    key     quantity            
    ----------------------------
    pr      P2 / P1                  2.00000000
    dr      rho2 / rho1              1.64067071
    tr      T2 / T1                  1.21901365
    >>> T2_T1 = res["tr"]
    >>> T2 = T2_T1 * T1
    >>> T2
    np.float64(353.5139597192979)

    See Also
    --------
    :func:`~pygasflow.solvers.shockwave.shock_compression`

    """
    to_dict = _should_solver_return_dict(to_dict)
    if pr is not None:
        is_scalar = isinstance(pr, Number)
        pr = np.atleast_1d(pr)
        dr = pr ** (1 / gamma)
        tr = pr ** (1 - 1 / gamma)
    elif dr is not None:
        is_scalar = isinstance(dr, Number)
        dr = np.atleast_1d(dr)
        pr = dr ** gamma
        tr = pr ** (1 - 1 / gamma)
    elif tr is not None:
        is_scalar = isinstance(tr, Number)
        tr = np.atleast_1d(tr)
        pr = tr ** (gamma / (gamma - 1))
        dr = pr ** (1 / gamma)
    else:
        raise ValueError(
            "Either `pr` or `dr` or `tr` must be numerical values.")
    if is_scalar:
        pr, dr, tr = pr[0], dr[0], tr[0]

    if to_dict:
        return FlowResultsDict(
            pr=pr, dr=dr, tr=tr,
            printer=print_isentropic_compression_results
        )
    return FlowResultsList(
        [pr, dr, tr],
        printer=print_isentropic_compression_results
    )


def print_isentropic_compression_results(results, number_formatter=None, blank_line=False):
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
    # NOTE: the white space wrapping '/' are necessary, otherwise Sphinx
    # will process these labels and convert them to Latex, thanks to
    # the substitutions I implemented in doc/conf.py
    labels = ["P2 / P1", "rho2 / rho1", "T2 / T1"]
    _print_results_helper(results, labels, "{:20}", number_formatter, blank_line)
