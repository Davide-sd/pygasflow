import numpy as np
import pygasflow.isentropic as ise
from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.decorators import check

@check([1])
def isentropic_solver(param_name, param_value, gamma=1.4, to_dict=False):
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


    Examples
    --------

    Compute all ratios starting from a single Mach number:

    >>> from pygasflow import isentropic_solver
    >>> isentropic_solver("m", 2)
    [2.0, 0.12780452546295096, 0.2300481458333117, 0.5555555555555556, 0.24192491286747442, 0.36288736930121157, 0.6666666666666667, 2.3515101530718505, 1.6875000000000002, 30.000000000000004, 26.379760813416457]

    Compute all parameters starting from the pressure ratio:

    >>> isentropic_solver("pressure", 0.12780452546295096)
    [1.9999999999999996, 0.12780452546295107, 0.23004814583331185, 0.5555555555555558, 0.24192491286747458, 0.3628873693012118, 0.6666666666666669, 2.3515101530718505, 1.6874999999999993, 30.00000000000001, 26.379760813416446]

    Compute the Mach number starting from the Mach Angle:

    >>> results = isentropic_solver("mach_angle", 25)
    >>> print(results[0])
    2.3662015831524985

    Compute the pressure ratios starting from two Mach numbers:

    >>> results = isentropic_solver("m", [2, 3])
    >>> print(results[1])
    [0.12780453 0.02722368]

    Compute the pressure ratios starting from two Mach numbers, returning a
    dictionary:

    >>> results = isentropic_solver("m", [2, 3], to_dict=True)
    >>> print(results["pr"])
    [0.12780453 0.02722368]

    """

    if not isinstance(param_name, str):
        raise ValueError("param_name must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'density', 'temperature', 'crit_area_sub', 'crit_area_super', 'mach_angle', 'prandtl_meyer']
    if param_name not in available_pnames:
        raise ValueError("param_name not recognized. Must be one of the following:\n{}".format(available_pnames))

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if a single mach number was provided, convert it to scalar so that
        # we have all scalar values in the output
        M = ret_correct_vals(M)
    elif param_name == "crit_area_sub":
        M = ise.m_from_critical_area_ratio.__no_check(param_value, "sub", gamma)
    elif param_name == "crit_area_super":
        M = ise.m_from_critical_area_ratio.__no_check(param_value, "super", gamma)

    func_dict = {
        'pressure': ise.m_from_pressure_ratio,
        'density': ise.m_from_density_ratio,
        'temperature': ise.m_from_temperature_ratio,
        'mach_angle': ise.m_from_mach_angle,
        'prandtl_meyer': ise.m_from_prandtl_meyer_angle,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check(param_value, gamma)
    # compute the different ratios
    pr, dr, tr, prs, drs, trs, urs, ar, ma, pm = ise.get_ratios_from_mach.__no_check(M, gamma)

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
