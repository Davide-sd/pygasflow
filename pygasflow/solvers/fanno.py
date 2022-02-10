import numpy as np
import pygasflow.fanno as fanno
from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.decorators import check

@check([1])
def fanno_solver(param_name, param_value, gamma=1.4, to_dict=False):
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

    Examples
    --------

    Compute all ratios starting from a Mach number:

    >>> from pygasflow import fanno_solver
    >>> fanno_solver("m", 2)
    [2.0, 0.408248290463863, 0.6123724356957945, 0.6666666666666667, 1.6875000000000002, 1.632993161855452, 0.3049965025814798, 0.523248143764548]

    Compute the subsonic Mach number starting from the critical friction
    parameter:

    >>> results = fanno_solver("friction_sub", 0.3049965025814798)
    >>> print(results[0])
    0.6572579935727846

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2:

    >>> results = fanno_solver("m", [0.5, 1.5], 1.2)
    >>> print(results[3])
    [1.07317073 0.89795918]

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2, returning a dictionary:

    >>> results = fanno_solver("m", [0.5, 1.5], 1.2, to_dict=True)
    >>> print(results["trs"])
    [1.07317073 0.89795918]

    """

    if not isinstance(param_name, str):
        raise ValueError("param_name must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'density', 'temperature', 'total_pressure_sub', 'total_pressure_super', 'velocity', 'friction_sub', 'friction_super', 'entropy_sub', 'entropy_super']
    if param_name not in available_pnames:
        raise ValueError("param_name not recognized. Must be one of the following:\n{}".format(available_pnames))

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if there is only one mach number, doesn't make any sense to keep it
        # into a numpy array. Let's extract it.
        M = ret_correct_vals(M)
    elif param_name == 'total_pressure_sub':
        M = fanno.m_from_critical_total_pressure_ratio.__no_check(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = fanno.m_from_critical_total_pressure_ratio.__no_check(param_value, "super", gamma)
    elif param_name == 'friction_sub':
        M = fanno.m_from_critical_friction.__no_check(param_value, "sub", gamma)
    elif param_name == 'friction_super':
        M = fanno.m_from_critical_friction.__no_check(param_value, "super", gamma)
    elif param_name == 'entropy_sub':
        M = fanno.m_from_critical_entropy.__no_check(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = fanno.m_from_critical_entropy.__no_check(param_value, "super", gamma)

    func_dict = {
        'pressure': fanno.m_from_critical_pressure_ratio,
        'density': fanno.m_from_critical_density_ratio,
        'temperature': fanno.m_from_critical_temperature_ratio,
        'velocity': fanno.m_from_critical_velocity_ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check(param_value, gamma)

    # compute the different ratios
    prs, drs, trs, tprs, urs, fps, eps = fanno.get_ratios_from_mach.__no_check(M, gamma)

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
