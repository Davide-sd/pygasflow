import numpy as np
import pygasflow.rayleigh as ray
from pygasflow.utils.common import ret_correct_vals
from pygasflow.utils.decorators import check

@check([1])
def rayleigh_solver(param_name, param_value, gamma=1.4, to_dict=False):
    """
    Compute all Rayleigh ratios and Mach number given an input parameter.

    Parameters
    ----------
    param_name : string
        Name of the parameter given in input. Can be either one of:

        * ``'m'``: Mach number
        * ``'pressure'``: Critical Pressure Ratio P/P*
        * ``'density'``: Critical Density Ratio rho/rho*
        * ``'velocity'``: Critical Velocity Ratio U/U*.
        * ``'temperature_sub'``: Critical Temperature Ratio T/T for
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

    Examples
    --------

    Compute all ratios starting from a single Mach number:

    >>> from pygasflow import rayleigh_solver
    >>> rayleigh_solver("m", 2)
    [2.0, 0.36363636363636365, 0.6875, 0.5289256198347108, 1.5030959785260414, 0.793388429752066, 1.4545454545454546, 1.2175752061512626]

    Compute the subsonic Mach number starting from the critical entropy ratio:

    >>> results = rayleigh_solver("entropy_sub", 0.5)
    >>> print(results[0])
    0.6634188478510624

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2:

    >>> results = rayleigh_solver("m", [0.5, 1.5], 1.2)
    >>> print(results[3])
    [0.71597633 0.79547115]

    Compute the critical temperature ratio starting from multiple Mach numbers
    for a gas having specific heat ratio gamma=1.2, returning a dictionary:

    >>> results = rayleigh_solver("m", [0.5, 1.5], 1.2, to_dict=True)
    >>> print(results["trs"])
    [0.71597633 0.79547115]

    """

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

    M = None
    if param_name == "m":
        M = param_value
        if not np.all(M >= 0):
            raise ValueError("Mach number must be >= 0.")
        # if there is only one mach number, doesn't make any sense to keep it
        # into a numpy array. Let's extract it.
        M = ret_correct_vals(M)
    elif param_name == 'total_pressure_sub':
        M = ray.m_from_critical_total_pressure_ratio.__no_check(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = ray.m_from_critical_total_pressure_ratio.__no_check(param_value, "super", gamma)
    elif param_name == 'total_temperature_sub':
        M = ray.m_from_critical_total_temperature_ratio.__no_check(param_value, "sub", gamma)
    elif param_name == 'total_temperature_super':
        M = ray.m_from_critical_total_temperature_ratio.__no_check(param_value, "super", gamma)
    elif param_name == 'temperature_sub':
        M = ray.m_from_critical_temperature_ratio.__no_check(param_value, "sub", gamma)
    elif param_name == 'temperature_super':
        M = ray.m_from_critical_temperature_ratio.__no_check(param_value, "super", gamma)
    elif param_name == 'entropy_sub':
        M = ray.m_from_critical_entropy.__no_check(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = ray.m_from_critical_entropy.__no_check(param_value, "super", gamma)

    func_dict = {
        'pressure': ray.m_from_critical_pressure_ratio,
        'density': ray.m_from_critical_density_ratio,
        'velocity': ray.m_from_critical_velocity_ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check(param_value, gamma)

    # compute the different ratios
    prs, drs, trs, tprs, ttrs, urs, eps = ray.get_ratios_from_mach.__no_check(M, gamma)

    if to_dict:
        return {
            "m": M,
            "prs": prs,
            "drs": drs,
            "trs": trs,
            "tprs": tprs,
            "ttrs": ttrs,
            "urs": urs,
            "eps": eps
        }
    return M, prs, drs, trs, tprs, ttrs, urs, eps
