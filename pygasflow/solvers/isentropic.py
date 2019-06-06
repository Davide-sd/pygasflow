import numpy as np
import pygasflow.isentropic as ise

def Isentropic_Solver(param_name, param_value, gamma=1.4):
    """
    Compute all isentropic ratios and Mach number given an input parameter.

    Parameters
    ----------
        param_name : string
            Name of the parameter given in input. Can be either one of:
            'm': Mach number
            'pressure': Pressure Ratio P/P0
            'density': Density Ratio rho/rho0
            'temperature': Temperature Ratio T/T0
            'crit_area_sub': Critical Area Ratio A/A* for subsonic case.
            'crit_area_super': Critical Area Ratio A/A* for supersonic case.
            'mach_angle': Mach Angle in degrees.
            'prandtl_meyer': Prandtl-Meyer Angle in degrees.
        param_value : float/list/array_like
            Actual value of the parameter. If float, list, tuple is given as
            input, a conversion will be attempted.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M : array_like
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
        ar : array_like
            Critical Area Ratio A/A*
        ma : array_like
            Mach Angle
        pm : array_like
            Prandtl-Meyer Angle
    """

    assert isinstance(gamma, (int, float)) and gamma > 1, "The specific heats ratio must be a number > 1."

    assert isinstance(param_name, str), "param_name must be a string"
    param_name = param_name.lower()
    assert param_name in ['m', 'pressure', 'density', 'temperature', 'crit_area_sub', 'crit_area_super', 'mach_angle', 'prandtl_meyer']
    
    # compute the Mach number
    param_value = np.asarray(param_value)

    M = None
    if param_name == "m":
        M = param_value
        assert np.all(M >= 0), "Mach number must be >= 0."
    elif param_name == "crit_area_sub":
        M = ise.M_From_Critical_Area_Ratio(param_value, "sub", gamma)
    elif param_name == "crit_area_super":
        M = ise.M_From_Critical_Area_Ratio(param_value, "sup", gamma)

    func_dict = {
        'pressure': ise.M_From_Pressure_Ratio,
        'density': ise.M_From_Density_Ratio,
        'temperature': ise.M_From_Temperature_Ratio,
        'mach_angle': ise.M_From_Mach_Angle,
        'prandtl_meyer': ise.M_From_Prandtl_Meyer_Angle,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check(param_value, gamma)

    # compute the different ratios
    pr, dr, tr, prs, drs, trs, urs, ar, ma, pm = ise.Get_Ratios_From_Mach.__no_check(M, gamma)
    
    return M, pr, dr, tr, prs, drs, trs, urs, ar, ma, pm