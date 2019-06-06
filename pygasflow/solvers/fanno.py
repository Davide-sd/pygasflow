import numpy as np
import pygasflow.fanno as fanno

def Fanno_Solver(param_name, param_value, gamma=1.4):
    """
    Compute all Fanno ratios and Mach number given an input parameter.

    Parameters
    ----------
        param_name : string
            Name of the parameter given in input. Can be either one of:
            'm': Mach number
            'pressure': CriticalPressure Ratio P/P*
            'density': CriticalDensity Ratio rho/rho*
            'temperature': CriticalTemperature Ratio T/T*
            'total_pressure_sub': Critical Total Pressure Ratio P0/P0* for subsonic case.
            'total_pressure_super': Critical Total Pressure Ratio P0/P0* for supersonic case.
            'velocity': Critical Velocity Ratio U/U*.
            'friction_sub': Critical Friction parameter 4fL*/D for subsonic case.
            'friction_super': Critical Friction parameter 4fL*/D for supersonic case.
            'entropy_sub': Entropy parameter (s*-s)/R for subsonic case.
            'entropy_super': Entropy parameter (s*-s)/R for supersonic case.
        param_value : float/list/array_like
            Actual value of the parameter. If float, list, tuple is given as
            input, a conversion will be attempted.
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
    
    Returns
    -------
        M : array_like
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
    """

    assert isinstance(gamma, (int, float)) and gamma > 1, "The specific heats ratio must be a number > 1."

    assert isinstance(param_name, str), "param_name must be a string"
    param_name = param_name.lower()
    assert param_name in ['m', 'pressure', 'density', 'temperature', 'total_pressure_sub', 'total_pressure_super', 'velocity', 'friction_sub', 'friction_super', 'entropy_sub', 'entropy_super'], "param_name not recognized."
    
    # compute the Mach number
    param_value = np.asarray(param_value)

    M = None
    if param_name == "m":
        M = param_value
        assert np.all(M >= 0), "Mach number must be >= 0."
    elif param_name == 'total_pressure_sub':
        M = fanno.M_From_Critical_Total_Pressure_Ratio(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = fanno.M_From_Critical_Total_Pressure_Ratio(param_value, "sup", gamma)
    elif param_name == 'friction_sub':
        M = fanno.M_From_Critical_Friction(param_value, "sub", gamma)
    elif param_name == 'friction_super':
        M = fanno.M_From_Critical_Friction(param_value, "sup", gamma)
    elif param_name == 'entropy_sub':
        M = fanno.M_From_Critical_Entropy(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = fanno.M_From_Critical_Entropy(param_value, "sup", gamma)

    func_dict = {
        'pressure': fanno.M_From_Critical_Pressure_Ratio,
        'density': fanno.M_From_Critical_Density_Ratio,
        'temperature': fanno.M_From_Critical_Temperature_Ratio,
        'velocity': fanno.M_From_Critical_Velocity_Ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__no_check(param_value, gamma)

    # compute the different ratios
    prs, drs, trs, tprs, urs, fps, eps = fanno.Get_Ratios_From_Mach(M, gamma)
    
    return M, prs, drs, trs, tprs, urs, fps, eps