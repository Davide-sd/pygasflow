import numpy as np
import pygasflow.rayleigh as ray

def Rayleigh_Solver(param_name, param_value, gamma=1.4):
    """
    Compute all Rayleigh ratios and Mach number given an input parameter.

    Parameters
    ----------
        param_name : string
            Name of the parameter given in input. Can be either one of:
            'm': Mach number
            'pressure': CriticalPressure Ratio P/P*
            'density': CriticalDensity Ratio rho/rho*
            'velocity': Critical Velocity Ratio U/U*.
            'temperature_sub': Critical Temperature Ratio T/T for subsonic case.
            'temperature_super': Critical Temperature Ratio T/T* for supersonic case.
            'total_pressure_sub': Critical Total Pressure Ratio P0/P0* for subsonic case.
            'total_pressure_super': Critical Total Pressure Ratio P0/P0* for supersonic case.
            'total_temperature_sub': Critical Total Temperature Ratio T0/T0* for subsonic case.
            'total_temperature_super': Critical Total Temperature Ratio T0/T0* for supersonic case.
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
        tprs : array_like
            Critical Total Temperature Ratio T0/T0*
        urs : array_like
            Critical Velocity Ratio U/U*
        eps : array_like
            Critical Entropy Ratio (s*-s)/R
    """

    assert isinstance(gamma, (int, float)) and gamma > 1, "The specific heats ratio must be a number > 1."

    assert isinstance(param_name, str), "param_name must be a string"
    param_name = param_name.lower()
    assert param_name in ['m', 'pressure', 'density', 'velocity', 
                        'temperature_sub', 'temperature_super', 
                        'total_pressure_sub', 'total_pressure_super', 
                        'total_temperature_sub', 'total_temperature_super', 
                        'entropy_sub', 'entropy_super'], "param_name not recognized."
    
    # compute the Mach number
    param_value = np.asarray(param_value)

    M = None
    if param_name == "m":
        M = param_value
        assert np.all(M >= 0), "Mach number must be >= 0."
    elif param_name == 'total_pressure_sub':
        M = ray.M_From_Critical_Total_Pressure_Ratio(param_value, "sub", gamma)
    elif param_name == 'total_pressure_super':
        M = ray.M_From_Critical_Total_Pressure_Ratio(param_value, "sup", gamma)
    elif param_name == 'total_temperature_sub':
        M = ray.M_From_Critical_Total_Temperature_Ratio(param_value, "sub", gamma)
    elif param_name == 'total_temperature_super':
        M = ray.M_From_Critical_Total_Temperature_Ratio(param_value, "sup", gamma)
    elif param_name == 'temperature_sub':
        M = ray.M_From_Critical_Temperature_Ratio(param_value, "sub", gamma)
    elif param_name == 'temperature_super':
        M = ray.M_From_Critical_Temperature_Ratio(param_value, "sup", gamma)
    elif param_name == 'entropy_sub':
        M = ray.M_From_Critical_Entropy(param_value, "sub", gamma)
    elif param_name == 'entropy_super':
        M = ray.M_From_Critical_Entropy(param_value, "sup", gamma)

    func_dict = {
        'pressure': ray.M_From_Critical_Pressure_Ratio,
        'density': ray.M_From_Critical_Density_Ratio,
        'velocity': ray.M_From_Critical_Velocity_Ratio,
    }
    if param_name in func_dict.keys():
        M = func_dict[param_name].__bypass_decorator(param_value, gamma)

    # compute the different ratios
    prs, drs, trs, tprs, ttrs, urs, eps = ray.Get_Ratios_From_Mach(M, gamma)
    
    return M, prs, drs, trs, tprs, ttrs, urs, eps