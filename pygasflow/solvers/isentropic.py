import numpy as np
import pygasflow.isentropic as ise
from pygasflow.utils.common import convert_to_ndarray

def isentropic_solver(param_name, param_value, gamma=1.4):
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
    param_value = convert_to_ndarray(param_value)

    M = None
    if param_name == "m":
        M = param_value
        assert np.all(M >= 0), "Mach number must be >= 0."
    elif param_name == "crit_area_sub":
        M = ise.m_from_critical_area_ratio(param_value, "sub", gamma)
    elif param_name == "crit_area_super":
        M = ise.m_from_critical_area_ratio(param_value, "super", gamma)

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
    
    return M, pr, dr, tr, prs, drs, trs, urs, ar, ma, pm



def print_isentropic(M, pr, dr, tr, prs, drs, trs, urs, ar, ma, pm):
    print("M \t\t {}".format(M))
    print("P/P0 \t\t {}".format(pr))
    print("rho/rho0 \t {}".format(dr))
    print("T/T0 \t\t {}".format(tr))
    print("P/P* \t\t {}".format(prs))
    print("rho/rho* \t {}".format(drs))
    print("T/T* \t\t {}".format(trs))
    print("U/U* \t\t {}".format(urs))
    print("A/A* \t\t {}".format(ar))
    print("Mach Angle \t {}".format(ma))
    print("Prandtl-Meyer \t {}".format(pm))
    print()


def main():
    s = isentropic_solver('m', 2, gamma=1.4)
    print(s)
    print_isentropic(*s)

if __name__ == "__main__":
    main()