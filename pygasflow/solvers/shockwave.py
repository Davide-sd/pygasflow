import numpy as np

from pygasflow.isentropic import (
    pressure_ratio as ise_PR,
    density_ratio as ise_DR,
    temperature_ratio as ise_TR
)

from pygasflow.shockwave import (
    mach_from_theta_beta,
    get_upstream_normal_mach_from_ratio,
    get_ratios_from_normal_mach_upstream,
    normal_mach_upstream,
    theta_from_mach_beta,
    beta_from_mach_theta,
    shock_angle_from_machs,
    mach_cone_angle_from_shock_angle,
    shock_angle_from_mach_cone_angle,
)

from pygasflow.utils.common import convert_to_ndarray

def shockwave_solver(param_name, param_value, angle_name="beta", angle_value=90, gamma=1.4, flag="weak"):
    """ 
    Try to compute all the ratios, angles and mach numbers across the shock wave.

    Remember: a normal shock wave has a wave angle beta=90 deg.

    Parameters
    ----------
        param_name : string
            Name of the parameter given in input. Can be either one of:
                ['pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2', 'beta', 'theta']
            If the parameter is a ratio, it is in the form downstream/upstream:
                'pressure': p2/p1
                'temperature': t2/t1
                'density': rho2/rho1
                'total_pressure': p02/p01
                'm1': Mach upstream of the shock wave
                'mn1': Normal Mach upstream of the shock wave
                'mn2': Normal Mach downstream of the shock wave
                'beta': The shock wave angle [in degrees]. It can only be used if angle_name='theta'.
                'theta': The deflection angle [in degrees]. It can only be used if angle_name='beta'.
        param_value : float
            Actual value of the parameter.
        angle_name : string
            Name of the angle given as parameter. It could either be:
                'beta': Shock wave angle.
                'theta: Flow deflection angle.
            Default to 'beta'.
        angle_value : float
            Value of the angle in degrees.
            Default to 90 degrees (normal shock wave)
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        flag : string
            Chose what solution to compute if the angle 'theta' is provided.
            Can be either 'weak' or 'strong'. 
            Default to 'weak'. 
    
    Returns
    -------
        beta : float
            Shock wave angle in degrees.
        theta : float
            Flow deflection angle in degrees.
        M1 : float
            Mach number upstream of the shock wave.
        Mn1 : float
            Normal Mach number upstream of the shock wave.
        M2 : float
            Mach number downstream of the shock wave.
        Mn2 : float
            Normal Mach number downstream of the shock wave.
        pr : float
            Pressure ratio across the shock wave.
        dr : float
            Density ratio across the shock wave.
        tr : float
            Temperature ratio across the shock wave.
        tpr : float
            Total Pressure ratio across the shock wave.
    """

    beta, theta = None, None    
    angle_name = angle_name.lower()       
    assert angle_name in ['beta', 'theta'], "Angle name must be either 'beta' or 'theta'."
    assert angle_value != None and isinstance(angle_value, (float, int)), "The angle value must be a real positive number."
    if angle_name == 'beta':
        beta = angle_value
        assert beta >= 0 and beta <= 90, "The shock wave angle must be 0 <= beta <= 90."
    else:
        theta = angle_value
        # TODO: is this assert correct? The 0 and 90 part????
        assert theta != None and theta >= 0 and theta <= 90, "The flow angle theta must be 0 <= theta <= 90."
    

    MN1, M1 = None, None

    param_name = param_name.lower()
    assert param_name in ['beta', 'theta', 'pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2'], "param_name must be either one of ['pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2']."
    
    if param_name in ['pressure', 'temperature', 'density', 'total_pressure', 'mn2']:
        MN1 = get_upstream_normal_mach_from_ratio(param_name, param_value, gamma)
    elif param_name == "mn1":
        MN1 = param_value
    elif param_name == "m1":
        M1 = param_value
    elif param_name == "theta":
        assert beta != None, "If you provide param_name='theta', it must be angle_name='beta'."
        theta = param_value
        assert theta >= 0 and theta <= 90, "The flow angle theta must be 0 <= theta <= 90."
        M1 = mach_from_theta_beta(theta, beta)
        # pass
    elif param_name == "beta":
        assert theta != None, "If you provide param_name='beta', it must be angle_name='theta'."
        beta = param_value
        assert beta >= 0 and beta <= 90, "The shock wave angle must be 0 <= beta <= 90."
        M1 = mach_from_theta_beta(theta, beta)
    else:   # 'm2'
        # TODO:
        # Is it even possible to solve it knowing only M2, beta or M2, theta?????
        raise NotImplementedError("Solving a shock wave with a given M2 is not yet implemented.")


    flag = flag.lower()
    assert flag in ["weak", "strong"], "Flag can be either 'weak' or 'strong'."

    
    if M1:
        # at this point, either beta or theta is set, not both!
        MN1 = normal_mach_upstream(M1, beta, theta, flag)
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)

        if beta:
            theta = theta_from_mach_beta(M1, beta, gamma)
        else:
            beta = beta_from_mach_theta(M1, theta, gamma)[flag]
        
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    else:
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)

        if beta:
            M1 = MN1 / np.sin(np.deg2rad(beta))
            theta = theta_from_mach_beta(M1, beta, gamma)
        else:
            # TODO:
            # Is it even possible to uniquely determine M1 = f(MN1, beta)????

            # M1 = Upstream_Mach_From_Normal_Mach_Theta(MN1, theta, flag, gamma)
            # beta = Beta_From_Mach_Theta(M1, theta, gamma)[flag]
            M1 = np.nan
            beta = np.nan
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    
    # TODO
    # 1. What if param_name is M2????
    #     
    return M1, MN1, M2, MN2, beta, theta, pr, dr, tr, tpr


def conical_shockwave_solver(M1, param_name, param_value, gamma=1.5, step=0.025):
    """ 
    Try to compute all the ratios, angles and mach numbers across the conical shock wave.

    Parameters
    ----------
        M1 : float
            Upstream Mach number. Must be M1 > 1.
        param_name : string
            Name of the parameter given in input. Can be either one of:
                ['mc', 'theta_c', 'beta']
                'mc': Mach number at the cone's surface.
                'theta_c': Half cone angle.
                'beta': shock wave angle.
        param_value : float
            Actual value of the parameter. Requirements:
                Mc >= 0
                0 < beta <= 90
                0 < theta_c < 90
        gamma : float
            Specific heats ratio. Default to 1.4. Must be > 1.
        step : float
            Angle-increment used on the shock wave angle iteration. Default to 0.025 deg.
    
    Returns
    -------
        M : float
            Upstream Mach number.
        Mc : float
            Mach number at the surface of the cone.
        theta_c : float
            Half cone angle.
        beta : float
            Shock wave angle.
        delta : float
            Flow deflection angle.
        pr : float
            Pressure ratio across the shock wave.
        dr : float
            Density ratio across the shock wave.
        tr : float
            Temperature ratio across the shock wave.
        tpr : float
            Total Pressure ratio across the shock wave.
        pc_p1 : float
            Pressure ratio between the cone's surface and the upstream condition.
        rhoc_rho1 : float
            Density ratio between the cone's surface and the upstream condition.
        Tc_T1 : float
            Temperature ratio between the cone's surface and the upstream condition.
    """
    assert isinstance(M1, (int, float)) and M1 > 1, "The upstream Mach number must be > 1."
    
    param_name = param_name.lower()
    assert param_name in ["mc", "beta", "theta_c"], "param_name can be either 'beta' or 'mc' or 'theta_c'."

    Mc, beta, theta_c = None, None, None
    if param_name == 'mc':
        Mc = param_value
        assert isinstance(Mc, (int, float)) and Mc >= 0, "The Mach number at the cone's surface must be Mc >= 0."
        
    elif param_name == 'beta':
        beta = param_value
        assert isinstance(beta, (int, float)) and beta > 0 and beta <= 90, "The shock wave angle must be 0 < beta <= 90."
    else:
        theta_c = param_value
        assert isinstance(theta_c, (int, float)) and theta_c > 0 and theta_c < 90, "The half cone angle must be 0 < theta_c < 90."
    
    if Mc:
        _, theta_c, beta = shock_angle_from_machs(M1, Mc, gamma, step)
    elif beta:
        Mc, theta_c = mach_cone_angle_from_shock_angle(M1, beta, gamma)
    elif theta_c:
        Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, step)
    
    # compute the ratios across the shockwave
    MN1 = M1 * np.sin(np.deg2rad(beta))
    pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)
    
    # delta is the flow deflection angle (Anderson's Figure 10.4)
    delta = theta_from_mach_beta(M1, beta, gamma)
    M_2 = MN2 /  np.sin(np.deg2rad(beta - delta))

    # ratios between cone surface and upstream conditions. Note that
    # p0c/p01 = p02/p01, already computed
    pc_p1 = ise_PR(Mc) * tpr / ise_PR(M1)
    rhoc_rho1 = ise_DR(Mc) / ise_DR(M_2) * dr
    Tc_T1 = ise_TR(Mc) / ise_TR(M_2) * tr

    return M1, Mc, theta_c, beta, delta, pr, dr, tr, tpr, pc_p1, rhoc_rho1, Tc_T1