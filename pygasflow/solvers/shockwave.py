import numpy as np
import warnings

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
    beta_from_upstream_mach,
    theta_from_mach_beta,
)

from pygasflow.utils.common import convert_to_ndarray
from pygasflow.utils.decorators import check_shockwave

# TODO:
# detachment detection, when provided theta > theta_max for the specified Mach
# number

@check_shockwave([1, 3])
def shockwave_solver(p1_name, p1_value, p2_name="beta", p2_value=90, gamma=1.4, flag="weak"):
    """ 
    Try to compute all the ratios, angles and mach numbers across the shock wave.

    Remember: a normal shock wave has a wave angle beta=90 deg.

    Parameters
    ----------
        p1_name : string
            Name of the first parameter given in input. Can be either one of:
                ['pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2', 'beta', 'theta']
            If the parameter is a ratio, it is in the form downstream/upstream:
                'pressure': p2/p1
                'temperature': t2/t1
                'density': rho2/rho1
                'total_pressure': p02/p01
                'm1': Mach upstream of the shock wave
                'mn1': Normal Mach upstream of the shock wave
                'mn2': Normal Mach downstream of the shock wave
                'beta': The shock wave angle [in degrees]. It can only be used if p2_name='theta'.
                'theta': The deflection angle [in degrees]. It can only be used if p2_name='beta'.
        p1_value : float
            Actual value of the parameter.
        p2_name : string
            Name of the second parameter. It could either be:
                'beta': Shock wave angle.
                'theta: Flow deflection angle.
                'mn1': Input Normal Mach number
            Default to 'beta'.
        p2_value : float
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
        M1 : float
            Mach number upstream of the shock wave.
        Mn1 : float
            Normal Mach number upstream of the shock wave.
        M2 : float
            Mach number downstream of the shock wave.
        Mn2 : float
            Normal Mach number downstream of the shock wave.
        beta : float
            Shock wave angle in degrees.
        theta : float
            Flow deflection angle in degrees.
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
    MN1, M1 = None, None

    p2_name = p2_name.lower()
    if p2_name not in ['beta', 'theta', 'mn1']:
        raise ValueError("p2_name must be either 'beta' or 'theta' or 'mn1.")
    if p2_value is None:
        raise ValueError("p2_value must be a real positive number.")
    if p2_name == 'beta':
        beta = p2_value
        if (not np.all(beta >= 0)) or (not np.all(beta <= 90)):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
    elif p2_name == 'theta':
        theta = p2_value
        # TODO: is this condition correct? The 0 and 90 part????
        if np.any(theta < 0) or np.any(theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
    else:
        MN1 = p2_value
    

    p1_name = p1_name.lower()
    available_p1names = ['beta', 'theta', 'pressure', 'temperature', 'density', 'total_pressure', 'm1', 'mn1', 'mn2']
    if p1_name not in available_p1names:
        raise ValueError("p1_name must be either one of {}".format(available_p1names))
    if p1_name in ['pressure', 'temperature', 'density', 'total_pressure', 'mn2']:
        MN1 = get_upstream_normal_mach_from_ratio.__no_check(p1_name, p1_value, gamma)
    elif p1_name == "mn1":
        if p1_name == p2_name:
            raise ValueError("p1_name must be different than p2_name")
        MN1 = p1_value
    elif p1_name == "m1":
        M1 = p1_value
    elif p1_name == "theta":
        if beta is None:
            raise ValueError("If you provide p1_name='theta', it must be p2_name='beta'.")
        theta = p1_value
        if (theta < 0) or (theta > 90):
            raise ValueError("The flow angle theta must be 0 <= theta <= 90.")
        if not isinstance(beta, np.ndarray):
            beta = beta * np.ones_like(theta)
        M1 = mach_from_theta_beta.__no_check(theta, beta)
    elif p1_name == "beta":
        if theta is None:
            raise ValueError("If you provide p1_name='beta', it must be p2_name='theta'.")
        beta = p1_value
        if (beta < 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 <= beta <= 90.")
        if not isinstance(theta, np.ndarray):
            theta = theta * np.ones_like(beta)
        M1 = mach_from_theta_beta.__no_check(theta, beta)
    else:   # 'm2'
        # TODO:
        # Is it even possible to solve it knowing only M2, beta or M2, theta?????
        raise NotImplementedError("Solving a shock wave with a given M2 is not yet implemented.")

    if (M1 is not None) and (MN1 is not None):
        beta = beta_from_upstream_mach.__no_check(M1, MN1)
        theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    elif M1 is not None:
        # at this point, either beta or theta is set, not both!
        MN1 = normal_mach_upstream.__no_check(M1, beta, theta, gamma, flag)
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)

        if beta is not None:
            theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
        else:
            beta = beta_from_mach_theta.__no_check(M1, theta, gamma)[flag]
        
        if isinstance(M1, (list, tuple, np.ndarray)):
            beta *= np.ones_like(M1)
            theta *= np.ones_like(M1)
        
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    else:
        # compute the different ratios
        pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream.__no_check(MN1, gamma)

        if beta is not None:
            M1 = MN1 / np.sin(np.deg2rad(beta))
            theta = theta_from_mach_beta.__no_check(M1, beta, gamma)
            if isinstance(M1, (list, tuple, np.ndarray)):
                beta *= np.ones_like(M1)
        else:
            # TODO:
            # Is it even possible to uniquely determine M1 = f(MN1, beta)????

            # M1 = Upstream_Mach_From_Normal_Mach_Theta(MN1, theta, flag, gamma)
            # beta = Beta_From_Mach_Theta(M1, theta, gamma)[flag]
            M1 = np.nan * np.ones_like(MN2)
            beta = np.nan * np.ones_like(MN2)
            warnings.warn("Undetermined case. Setting M1 = beta = M2 = NaN")
        M2 = MN2 / np.sin(np.deg2rad(beta - theta))
    
    # TODO
    # 1. What if p1_name is M2????
    #     
    return M1, MN1, M2, MN2, beta, theta, pr, dr, tr, tpr

@check_shockwave
def conical_shockwave_solver(M1, param_name, param_value, gamma=1.4, flag="weak"):
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
        flag : string
            Can be either 'weak' or 'strong'. Default to 'weak' (in conical
            shockwaves, the strong solution is rarely encountered).
    
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
    param_name = param_name.lower()
    if param_name not in ["mc", "beta", "theta_c"]:
        raise ValueError("param_name can be either 'beta' or 'mc' or 'theta_c'.")

    Mc, beta, theta_c = None, None, None
    if param_name == 'mc':
        Mc = param_value
        if np.any(M1 <= Mc):
            raise ValueError("It must be M1 > Mc.")
        if (not isinstance(Mc, (int, float))) or (Mc < 0):
            raise ValueError("The Mach number at the cone's surface must be Mc >= 0.")
    elif param_name == 'beta':
        beta = param_value
        if (not isinstance(beta, (int, float))) or (beta <= 0) or (beta > 90):
            raise ValueError("The shock wave angle must be 0 < beta <= 90.")
    else:
        theta_c = param_value
        if (not isinstance(theta_c, (int, float))) or (theta_c <= 0) or (theta_c > 90):
            raise ValueError("The half cone angle must be 0 < theta_c < 90.")
    
    if Mc:
        _, theta_c, beta = shock_angle_from_machs(M1, Mc, gamma, flag)
    elif beta:
        Mc, theta_c = mach_cone_angle_from_shock_angle(M1, beta, gamma)
    elif theta_c:
        Mc, _, beta = shock_angle_from_mach_cone_angle(M1, theta_c, gamma, flag)
    
    # compute the ratios across the shockwave
    MN1 = M1 * np.sin(np.deg2rad(beta))
    pr, dr, tr, tpr, MN2 = get_ratios_from_normal_mach_upstream(MN1, gamma)
    
    # delta is the flow deflection angle (Anderson's Figure 10.4)
    delta = theta_from_mach_beta(M1, beta, gamma)
    M2 = MN2 /  np.sin(np.deg2rad(beta - delta))

    # ratios between cone surface and upstream conditions. Note that
    # p0c/p01 = p02/p01, already computed
    pc_p1 = ise_PR(Mc) * tpr / ise_PR(M1)
    rhoc_rho1 = ise_DR(Mc) / ise_DR(M2) * dr
    Tc_T1 = ise_TR(Mc) / ise_TR(M2) * tr

    # set Mc, theta_c to have the same shape as M1 and the other ratios. This is
    # necessary because Mc or theta_c are parameters passed in by the user, in 
    # that case they are scalars.
    theta_c = theta_c * np.ones_like(M1)
    if not isinstance(Mc, np.ndarray):
        Mc = Mc * np.ones_like(M1)

    return M1, Mc, theta_c, beta, delta, pr, dr, tr, tpr, pc_p1, rhoc_rho1, Tc_T1

if __name__ == "__main__":
    # print(conical_shockwave_solver(2, "theta_c", 38.76391345757847))
    # print(conical_shockwave_solver(2, "beta", 61.485371643068866))
    # print(conical_shockwave_solver(3, "mc", 2))
    # print(conical_shockwave_solver(2, 'theta_c', 45))
    # print(shockwave_solver("m1", [2, 5], "theta", [20, 20]))
    # print(shockwave_solver("beta", 60, "theta", 45))
    # print(conical_shockwave_solver(2, "beta", 20))
    print(conical_shockwave_solver(2, "mc", 0.8))