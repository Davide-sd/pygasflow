"""This module contains functions to estimate the heat flux of the gas
at the stagnation point or at a stagnation line.
"""

import numpy as np
from scipy.optimize import bisect
from scipy.constants import sigma

def wall_temperature(eps, R, phi, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=True):
# def wall_temperature(eps, hf):
    """
    """
    def func(Tra):
        # return sigma * eps * Tra**4 - hf
        return sigma * eps * Tra**4 - heat_flux(R, phi, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tra, Tr, Pr, kinf, laminar=laminar, omega=omega, sphere=sphere)

    return bisect(func, 0, 1e05)


def heat_flux(R, phi, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tw, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=True):
    """Compute the heat flux of the gas at the wall at a stagnation point or
    at a stagnation line for a sphere/sweep cylinder in a laminar/turbulent
    flow.

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere or cylinder.
    phi : float or array_like
        Sweep angle [radians]
    u_grad : float or array_like
        Velocity gradient at the stagnation line.
    Reinf_R : float or array_like
        Free stream Reynolds number computed at R.
    pe_pinf : float or array_like
        Pressure ratio between the pressure at the edge of the boundary layer
        and the free stream pressure.
    Ts_Tinf : float or array_like
        Temperature ratio between the reference temperature and the the
        free stream temperature.
    Tw : float or array_like
        Wall temperature.
    Tr : float or array_like
        Recovery temperature.
    Pr : float or array_like
        Prandtl number.
    kinf : float
        Free stream thermal conductivity of the gas.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    sphere : bool, optional
        If True, compute the results for a sphere. Otherwise, compute the
        result for a sweep cylinder.

    Returns
    -------
    out : float or array_like
    """

    if (laminar is False) and sphere:
        raise NotImplementedError(
            "sphere and turbulent flow not yet supported.")
    if sphere:
        C = 0.763
        n = 0.5
    else:
        C = 0.57 if laminar else 0.0345
        n = 0.5 if laminar else 0.21
    # eq (7.164)
    gsp = C * np.sin(phi)**(1 - 2 * n) * (pe_pinf)**(1 - n) * (Ts_Tinf)**(n * (1 + omega) - 1) * (R / uinf * u_grad)**n * Reinf_R**(1 - n)
    # eq (7.165)
    return np.cbrt(Pr) * kinf * gsp / R * (Tr - Tw)


if __name__ == "__main__":
    from ambiance import Atmosphere
    a = Atmosphere(30e03)
    Tinf = a.temperature[0]
    pinf = a.pressure[0]
    rhoinf = a.density[0]
    Minf = 6
    gamma = 1.4
    ainf = np.sqrt(gamma * 287 * Tinf)
    uinf = Minf * ainf
    from pygasflow.atd.nd_numbers import Prandtl
    Pr = Prandtl(gamma)
    R = 0.25
    phi = np.deg2rad(30)
    from pygasflow.atd.avf.wall_shear_stress_sp import velocity_gradient
    from pygasflow import shockwave_solver as ss
    res = ss("m1", 6, to_dict=True)
    ps = (pinf + 0.5 * rhoinf * uinf**2) * res["tpr"]
    rhos = rhoinf * res["dr"]
    u_grad = velocity_gradient(R, pinf, 2 * ps, rhos)
    from pygasflow.atd.viscosity import viscosity_air_power_law
    muinf = viscosity_air_power_law(Tinf)
    Reinf_R = rhoinf * uinf / muinf * R
    from pygasflow.atd.avf import reference_temperature
    Tw = 1000
    r = np.cbrt(Pr)
    Tr = Tinf * (1 + r * (gamma - 1) / 2 * Minf**2)
    Ts = reference_temperature(Tinf, Tw, Tr=Tr)
    from pygasflow.atd.thermal_conductivity import thermal_conductivity_power_law
    kinf = thermal_conductivity_power_law(Tinf)
    print(heat_flux(R, phi, u_grad, Reinf_R, ps / pinf, Ts / Tinf, Tw, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=True))
    print(heat_flux(R, phi, u_grad, Reinf_R, ps / pinf, Ts / Tinf, Tw, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=False))
    print(heat_flux(R, phi, u_grad, Reinf_R, ps / pinf, Ts / Tinf, Tw, Tr, Pr, kinf, laminar=False, omega=0.65, sphere=False))

    print(wall_temperature(0.5, R, phi, u_grad, Reinf_R, ps / pinf, Ts / Tinf, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=True))
    # print(wall_temperature(0.5, 5763.237665003821))
    Tw = 671.4616249379453
    Ts = reference_temperature(Tinf, Tw, Tr=Tr)
    print(heat_flux(R, phi, u_grad, Reinf_R, ps * 2 / pinf, Ts / Tinf, Tw, Tr, Pr, kinf, laminar=False, omega=0.65, sphere=False))
