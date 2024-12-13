"""This module contains functions to estimate the heat flux of the gas
over a flat plate.
"""

import numpy as np
from scipy.optimize import bisect
from scipy.constants import sigma


def wall_temperature(L, x, Reinf_L, Ts_Tinf, Tr, Pr, kinf, eps, sigma=sigma, laminar=True, omega=0.65):
    """Compute the wall temperature (the radiation adiabatic temperature) for
    a radiation cooled flat plate with a prescribed heat flux for a
    compressible flow.

    Parameters
    ----------
    L : float
        Length of the flat plate
    x : float or array_like
        Boundary-layer running length.
    Reinf_L : float
        Free stream Reynolds number computed at L (length of the flat plate).
    Ts_Tinf : float
        Temperature ratio between the reference temperature and the the
        free stream temperature.
    Tr : float
        Recovery temperature.
    Pr : float
        Prandtl number.
    kinf : float
        Free stream thermal conductivity of the gas.
    eps : float
        Surface's emissivity.
    sigma : float, optional
        Boltzmann constant. Default to 5.670374419e-08 W / (m^2 * K^4).
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like
    """
    def func(Tra, x):
        # eq (7.156)
        return sigma * eps * Tra**4 - heat_flux(L, x, Reinf_L, Ts_Tinf, Tra, Tr, Pr, kinf, laminar=laminar, omega=omega)

    x = np.asarray(x)
    if x.shape:
        Tra = np.zeros_like(x, dtype=float)
        for i, _x in enumerate(x):
            Tra[i] = bisect(func, 0, 1e05, args=(_x, ), maxiter=1000)
        return Tra
    return bisect(func, 0, 1e05, args=(x, ))


def heat_flux(L, x, Reinf_L, Ts_Tinf, Tw, Tr, Pr, kinf, laminar=True, omega=0.65):
    """Compute the heat flux of the gas at the surface of a flat plate from a
    prescribed wall temperature in a compressible flow.

    Parameters
    ----------
    L : float
        Length of the flat plate
    x : float or array_like
        Boundary-layer running length.
    Reinf_L : float
        Free stream Reynolds number computed at L (length of the flat plate).
    Ts_Tinf : float
        Temperature ratio between the reference temperature and the the
        free stream temperature.
    Tw : float
        Wall temperature.
    Tr : float
        Recovery temperature.
    Pr : float
        Prandtl number.
    kinf : float
        Free stream thermal conductivity of the gas.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like
    """
    if laminar:
        C = 0.57
        n = 0.5
    else:
        C = 0.0345
        n = 0.21
    # eq (7.157)
    gfp = C * (Ts_Tinf)**(n * (1 + omega) - 1) * (L / x)**n * Reinf_L**(1 - n)
    # eq (7.158)
    return kinf * np.cbrt(Pr) * gfp * (1 / L) * (Tr - Tw)


# if __name__ == "__main__":
#     from pygasflow.atd.avf import reference_temperature
#     from pygasflow.atd.viscosity import viscosity_air_power_law
#     from pygasflow.atd.thermal_conductivity import thermal_conductivity_power_law

#     Tinf = 226.50908361133006
#     rhoinf = 0.01841010086243616
#     muinf = viscosity_air_power_law(Tinf)
#     kinf = thermal_conductivity_power_law(Tinf)
#     gamma = 1.4
#     Minf = 6
#     Lref = 80
#     ainf = np.sqrt(gamma * 287 * Tinf)
#     uinf = Minf * ainf
#     Reinf_L = rhoinf * uinf / muinf * Lref

#     Prs = 0.74  # reference Prandtl number
#     rs_lam = np.sqrt(Prs)
#     rs_tur = np.cbrt(Prs)
#     Me, Te = Minf, Tinf

#     Tw1 = 1000
#     Ts1 = reference_temperature(Te, Tw1, rs=rs_lam, Me=Me, gamma_e=1.4)
#     Ts2 = reference_temperature(Te, Tw1, rs=rs_tur, Me=Me, gamma_e=1.4)

#     Tw2 = 2000
#     Ts3 = reference_temperature(Te, Tw2, rs=rs_lam, Me=Me, gamma_e=1.4)
#     Ts4 = reference_temperature(Te, Tw2, rs=rs_tur, Me=Me, gamma_e=1.4)

#     L = 1
#     x = [0.1, 0.5, 0.8]

#     from pygasflow.atd.nd_numbers import Prandtl
#     Pr = Prandtl(gamma)
#     r = np.sqrt(Pr)
#     Tr = Tinf * (1 + r * (gamma - 1) / 2 * Minf**2)

#     print(Ts1, Tinf, Tr, Pr, kinf)
#     print(wall_temperature(L, x, Reinf_L, Ts1 / Tinf, Tr, Pr, kinf, 0.5))
