"""This module contains functions to estimate the heat flux of the gas
at the stagnation point or at a stagnation line.
"""

import numpy as np
from scipy.optimize import bisect
from scipy.constants import sigma


def wall_temperature(eps, R, uinf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tr, Pr, kinf, laminar=True, omega=0.65, sphere=True, phi=0):
    """Compute the wall temperature at a stagnation point or stagnation line
    for a sphere, a cylinder or a swept-cylinder. The wall temperature
    (radiation adiabatic temperature) is computed with the assumption that the
    vehicle surface is radiation cooled and the heat flux into the wall, q_w,
    is small.

    Notes
    -----
    The general heat balance is: q_w = q_rad - q_gw
    where q_w is the heat flux into the wall, q_gw is the heat flux in the gas
    at the wall, q_rad is the heat flux radiated away.

    Quoting from the book:

    In the assumption that q_w is small, then q_gw = q_rad: the heat flux
    coming to the surface is radiated away from it. Hence, the
    "radiation-adiabatic temperature" Tra will result: no heat is exchanged
    between gas and material, but the surface radiates heat away.

    With steady flow conditions and a steady heat flux q_w into the wall,
    Tra also is a conservative estimate of the surface temperature. Depending
    on the employed structure and materials concept (either a cold primary structure with a thermal protection system (TPS), or a hot primary
    structure), and on the flight trajectory segment, the actual wall
    temperature during flight may be somewhat lower, but will be in any case
    near to the radiation-adiabatic temperature.

    Tw < Tra < Tr < Tt

    References
    ----------
    Basic of Aerothermodynamics, Ernst H. Hirschel

    """
    def func(Tra):
        # eq (7.163)
        return sigma * eps * Tra**4 - heat_flux(R, uinf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tra, Tr, Pr, kinf, laminar=laminar, omega=omega, sphere=sphere, phi=phi)

    return bisect(func, 0, 1e05)


def heat_flux(R, uinf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tw, Tr, Pr, kinf, sphere=True, phi=0, laminar=True, omega=0.65):
    """Compute the heat flux of the gas at the wall at a stagnation point or
    at a stagnation line for a sphere/sweep cylinder in a laminar/turbulent
    flow.

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere or cylinder.
    uinf : floar or array_like
        Free stream velocity.
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
    sphere : bool, optional
        If True, compute the results for a sphere. Otherwise, compute the
        result for a sweep cylinder.
    phi : float or array_like, optional.
        Cylinder's sweep angle [radians]. Default to 0 deg: cylinder surface is
        normal to the free stream.
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


def heat_flux_fay_riddell(u_grad, Pr_w, rho_w, mu_w, rho_e, mu_e, he, hw, Le=None, hD=None, sphere=True, m=0.52):
    """Compute the heat flux of the gas at the wall at a stagnation point or
    at a stagnation line for a sphere/cylinder in a laminar flow, according
    to Fay and Riddell.

    Parameters
    ----------
    u_grad : float or array_like
        Velocity gradient at the stagnation line.
    Pr_w : float or array_like
        Prandtl number.
    rho_w : float or array_like
        Density at the wall.
    mu_w : float or array_like
        Viscosity at the wall.
    rho_e : float or array_like
        Density at the edge of the boundary layer.
    mu_e : float or array_like
        Viscosity at the edge of the boundary layer.
    Le : float or array_like
        Lewis number. Default to None, indicating perfect gas (which is
        equivalent to set ``Le=1``).
    hD : float or array_like
        Average atomic dissociation energy multiplied by the atom mass
        fraction at the edge of the boundary layer.
    he : float or array_like
        Boundary-layer edge enthalpy.
    hw : float or array_like
        Wall enthalpy.
    sphere : bool, optional
        If True, compute the results for a sphere. Otherwise, compute the
        result for a 2D cylinder.
    m : float, optional
        Default to 0.52 (for equilibrium case). Set ``m=0.63`` for the frozen
        case.

    Returns
    -------
    out : float or array_like

    References
    ----------

    * Basic of Aerothermodynamics, Ernst H. Hirschel
    * Theory of Stagnation Point Heat Transfer in Dissociated Gas,
      J. A. Fay and  F. R. Riddell
    """
    k = 0.763
    if sphere is False:
        k = 0.57
    if Le is None:
        Le = 0
        hD = 0
    elif hD is None:
        raise ValueError(
            "When the Lewis number is provided, hD must be provided too.")
    # eq (7.161)
    return k * Pr_w**(-0.6) * (rho_w * mu_w)**0.1 * (rho_e * mu_e)**0.4 * (1 + (Le**m - 1) * (hD / he)) * (he - hw) * np.sqrt(u_grad)


def heat_flux_scott(R, u_inf, rho_inf):
    """Compute the heat flux of the gas at the wall at a stagnation point of a
    sphere, according to Scott. The heat flux is in [W / cm^2]

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere [m].
    u_inf : float or array_like
        Free stream velocity [m / s].
    rho_inf : float or array_like
        Free stream density [kg / m^3].

    Returns
    -------
    out : float or array_like

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * An AOTV Aeroheating and Thermal Protection Study,  Scott, C. D., Ried,
      R. C., Maraia, R. J., Li, C. P., and Derry, S. M.
    """
    # eq (5.43)
    return 18300 * np.sqrt(rho_inf / R) * (u_inf / 1e04)**3.05


def heat_flux_detra(R, u_inf, rho_inf, u_co, rho_sl, metric=True):
    """Compute the heat flux of the gas at the wall at a stagnation
    point of a sphere, according to Detra et al. The heat flux is in [W / cm^2]
    or [Bt / ft^2] depending on the value of ``metric``.

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere [m].
    u_inf : float or array_like
        Free stream velocity [m / s].
    rho_inf : float or array_like
        Free stream density [kg / m^3].
    u_co : float or array_like
        Circular orbit velocity [m / s].
    rho_sl : float or array_like
        Density at sea level [kg / m^3].
    metric : bool, optional
        If True (default value) use metric system: Rn [m] and the heat flux
        will be [W / cm^2]. If False, use imperial system: Rn [ft] and the
        heat flux will be in [Btu / ft^2].

    Returns
    -------
    out : float or array_like

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * Addendum to Heat Transfer to Satellite Vehicles Reentering the
      Atmosphere, Detra, R. W., Kemp, N. H., and Riddell, F. R
    """
    if metric:
        # eq (5.44a)
        return 11030 / np.sqrt(R) * np.sqrt(rho_inf / rho_sl) * (u_inf / u_co)**3.15
    # eq (5.44b)
    return 17600 / np.sqrt(R) * np.sqrt(rho_inf / rho_sl) * (u_inf / u_co)**3.15


def heat_flux_radiation_martin(Rn, u_inf, rho_inf, rho_sl, metric=True):
    """Compute the gas-to-surface radiation heat flux for a re-entry vehicle.

    Parameters
    ----------
    Rn : float or array_like
        Blunt node radius.
    u_inf : float or array_like
        Free stream velocity.
    rho_inf : float or array_like
        Free stream density.
    rho_sl : float or array_like
        Density at sea level.
    metric : bool, optional
        If True (default value) use metric system: Rn [m] and u_inf [m/s].
        If False, use imperial system: Rn [ft] and u_inf [ft/s].

    Returns
    -------
    out : float or array_like

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * Air Radiation Revisited, K. Sutton
    """
    f2m = 0.3048
    if not metric:
        f2m = 1
    # eq (5.46)
    return 100 * Rn * f2m * (u_inf * f2m / 1e04)**8.5 * (rho_inf / rho_sl)**1.6
