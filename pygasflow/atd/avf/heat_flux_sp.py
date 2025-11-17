"""
This module contains functions to estimate the heat flux of the gas
at the stagnation point or at a stagnation line.
"""

import numpy as np
from scipy.optimize import bisect
from scipy.constants import sigma
from pygasflow.utils.common import (
    _check_mix_of_units_and_dimensionless,
    _is_pint_quantity
)
import pygasflow


def wall_temperature(
    eps, R, u_inf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tr, Pr, k_inf,
    omega=0.65, laminar=True, phi=0, sphere=True
):
    """
    Compute the wall temperature at a stagnation point or stagnation line
    for a sphere, a cylinder or a swept-cylinder. The wall temperature
    (radiation adiabatic temperature) is computed with the assumption that the
    vehicle surface is radiation cooled and the heat flux into the wall, q_w,
    is small.

    Parameters
    ----------
    eps : float
        Emissivity (0 <= eps <= 1).
    R : float or array_like
        Radius of the sphere or cylinder.
    u_inf : floar or array_like
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
    k_inf : float
        Free stream thermal conductivity of the gas.
    phi : float or array_like, optional.
        Cylinder's sweep angle [radians]. Default to 0 deg: cylinder surface is
        normal to the free stream.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    sphere : bool, optional
        If True, compute the results for a sphere. Otherwise, compute the
        result for a sweep cylinder.

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
    on the employed structure and materials concept (either a cold primary 
    structure with a thermal protection system (TPS), or a hot primary
    structure), and on the flight trajectory segment, the actual wall
    temperature during flight may be somewhat lower, but will be in any case
    near to the radiation-adiabatic temperature.

    Tw < Tra < Tr < Tt

    References
    ----------
    Basic of Aerothermodynamics, Ernst H. Hirschel

    See Also
    --------
    heat_flux

    """
    def func(Tra):
        # eq (7.163)
        return sigma * eps * Tra**4 - heat_flux(R, u_inf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tra, Tr, Pr, k_inf, laminar=laminar, omega=omega, sphere=sphere, phi=phi)

    return bisect(func, 0, 1e05)


def heat_flux(
    R, u_inf, u_grad, Reinf_R, pe_pinf, Ts_Tinf, Tw, Tr, Pr, k_inf,
    phi=0, omega=0.65, laminar=True, sphere=True
):
    """
    Compute the convective heat flux of the gas at the wall at a stagnation 
    point or at a stagnation line for a sphere/sweep cylinder in a 
    laminar/turbulent flow.

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere or cylinder.
    u_inf : floar or array_like
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
    k_inf : float
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
    q_dot : float or array_like

    References
    ----------
    Basic of Aerothermodynamics, Ernst H. Hirschel

    See Also
    --------
    wall_temperature

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
    gsp = C * np.sin(phi)**(1 - 2 * n) * (pe_pinf)**(1 - n) * (Ts_Tinf)**(n * (1 + omega) - 1) * (R / u_inf * u_grad)**n * Reinf_R**(1 - n)
    # eq (7.165)
    return np.cbrt(Pr) * k_inf * gsp / R * (Tr - Tw)


def heat_flux_fay_riddell(
    u_grad, Pr_w, rho_w, mu_w, rho_e, mu_e, he, hw,
    Le=None, hD=None, sphere=True, m=0.52
):
    """
    Compute the convective heat flux of the gas at the wall at a stagnation 
    point or at a stagnation line for a sphere/cylinder in a laminar flow, 
    according to Fay and Riddell.

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
    q_dot : float or array_like

    Examples
    --------

    Compute the convective heat flux using these parameters (coming from
    Exercise 5.2, "Hypersonic Aerothermodynamics", John J. Bertin):

    >>> import pint
    >>> import pygasflow
    >>> from pygasflow.atd.avf.heat_flux_sp import heat_flux_fay_riddell
    >>> from pygasflow.utils.common import canonicalize_pint_dimensions
    >>> ureg = pint.UnitRegistry()
    >>> ureg.formatter.default_format = "~"
    >>> ureg.define("pound_mass = 0.45359237 kg = lbm")
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> lbf, lbm, Btu, ft, s = ureg.lbf, ureg.lbm, ureg.Btu, ureg.ft, ureg.s
    >>> Pr = 0.7368421052631579
    >>> u_grad = 12871.540335275073 * 1 / s
    >>> rho_w = 1.2611943627968788e-05 * lbf * s ** 2 / ft ** 4
    >>> rho_e = 6.525428485981234e-07 * lbf * s ** 2 / ft ** 4
    >>> mu_w = 1.0512765233552152e-06 * lbf * s / ft ** 2
    >>> mu_e = 4.9686546490717815e-06 * lbf * s / ft ** 2
    >>> h_t2 = 11586.824574050748 * Btu / lbm
    >>> h_w = 599.5031167908519 * Btu / lbm
    >>> q = heat_flux_fay_riddell(u_grad, Pr, rho_w, mu_w, rho_e, mu_e, h_t2, h_w, sphere=True)
    >>> q = canonicalize_pint_dimensions(q)
    >>> q
    <Quantity(2.36807802, 'force_pound * second * british_thermal_unit / foot ** 3 / pound_mass')>

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

    if _is_pint_quantity(he) and (hD == 0):
        hD *= he.units

    _check_mix_of_units_and_dimensionless([
        rho_w, rho_e, mu_w, mu_e, he, hw, hD, u_grad
    ])
    # eq (7.161)
    return k * Pr_w**(-0.6) * (rho_w * mu_w)**0.1 * (rho_e * mu_e)**0.4 * (1 + (Le**m - 1) * (hD / he)) * (he - hw) * np.sqrt(u_grad)


def heat_flux_scott(R, u_inf, rho_inf):
    """
    Compute the convective heat flux of the gas at the wall at a stagnation 
    point of a sphere, according to Scott. The heat flux is in [W / cm^2].

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
    q_dot : float or array_like

    Examples
    --------

    >>> import pint
    >>> import pygasflow
    >>> from pygasflow.atd.avf.heat_flux_sp import heat_flux_scott
    >>> ureg = pint.UnitRegistry()
    >>> ureg.formatter.default_format = "~"
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> m, s, kg = ureg.m, ureg.s, ureg.kg
    >>> R = 0.3 * m
    >>> u_inf = 4000 * m / s
    >>> rho_inf = 0.0019662686791414754 * kg / m**3
    >>> q_dot = heat_flux_scott(R, u_inf, rho_inf)
    >>> q_dot
    <Quantity(90.5721895, 'watt / centimeter ** 2')>

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * An AOTV Aeroheating and Thermal Protection Study,  Scott, C. D., Ried,
      R. C., Maraia, R. J., Li, C. P., and Derry, S. M.
    """
    _check_mix_of_units_and_dimensionless([R, rho_inf, u_inf])
    is_pint = _is_pint_quantity(R)
    if is_pint:
        R = R.to("m").magnitude
        rho_inf = rho_inf.to("kg / m**3").magnitude
        u_inf = u_inf.to("m / s").magnitude

    # eq (5.43)
    q_dot = 18300 * np.sqrt(rho_inf / R) * (u_inf / 1e04)**3.05

    if is_pint:
        ureg = pygasflow.defaults.pint_ureg
        q_dot *= ureg.W / ureg.cm**2

    return q_dot


def heat_flux_detra(R, u_inf, rho_inf, u_co, rho_sl, metric=True):
    """
    Compute the convective heat flux of the gas at the wall at a stagnation
    point of a sphere, according to Detra et al. The heat flux is in [W / cm^2]
    or [Btu / ft^2 / s] depending on the value of ``metric``.

    Parameters
    ----------
    R : float or array_like
        Radius of the sphere, in meters if `metric=True`, otherwise in foot.
    u_inf : float or array_like
        Free stream velocity [m / s].
    rho_inf : float or array_like
        Free stream density [kg / m^3].
    u_co : float or array_like
        Circular orbit velocity [m / s].
    rho_sl : float or array_like
        Density at sea level [kg / m^3].
    metric : bool, optional
        If True (default value) use metric system: R [m] and the heat flux
        will be [W / cm^2]. If False, use imperial system: R [ft] and the
        heat flux will be in [Btu / ft^2 / s].

    Returns
    -------
    q_dot : float or array_like

    Examples
    --------

    >>> import pint
    >>> import pygasflow
    >>> from pygasflow.atd.avf.heat_flux_sp import heat_flux_detra
    >>> ureg = pint.UnitRegistry()
    >>> ureg.formatter.default_format = "~"
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> m, s, kg = ureg.m, ureg.s, ureg.kg
    >>> R = 0.3 * m
    >>> u_inf = 4000 * m / s
    >>> u_co = 7950 * m / s
    >>> rho_inf = 0.0019662686791414754 * kg / m**3
    >>> rho_sl = 1.225000018124288 * kg / m**3
    >>> q_dot = heat_flux_detra(R, u_inf, rho_inf, u_co, rho_sl)
    >>> q_dot
    <Quantity(92.7451074, 'watt / centimeter ** 2')>

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * Addendum to Heat Transfer to Satellite Vehicles Reentering the
      Atmosphere, Detra, R. W., Kemp, N. H., and Riddell, F. R
    """
    _check_mix_of_units_and_dimensionless([R, rho_inf, u_inf, u_co, rho_sl])
    is_pint = _is_pint_quantity(R)

    # NOTE: the original correlation was developed using the imperial
    # system, where the constant C=17600 Btu / (ft**1.5 * s).
    # Equation (5.44a) of "Hypersonic Aerothermodynamics" uses the metric
    # constant of 11030. Here, I use the correct value in order to achieve
    # the exact q_dot value when using metric or imperial system.
    # The conversion was achieved with pint:
    #   C_imp = 17600 * ureg.Btu / (ureg.second * ureg.foot**1.5)
    #   C_metric = C_imp.to(ureg.watt * ureg.meter**0.5 / ureg.centimeter**2)
    # Note the `meter**0.5` which is related to the nose radius in meters,
    # and `centimeter**2` which is related to the heat flux.
    C = 11034.832249233914 if metric else 17600
    ureg = None

    if is_pint:
        ureg = pygasflow.defaults.pint_ureg
        if metric:
            C *= ureg.W * ureg.m**0.5 / ureg.cm**2
            R = R.to("m")
            rho_inf = rho_inf.to("kg / m**3")
            rho_sl = rho_sl.to("kg / m**3")
            u_inf = u_inf.to("m / s")
            u_co = u_co.to("m / s")
        else:
            C *= ureg.Btu / (ureg.s * ureg.ft**1.5)
            R = R.to("ft")
            rho_inf = rho_inf.to("lbf * s**2 / ft**4")
            rho_sl = rho_sl.to("lbf * s**2 / ft**4")
            u_inf = u_inf.to("ft / s")
            u_co = u_co.to("ft / s")

    # eq (5.44a), (5.44b)
    q_dot = C / np.sqrt(R) * np.sqrt(rho_inf / rho_sl) * (u_inf / u_co)**3.15
    return q_dot


def heat_flux_radiation_martin(Rn, u_inf, rho_inf, rho_sl, metric=True):
    """
    Compute the gas-to-surface radiation heat flux for a re-entry vehicle.

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
    q_dot_r : float or array_like

    References
    ----------

    * Hypersonic Aerothermodynamics, John J. Bertin
    * Air Radiation Revisited, K. Sutton
    """
    _check_mix_of_units_and_dimensionless([Rn, u_inf, rho_inf, rho_sl])
    is_pint = _is_pint_quantity(Rn)

    # NOTE: Sadly, Bertin doesn't mention what unit system is used in
    # eq (5.46). Looking at problem 5.4 of Hypersonic Aerothermodynamics,
    # which asks to compare the results of eq (5.46) with eq (5.44b),
    # I infer that eq (5.46) uses imperial system. So, Rn [feet],
    # u_inf [feet/s], q [But / (ft**2 * s)].

    if is_pint:
        Rn = Rn.to("feet").magnitude
        u_inf = u_inf.to("feet / s").magnitude
        rho_inf = rho_inf.to("lbf * s**2 / ft**4").magnitude
        rho_sl = rho_sl.to("lbf * s**2 / ft**4").magnitude
    else:
        if metric:
            m2f = 3.2808398950131235
            Rn *= m2f
            u_inf *= m2f

    # eq (5.46)
    q_dot_r =  100 * Rn * (u_inf / 1e04)**8.5 * (rho_inf / rho_sl)**1.6

    if is_pint:
        ureg = pygasflow.defaults.pint_ureg
        q_dot_r *= ureg.Btu / ureg.feet**2 / ureg.s
        if metric:
            q_dot_r = q_dot_r.to("W / cm**2")
    else:
        if metric:
            q_dot_r *= 1.1356528268612096 # (Btu/ft**2/s) to (W/cm**2)

    return q_dot_r
