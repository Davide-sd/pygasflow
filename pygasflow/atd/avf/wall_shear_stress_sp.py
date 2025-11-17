"""
This module contains functions to estimate the **wall shear stress along
the attachment line of a swept cylinder for a compressible, laminar/turbulent
flow**.
"""

import numpy as np


def velocity_gradient(R, pinf, ps, rhos, k=1.33, phi=0):
    """Compute the gradient of the inviscid velocity along the stagnation point
    or a stagnation line.

    Parameters
    ----------
    R : float or array_like
        Radius of the infinite-long swept cylinder.
    pinf : float or array_like
        Free stream pressure.
    ps : float or array_like
        Pressure at the stagnation point.
    rhos : float or array_like
        Density at the stagnation point.
    k : float, optional
        Proportionality factor. Default to ``k=1.33`` for a cylinder.
        Set ``k=1`` for a sphere.
    phi : float or array_like, optional
        Sweep angle in case of cylinder [radians]. ``phi=0`` corresponds to
        a cylinder normal to the free stream.

    Returns
    -------
    out : float or array_like

    References
    ----------
    * Basic of Aerothermodynamics, Ernst H. Hirschel
    * "Hypersonic Aerothermodynamics", John J. Bertin

    Examples
    --------

    Compute the velocity gradient at the stagnation point of a sphere
    of radius 1 feet, flying at 24000 ft/s at an altitude of 240000 ft.
    Assume the wall temperature is 2500°R.

    >>> import pygasflow
    >>> import pint
    >>> from pygasflow import *
    >>> from pygasflow.atd.avf.wall_shear_stress_sp import velocity_gradient
    >>> ureg = pint.UnitRegistry()
    >>> ureg.formatter.default_format = "~"
    >>> ureg.define("pound_mass = 0.45359237 kg = lbm")
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> R = ureg.Quantity(0.06856070504972671, "Btu / lbm / degR")
    >>> r = 1 * ureg.ft
    >>> u1 = 24e03 * ureg.ft / ureg.s
    >>> Tw = 2500 * ureg.degR
    >>> gamma = 1.4
    
    From the atmospheric model at an altitude of 240000 ft:

    >>> T1 = 381.61885288502907 * ureg.degR
    >>> rho1 = 3.2852596810182865 * ureg.lbm / ureg.ft**3
    >>> p1 = 0.0668887801071935 * ureg.lbf / ureg.ft**2

    Then:

    >>> a1 = sound_speed(gamma, R, T1).to("feet / s")
    >>> M1 = u1 / a1
    >>> res1 = isentropic_solver("m", M1, gamma=gamma, to_dict=True)
    >>> Tt1 = (1 / res1["tr"]) * T1
    >>> pt1 = (1 / res1["pr"]) * p1
    >>> shock = normal_shockwave_solver("mu", M1, gamma=gamma, to_dict=True)
    >>> pt2 = shock["tpr"] * pt1
    >>> Tt2 = Tt1
    >>> rhot2 = (pt2 / (R * Tt2)).to("lbf * s**2 / feet**4")
    >>> u_grad = velocity_gradient(r, p1, pt2, rhot2, k=1)
    >>> u_grad
    <Quantity(12871.5403, '1 / second')>

    """
    # eq (6.164) of "Basic of Aerothermodynamics", Ernst H. Hirschel
    # eq (5.39) of "Hypersonic Aerothermodynamics", John J. Bertin
    return np.cos(phi) * (k / R * np.sqrt(2 * (ps - pinf) / rhos))


def wss_cyl_c(R, phi, u_grad, mu_inf, u_inf, rho_inf, pe_pinf, Ts_Tinf, omega=0.65, laminar=True):
    """
    Compute the wall shear stress along the stagnation line of a infinite
    long swept cylinder for a compressible laminar/turbulent flow.

    Parameters
    ----------
    R : float or array_like
        Radius of the infinite-long swept cylinder.
    phi : float or array_like
        Sweep angle [radians]
    u_grad : float or array_like
        Velocity gradient at the stagnation line.
    mu_inf : float or array_like
        Free stream viscosity.
    u_inf : float or array_like
        Free stream velocity.
    rho_inf : float or array_like
        Free stream density.
    pe_pinf : float or array_like
        Pressure ratio between the pressure at the edge of the boundary layer
        and the free stream pressure.
    Ts_Tinf : float or array_like
        Temperature ratio between the reference temperature and the the
        free stream temperature.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.

    Returns
    -------
    tau_w_scy : float or array_like
        Wall shear stress for a swept cylinder.

    See Also
    --------
    velocity_gradient

    References
    ----------
    Basic of Aerothermodynamics, Ernst H. Hirschel
    
    """
    if laminar:
        C = 0.57
        n = 0.5
    else:
        C = 0.0345
        n = 0.21
    Reinf = rho_inf * u_inf / mu_inf * R
    # eq (7.152)
    f = C * np.sin(phi)**(2 * (1 - n)) * pe_pinf**(1 - n) * Ts_Tinf**(n * (1 + omega) - 1) * (R / u_inf * u_grad)**n / Reinf**n
    # eq (7.151)
    return mu_inf * u_inf / R * f
