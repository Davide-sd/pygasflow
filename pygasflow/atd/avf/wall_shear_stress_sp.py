"""This module contains functions to estimate the **wall shear stress along
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
    """
    # eq (6.164)
    return np.cos(phi) * (k / R * np.sqrt(2 * (ps - pinf) / rhos))


def wss_cyl_c(R, phi, u_grad, muinf, uinf, rhoinf, pe_pinf, Ts_Tinf, omega=0.65, laminar=True):
    """Compute the wall shear stress along the stagnation line of a infinite
    long swept cylinder for a compressible laminar/turbulent flow.

    Parameters
    ----------
    R : float or array_like
        Radius of the infinite-long swept cylinder.
    phi : float or array_like
        Sweep angle [radians]
    u_grad : float or array_like
        Velocity gradient at the stagnation line.
    muinf : float or array_like
        Free stream viscosity.
    uinf : float or array_like
        Free stream velocity.
    rhoinf : float or array_like
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
    out : float or array_like

    See Also
    --------
    velocity_gradient
    """
    if laminar:
        C = 0.57
        n = 0.5
    else:
        C = 0.0345
        n = 0.21
    Reinf = rhoinf * uinf / muinf * R
    # eq (7.152)
    f = C * np.sin(phi)**(2 * (1 - n)) * pe_pinf**(1 - n) * Ts_Tinf**(n * (1 + omega) - 1) * (R / uinf * u_grad)**n / Reinf**n
    # eq (7.151)
    return muinf * uinf / R * f
