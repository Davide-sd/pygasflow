"""This module contains a lot of functions to estimate the **boundary-layer
thicknesses of a incompressible/compressible, laminar/turbulent flat plate**.

The four most important functions are: ``deltas_lam_ic``, ``deltas_tur_ic``,
``deltas_lam_c``, ``deltas_tur_c`` as they are wrapper functions that allows
to estimates all thicknesses with a single call.
"""

import numpy as np
from pygasflow.utils.common import _should_solver_return_dict


def delta_lam_ic(x, Re, c=5):
    """Boundary-layer thickness of the incompressible laminar flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    c : float, optional
        Proportionality constant from Blasius theory. Default to 5.

    Returns
    -------
    out : float or array_like

    Notes
    -----
    delta_lam_ic is the distance at which locally the tangential velocity
    component u(y) has approached the inviscid external velocity u_e by
    `eps * u_e`:

    u_e - u(y) <= eps * u_e

    Usually, the boundary-layer thickness is defined with `eps=0.01`, which
    gives c=5. If `eps=0.001` is taken, then c=6.

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.88)
    return c * x / Re**0.5


def delta_tur_ic(x, Re):
    """Boundary-layer thickness of the incompressible turbulent flat plate.
    This is valid for a low Reynolds number. It comes from the (1/7) power
    velocity distribution law.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.94)
    return 0.37 * x / Re**0.2


def delta_tur_ic_viscous(x, Re):
    """Viscous sub-layer thickness of the incompressible turbulent flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.97)
    return 29.06 * x / Re**0.9


def viscous_edge_velocity(ue, Re):
    """Viscous sub-layer edge velocity. It is the velocity corresponding to
    ``delta_tur_ic_viscous``.

     Parameters
    ----------
    ue : float or array_like
        Edge velocity at the boundary layer.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.98)
    return 2.12 * ue / Re**0.1


def delta_tur_ic_scaling(x, Re):
    """Scaling thickness for the incompressible turbulent flat plate, where
    the non-dimensional velocity u+ and the wall distance y+ are equal.
    It is somewhat similar to the viscous sub-layer thickness.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.99)
    return 33.78 * x / Re**0.8


def delta_lam_c(x, Re, Ts_Tinf, omega=0.65, c=5):
    """Boundary-layer thickness of the compressible laminar flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    c : float, optional
        Proportionality constant from Blasius theory. Default to 5.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------

    delta_lam_ic

    """
    d_l_ic = delta_lam_ic(x, Re, c=c)
    # eq (7.100)
    return d_l_ic * Ts_Tinf**(0.5 * (1 + omega))


def delta_tur_c(x, Re, Ts_Tinf, omega=0.65):
    """Boundary-layer thickness of the compressible turbulent flat plate.
    This is valid for a low Reynolds number. It comes from the (1/7) power
    velocity distribution law.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like

    References
    ----------
    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_tur_ic
    """
    d_t_ic = delta_tur_ic(x, Re)
    # eq (7.104)
    return d_t_ic * Ts_Tinf**(0.2 * (1 + omega))


def delta_tur_c_viscous(x, Re, Ts_Tinf, omega=0.65):
    """Viscous sub-layer thickness of the compressible turbulent flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_tur_ic_viscous
    """
    # eq (7.105)
    return delta_tur_ic_viscous(x, Re) * Ts_Tinf**(0.9 * (1 + omega))


def delta_tur_c_scaling(x, Re, Ts_Tinf, omega=0.65):
    """Scaling thickness for the compressible turbulent flat plate, where
    the non-dimensional velocity u+ and the wall distance y+ are equal.
    It is somewhat similar to the viscous sub-layer thickness .

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich
    """
    # eq (7.106)
    return delta_tur_ic_scaling(x, Re) * Ts_Tinf**(0.8 * (1 + omega))


def delta_1_lam_ic(x, Re):
    """Compute the integral parameter delta_1, also known as the boundary-layer
    displacement thickness, for the laminar incompressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_2_lam_ic
    """
    # eq (7.114)
    return 1.7208 * x / Re**0.5


def delta_2_lam_ic(x, Re):
    """Compute the integral parameter delta_2, also known as the momentum
    thickness, for the laminar incompressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_1_lam_ic
    """
    # eq (7.115)
    return 0.6641 * x / Re**0.5


def delta_1_tur_ic(x, Re):
    """Compute the integral parameter delta_1, also known as the boundary-layer
    displacement thickness, for the turbulent incompressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_2_tur_ic
    """
    # eq (7.116)
    return 0.0463 * x / Re**0.2


def delta_2_tur_ic(x, Re):
    """Compute the integral parameter delta_1, also known as the momentum
    thickness, for the turbulent incompressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.

    Returns
    -------
    out : float or array_like

    References
    ----------

    "Basic of aerothermodynamics" by Ernst Heinrich

    See Also
    --------
    delta_1_tur_ic
    """
    # eq (7.117)
    return 0.0360 * x / Re**0.2


def shape_factor_lam_ic():
    """Compute the shape factor, H12, for the laminar incrompressible case.

    Returns
    -------
    out : float
    """
    # eq (7.118)
    return 1.7208 / 0.6641


def shape_factor_tur_ic():
    """Compute the shape factor, H12, for the turbulent incrompressible case.

    Returns
    -------
    out : float
    """
    # eq (7.119)
    return 0.0463 / 0.0360


def delta_1_lam_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega=0.65, gammainf=1.4):
    """Compute the integral parameter delta_1, also known as the boundary-layer
    displacement thickness, for the laminar compressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Tw_Tinf : float or array_like
        Temperature Ratio Tw / Tinf between the wall temperature and the
        free stream temperature Tinf.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    Minf : float or array_like
        Free stream Mach number.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    gammainf : float, optional
        Free stream specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.120)
    return delta_1_lam_ic(x, Re) * (-0.122 + 1.22 * Tw_Tinf + 0.333 * (gammainf - 1) / 2 * Minf**2) * Ts_Tinf**(0.5 * (omega - 1))


def delta_2_lam_c(x, Re, Ts_Tinf, omega=0.65):
    """Compute the integral parameter delta_2, also known as the momentum
    thickness, for the laminar compressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.121)
    return delta_2_lam_ic(x, Re) * Ts_Tinf**(0.5 * (omega - 1))


def delta_1_tur_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega=0.65, gammainf=1.4):
    """Compute the integral parameter delta_1, also known as the boundary-layer
    displacement thickness, for the turbulent compressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Tw_Tinf : float or array_like
        Temperature Ratio Tw / Tinf between the wall temperature and the
        free stream temperature Tinf.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    Minf : float or array_like
        Free stream Mach number.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    gammainf : float, optional
        Free stream specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.122), (7.123)
    d1tic = 0.0504 * x / Re**0.2
    return d1tic * (0.129 + 0.871 * Tw_Tinf + 0.648 * ((gammainf - 1) / 2) * Minf**2) * Ts_Tinf**(0.2 * (omega - 4))


def delta_2_tur_c(x, Re, Ts_Tinf, omega=0.65):
    """Compute the integral parameter delta_2, also known as the momentum
    thickness, for the turbulent compressible case.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Tw_Tinf : float or array_like
        Temperature Ratio Tw / Tinf between the wall temperature and the
        free stream temperature Tinf.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    Minf : float or array_like
        Free stream Mach number.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    gammainf : float, optional
        Free stream specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.124)
    return delta_2_tur_ic(x, Re) * Ts_Tinf**(0.2 * (omega - 4))


def shape_factor_lam_c(x, Re, Tw_Tinf, Minf, gammainf=1.4):
    """Compute the shape factor, H12, for the laminar crompressible case.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.125)
    return shape_factor_lam_ic() * (-0.122 + 1.22 * Tw_Tinf + 0.333 * ((gammainf - 1) / 2) * Minf**2)


def shape_factor_tur_c(Tw_Tinf, gammainf, Minf):
    """Compute the shape factor, H12, for the turbulent incrompressible case.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.126), (7.127)
    return 1.4 * (0.129 + 0.871 * Tw_Tinf + 0.648 * ((gammainf - 1) / 2) * Minf**2)


def deltas_lam_ic(x, Re, to_dict=False):
    """Compute different boundary-layer thicknesses for the incompressible
    laminar flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    delta : float or array_like
        Boundary-layer thickness computed with the ``delta_lam_ic``.
    delta_1 : float or array_like
        Boundary-layer displacement thickness computed with ``delta_1_lam_ic``.
    delta_2 : float or array_like
        Momentum thickness computed with the ``delta_2_lam_ic``.
    H12 : float or array_like
        Shape factor computed with the ``shape_factor_lam_ic``.

    See Also
    --------
    delta_lam_ic, delta_1_lam_ic, delta_2_lam_ic, shape_factor_lam_ic,
    deltas_lam_c, deltas_tur_ic, deltas_tur_c
    """
    to_dict = _should_solver_return_dict(to_dict)
    results = [
        delta_lam_ic(x, Re),
        delta_1_lam_ic(x, Re),
        delta_2_lam_ic(x, Re),
        shape_factor_lam_ic()
    ]
    if to_dict is False:
        return results
    keys = ["delta", "delta_1", "delta_2", "H12"]
    return {k: v for k, v in zip(keys, results)}


def deltas_lam_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega=0.65, gammainf=1.4, to_dict=False):
    """Compute different boundary-layer thicknesses for the compressible
    laminar flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Tw_Tinf : float or array_like
        Temperature Ratio Tw / Tinf between the wall temperature and the
        free stream temperature Tinf.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    Minf : float or array_like
        Free stream Mach number.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    gammainf : float, optional
        Free stream specific heats ratio. Default to 1.4. Must be > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    delta : float or array_like
        Boundary-layer thickness computed with ``delta_lam_c``.
    delta_1 : float or array_like
        Boundary-layer displacement thickness computed with ``delta_1_lam_c``.
    delta_2 : float or array_like
        Momentum thickness computed with ``delta_2_lam_c``.
    H12 : float or array_like
        Shape factor computed with ``shape_factor_lam_c``.

    See Also
    --------
    delta_lam_c, delta_1_lam_c, delta_2_lam_c, shape_factor_lam_c,
    deltas_lam_ic, deltas_tur_ic, deltas_tur_c
    """
    to_dict = _should_solver_return_dict(to_dict)
    results = [
        delta_lam_c(x, Re, Ts_Tinf, omega),
        delta_1_lam_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega, gammainf),
        delta_2_lam_c(x, Re, Ts_Tinf, omega),
        shape_factor_lam_c(x, Re, Tw_Tinf, Minf, gammainf)
    ]
    if to_dict is False:
        return results
    keys = ["delta", "delta_1", "delta_2", "H12"]
    return {k: v for k, v in zip(keys, results)}


def deltas_tur_ic(x, Re, to_dict=False):
    """Compute different boundary-layer thicknesses for the incompressible
    turbulent flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    delta : float or array_like
        Boundary-layer thickness computed with ``delta_tur_ic``.
    delta_vs : float or array_like, optional
        Viscous sub-layer thickness computed with ``delta_tur_ic_viscous``.
    delta_sc : float or array_like, optional
        Scaling thickness computed with ``delta_tur_ic_scaling``.
    delta_1 : float or array_like
        Boundary-layer displacement thickness computed with ``delta_1_tur_ic``.
    delta_2 : float or array_like
        Momentum thickness computed with ``delta_2_tur_ic``.
    H12 : float or array_like
        Shape factor computed with ``shape_factor_tur_ic``.

    See Also
    --------
    delta_tur_ic, delta_1_tur_ic, delta_2_tur_ic, shape_factor_tur_ic,
    deltas_tur_c, deltas_lam_ic, deltas_lam_c
    """
    to_dict = _should_solver_return_dict(to_dict)
    results = [
        delta_tur_ic(x, Re),
        delta_tur_ic_viscous(x, Re),
        delta_tur_ic_scaling(x, Re),
        delta_1_tur_ic(x, Re),
        delta_2_tur_ic(x, Re),
        shape_factor_tur_ic()
    ]
    if to_dict is False:
        return results
    keys = ["delta", "delta_vs", "delta_sc", "delta_1", "delta_2", "H12"]
    return {k: v for k, v in zip(keys, results)}


def deltas_tur_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega=0.65, gammainf=1.4, to_dict=False):
    """Compute different boundary-layer thicknesses for the compressible
    turbulent flat plate.

    Parameters
    ----------
    x : float or array_like
        Location where to compute the thickness.
    Re : float or array_like
        Free-stream Reynolds number computed at `x`.
    Tw_Tinf : float or array_like
        Temperature Ratio Tw / Tinf between the wall temperature and the
        free stream temperature Tinf.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    Minf : float or array_like
        Free stream Mach number.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    gammainf : float, optional
        Free stream specific heats ratio. Default to 1.4. Must be > 1.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    delta : float or array_like
        Boundary-layer thickness computed with ``delta_tur_c``.
    delta_vs : float or array_like, optional
        Viscous sub-layer thickness computed with ``delta_tur_c_viscous``.
    delta_sc : float or array_like, optional
        Scaling thickness computed with ``delta_tur_c_scaling``.
    delta_1 : float or array_like
        Boundary-layer displacement thickness computed with ``delta_1_tur_ic``.
    delta_2 : float or array_like
        Momentum thickness computed with ``delta_2_tur_c``.
    H12 : float or array_like
        Shape factor computed with ``shape_factor_tur_c``.

    See Also
    --------
    delta_tur_c, delta_1_tur_c,  delta_2_tur_c, shape_factor_tur_c,
    deltas_tur_ic, deltas_lam_ic, deltas_lam_c
    """
    to_dict = _should_solver_return_dict(to_dict)
    results = [
        delta_tur_c(x, Re, Ts_Tinf, omega),
        delta_tur_c_viscous(x, Re, Ts_Tinf, omega),
        delta_tur_c_scaling(x, Re, Ts_Tinf, omega),
        delta_1_tur_c(x, Re, Tw_Tinf, Ts_Tinf, Minf, omega, gammainf),
        delta_2_tur_c(x, Re, Ts_Tinf, omega),
        shape_factor_tur_c(Tw_Tinf, gammainf, Minf)
    ]
    if to_dict is False:
        return results
    keys = ["delta", "delta_vs", "delta_sc", "delta_1", "delta_2", "H12"]
    return {k: v for k, v in zip(keys, results)}
