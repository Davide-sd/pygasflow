"""This module exposes functionalities to estimate the hypersonic viscous
interaction over a flat plate, in which for large Mach numbers and small
Reynolds numbers the attached viscous flow is no more of boundary-layer type.
Instead, observation shows that past a flat plate, a very large surface
pressure is present just downstream of the leading edge, which is in contrast
to high-Reynolds number boundary-layer flows, where the surface pressure is
the free-stream pressure.
"""

import numpy as np
from pygasflow.atd.viscosity import viscosity_air_power_law


def interaction_parameter(Minf, Re_inf, Cinf=1, laminar=True):
    """Compute the viscous interaction parameter, Chi, which correlates
    pressure changes.

    Parameters
    ----------
    Minf : float or array_like
        Free stream Mach number.
    Re_inf : float or array_like
        Free stream Reynolds number computed at some location.
        If a unitary Reynolds number is provided, then the unitary interaction
        parameter will be returned.
    Cinf : float or array_like
        Chapman-Rubesin linear viscosity-law constant.
    laminar : bool
        Default to True. Set ``laminar=False`` to compute the viscous
        interaction parameter for turbulent flow.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    rarefaction_parameter, chapman_rubesin
    """
    if laminar:
        # eq (9.23), (9.24)
        return Minf**3 * np.sqrt(Cinf) / np.sqrt(Re_inf)
    # eq (9.28)
    return (Minf**9 * Cinf / Re_inf)**0.2


def rarefaction_parameter(Minf, Re_inf, Cinf):
    """Compute the rarefaction parameter, V, which correlates
    viscous/inviscid-induced perturbations in the skin friction and heat
    transfer.

    Parameters
    ----------
    Minf : float or array_like
        Free stream Mach number.
    Re_inf : float or array_like
        Free stream Reynolds number computed at some location.
        If a unitary Reynolds number is provided, then the unitary interaction
        parameter will be returned.
    Cinf : float or array_like
        Chapman-Rubesin linear viscosity-law constant.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    interaction_parameter, chapman_rubesin
    """
    # eq (9.38)
    return Minf * np.sqrt(Cinf) / np.sqrt(Re_inf)


def chapman_rubesin(Tw, Tinf, func=None):
    """Compute the Chapman-Rubesin factor (linear viscosity-law constant).

    Parameters
    ----------
    Tw : float or array_like
        Temperature at the wall.
    Tinf : float or array_like
        Free stream temperature.
    func : callable
        Function to compute the viscosity. If None (default value), a power
        law viscosity will be used.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    interaction_parameter, rarefaction_parameter
    """
    if func is None:
        func = viscosity_air_power_law
    mu_w = func(Tw)
    mu_inf = func(Tinf)
    # eq (9.17)
    return mu_w * Tinf / (mu_inf * Tw)


def critical_distance(Chi_u, weak=True, cold_wall=True, Chi_crit=None):
    """Compute the critical distance, x_crit.

    In a flat plate:

    * For x < x_crit there is strong interaction.
    * For x > x_crit there is weak interaction.

    Parameters
    ----------
    Chi_u : float or array_like
        Unitary viscous interaction parameter.
    weak : bool, optional
        Selector to decide the value of Chi_crit.
    cold_wall : bool, optional
        Selector to decide the value of Chi_crit.
    Chi_crit : None or float, optional
        Specify a numerical value to Chi_crit. This overrides ``weak`` and
        ``cold_wall``.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    interaction_parameter
    """
    if Chi_crit is None:
        if weak and cold_wall:
            Chi_crit = 11
        elif weak:
            Chi_crit = 3
        elif (weak is False) and cold_wall:
            Chi_crit = 13
        else:
            Chi_crit = 4
    # eq (9.41)
    return (Chi_u / Chi_crit)**2


def wall_pressure_ratio(Chi, Tw_Tt, weak=True, laminar=True, gamma=1.4):
    """Compute the pressure ratio Pw / Pinf between the pressure at the wall
    and the free stream pressure.

    Parameters
    ----------
    Chi : float or array_like
        Viscous interaction parameter.
    Tw_Tt : float or array_like
        Temperature ratio between the temperature at the wall and the free
        stream temperature.

        * Set ``Tw_Tt=0`` to specify a cold wall.
        * Set ``Tw_Tt=1`` to specify a hot wall.

    weak : bool
        Default to True indicating a weak interaction. Set ``weak=False``
        to specify for a strong interaction.
    laminar : bool
        Default to True. Set ``laminar=False`` to compute the pressure
        ratio for a turbulent flow.
    gamma : float
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    out : float or array_like
    """
    if weak and laminar:
        # eq (9.25)
        return 1 + 0.578 * gamma * (gamma - 1) / 4 * (1 + 3.35 * Tw_Tt) * Chi
    elif weak:
        # eq (9.29)
        return 1 + 0.057 * ((1 + 1.3 * Tw_Tt) / (1 + 2.5 * Tw_Tt)**0.6) * Chi
    # eq (9.35)
    return np.sqrt(3) / 4 * np.sqrt((gamma + 1) / (2 * gamma)) * gamma * (gamma - 1) * (0.664 + 1.73 * Tw_Tt) * Chi


def length_shock_formation_region(Vu, coeff=0.1):
    """Compute the length of the shock formation region.

    Parameters
    ----------
    Vu : float or array_like
        Unitary rarefaction parameter.
    coeff : float or array_like
        A coefficient indicating the departure from the strong interaction
        region. It goes 0.1 for the surface pressure, to 0.3 for the heat
        transfer. Default to 0.1.

    Returns
    -------
    out : float or array_like

    References
    ----------
    "Criterion for slip near the leading edge of a flat plate in hypersonic
    flow", L. Talbot, 1963

    See Also
    --------
    rarefaction_parameter
    """
    # eq (9.47)
    return Vu**2 / coeff**2
