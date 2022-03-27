import numpy as np


def Prandtl(*args, **kwargs):
    """Compute the Prandtl number.

    There are 5 modes of operation:

    1. ``Prandtl(mu, cp, k)``
    2. ``Prandtl(gas)`` where ``gas`` is a Cantera's ``Solution`` object.
    3. ``Prandtl(gamma)`` which is a good approximation for both monoatomic and
       polyatomic gases. It is derived from Eucken's formula for thermal
       conductivity.
    4. ``Prandtl(Pe=Pe, Re=Re)`` by providing Peclét and Reynolds numbers.
    5. ``Prandtl(Le=Le, Sc=Sc)`` by providing Lewis and Schmidt numbers.

    Parameters
    ----------
    mu : float or array_like
        Viscosity of the gas.
    cp : float or array_like
        Specific heat at constant pressure.
    k : float or array_like
        Thermal conductivity of the gas.
    Pe : float or array_like
        Peclét number.
    Re : float or array_like
        Reynolds number.
    Le : float or array_like
        Lewis number.
    Sc : float or array_like
        Schmidt number.
    gamma : float
        Specific heats ratio. Default to None. Must be gamma > 1.
    gas : ct.Solution
        A Cantera's ``Solution`` object.

    Returns
    -------
    Pr : float or array_like
        Prandtl number.

    Notes
    -----

    * Pr -> 0: the thermal boundary layer is much thicker than the flow
      boundary layer, which is typical for the flow of liquid metals.
    * Pr -> oo: the flow boundary layer is much thicker than the thermal
      boundary layer, which is typical for liquids.
    * Pr = O(1): the thermal boundary layer has a thickness of the order of
      that of the flow boundary layer. This is typical for gases.

    Examples
    --------

    Compute the Prandtl number of air with specific heat ratio of 1.4:

    >>> from pygasflow.atd.nd_numbers import Prandtl
    >>> Prandtl(1.4)
    0.7368421052631579

    Compute the Prandtl number of air at T=350K using a Cantera's ``Solution``
    object:

    >>> import cantera as ct
    >>> air = ct.Solution("gri30.yaml")
    >>> air.TPX = 350, ct.one_atm, {"N2": 0.79, "O2": 0.21}
    >>> Prandtl(air)
    0.7139365242266411

    Compute the Prandtl number by providing mu, cp, k:

    >>> from pygasflow.atd.viscosity import viscosity_air_southerland
    >>> from pygasflow.atd.thermal_conductivity import thermal_conductivity_hansen
    >>> cp = 1004
    >>> mu = viscosity_air_southerland(350)
    >>> k = thermal_conductivity_hansen(350)
    >>> Prandtl(mu, cp, k)
    0.7370392202421769

    See Also
    --------
    Peclet, Lewis, Reynolds, Schmidt
    """
    if len(args) == 1:
        if isinstance(args[0], (int, float, np.ndarray)):
            # eq (4.19)
            gamma = args[0]
            return 4 * gamma / (9 * gamma - 5)
        else:
            import cantera as ct
            if isinstance(args[0], ct.Solution):
                gas = args[0]
                return gas.viscosity * gas.cp_mass / gas.thermal_conductivity
        raise ValueError(
            "When 1 argument is provided, it must be an instance of int, or "
            "float, or np.ndarray or ct.Solution"
        )
    elif len(args) == 3:
        # eq (4.19)
        mu, cp, k = args
        return mu * cp / k

    Pe = kwargs.get("Pe", None)
    Re = kwargs.get("Re", None)
    if (Pe is not None) and (Re is not None):
        # eq (4.67)
        return Pe / Re

    Le = kwargs.get("Le", None)
    Sc = kwargs.get("Sc", None)
    if (Le is not None) and (Sc is not None):
        # eq (4.93)
        return Le * Sc

    raise ValueError("Combination of parameters not recognized. "
        "Please, read the documentation.")


def Knudsen(*args):
    """Compute the Knudsen number.

    There are 2 modes of operation:

    * ``Knudsen(lambda, L)``
    * ``Knudsen(Minf, Reinf_L, gamma)``

    Parameters
    ----------
    lambda : float or array_like
        Mean free path in the gas.
    L : float or array_like
        Characteristic length, which must be chosen according to the flow
        under consideration.For example, for boundary-layer flow it would be
        based on the boundary-layer thickness.
    Minf : float or array_like
        Free stream Mach number.
    Reinf_L : float or array_like
        Free stream Reynolds number computed at a characteristic length.
    gamma : float
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Kn : float or array_like

    Notes
    -----
    Knudsen number is employed to distinguish approximately between flow
    regimes:

    * Kn <= 0.01: continuum flow
    * 0.01 <= Kn <= 0.1: continuum flow with slip effects (slip flow and
      temperature jumps at a body surface).
    * 0.1 <= Kn <= 10: disturbed free molecular flow (gas particles collide
      with the body surface and with each other).
    * Kn >= 10: free molecular flow (gas particles collide only with the body
      surface).

    See Also
    --------
    Reynolds
    """
    if len(args) == 2:
        # eq (2.13)
        _lambda, _L = args
        return _lambda / _L
    elif len(args) == 3:
        # eq (2.16)
        Minf, Reinf_L, gamma = args
        return np.sqrt(gamma / (2 * np.pi)) * 16 / 5 * Minf / Reinf_L
    raise ValueError("Combination of parameters not recognized. "
        "Please, read the documentation.")


def Stanton(*args):
    """Compute the Stanton number, which represents a dimensionless form of
    the heat flux q_gw.

    There are 3 modes of operation:

    * Stanton(q_gw, q_inf)
    * Stanton(q_gw, rho_inf, v_inf)
    * Stanton(q_gw, rho_inf, v_inf, delta_h)

    Parameters
    ----------
    q_gw : float or array_like
        Heat flux in the gas at the wall. It is the heat transported towards
        the surface of the vehicle by diffusion mechanisms.
    q_inf : float or array_like
        Heat transported towards a flight vehicle:
        q_inf = rho_inf * v_inf * h_t
    rho_inf : float or array_like
        Free stream density.
    v_inf : float or array_like
        Free stream velocity.
    delta_h : float or array_like
        Difference between the enthalpy related to the recovery temperature and
        the enthalpy related to the wall temperature, hr - hw.

    Returns
    -------
    Sn : float or array_like
    """
    if len(args) == 2:
        # eq (3.3)
        q_gw, q_inf = args
        return q_gw / q_inf
    elif len(args) == 3:
        # eq (3.4)
        q_gw, rho_inf, v_inf = args
        return 2 * q_gw / (rho_inf * v_inf**3)
    elif len(args) == 4:
        # eq (3.5)
        q_gw, rho_inf, v_inf, delta_h = args
        return q_gw / (rho_inf * v_inf * delta_h)
    raise ValueError("Combination of parameters not recognized. "
        "Please, read the documentation.")


def Strouhal(*args):
    """Compute the Strouhal number.

    There are 2 modes of operation:

    * ``Strouhal(t_res, t_ref)``
    * ``Strouhal(L_ref, t_ref, v_ref)``

    Parameters
    ----------
    L_ref : float or array_like
        Reference length (for example, the body vehicle length).
    t_ref : float or array_like
        Reference time.
    v_ref : float or array_like
        Reference velocity (for example, free stream velocity).
    t_res : float or array_like
        Residence time, defined as t_res = L_ref / v_ref.

    Returns
    -------
    Sr : float or array_like

    Notes
    -----
    In our applications we speak about steady, quasi-steady, and unsteady
    flow problems. The measure for the distinction of these three flow modes is
    the Strouhal number, Sr:

    * Sr = 0: steady flow.
    * Sr -> 0: quasi-steady flow. The residence time is small compared to the
      reference time, in which a change of flow parameters happens.
      For practical purposes, quasi-steady flow is permitted for Sr <= 0.2.
    * Sr = O(1): unsteady flow.

    Note: the movement of a flight vehicle may be permitted to be considered
    as at least quasi-steady, while at the same time truly unsteady movements
    of a control surface may occur. In addition there might be configuration
    details, where highly unsteady vortex shedding is present.
    """
    if len(args) == 2:
        # eq (4.8)
        t_res, t_ref = args
        return t_res / t_ref
    elif len(args) == 3:
        # eq (4.6)
        L_ref, t_ref, v_ref = args
        return L_ref / (t_ref * v_ref)
    raise ValueError("Combination of parameters not recognized. "
        "Please, read the documentation.")


def Reynolds(rho, u, mu, L=1):
    """Compute the Reynolds number, which is the ratio between the inertial
    forces to the viscous forces.

    Parameters
    ----------
    rho : float or array_like
        Density.
    u : float or array_like
        Velocity.
    mu : float or array_like
        Viscosity.
    L : float or array_like, optional
        Characteristic length. Default to ``L=1``, which computes the unitary
        Reynolds number.

    Returns
    -------
    Re : float or array_like

    Notes
    -----
    Reynolds number is the principle similarity parameter governing viscous
    phenomena:

    * Re -> 0: the molecular transport of momentum is much larger than the
      convective transport, the flow is the "creeping" or Stokes flow.
      The convective transport can be neglected.
    * Re -> oo: the convective transport of momentum is much larger than the
      molecular transport, the flow can be considered as inviscid, i. e.
      molecular transport can be neglected.
    * Re = O(1): the molecular transport of momentum has the same order
      of magnitude as the convective transport, the flow is viscous, i. e.
      it is boundary-layer, or in general, shear-layer flow.
    """
    # eq (4.40)
    return rho * u * L / mu


def Peclet(rho, mu, cp, L, k):
    """Compute the Peclét number.

    Parameters
    ----------
    rho : float or array_like
        Density.
    mu : float or array_like
        Viscosity.
    cp : float or array_like
        Specific heat at constant pressure.
    L : float or array_like
        Characteristic length.
    k : float or array_like
        Thermal conductivity.

    Returns
    -------
    Pe : float or array_like

    Notes
    -----
    Peclét number relates the molecular transport of heat to the convective
    transport. In particular:

    * Pe -> 0: the molecular transport of heat is much larger than the
      convective transport.
    * Pe -> oo: the convective transport of heat is much larger than the
      molecular transport.
    * Pe = O(1): the molecular transport of heat has the same order of
      magnitude as the convective transport.
    """
    # eq (4.66)
    return rho * mu * cp * L / k


def Lewis(rho, D, cp, k):
    """Compute the Lewis number, which is interpreted as the ratio of 'heat
    transport by mass diffusion' to 'heat transport by conduction' in a flow
    with chemical non-equilibrium.

    Parameters
    ----------
    rho : float or array_like
        Density.
    D : float or array_like
        Mass diffusivity.
    cp : float or array_like
        Specific heat at constant pressure.
    k : float or array_like
        Thermal conductivity

    Returns
    -------
    Lw : float or array_like
    """
    # eq (4.71)
    return rho * D * cp / k


def Eckert(M, gamma=1.4):
    """Compute the Eckert number, which can be interpreted as ratio of
    kinetic energy to thermal energy of the flow.

    Parameters
    ----------
    M : float or array_like
        Mach number.
    gamma : float
        Specific heats ratio. Default to None. Must be gamma > 1.

    Returns
    -------
    E : float or array_like
    """
    # eq (4.73)
    return (gamma - 1) * M**2


def Schmidt(*args):
    """Compute the Schmidt number.

    There are 2 modes of operation:

    * ``Schmidt(Pr, Le)``
    * ``Schmidt(rho, mu, D)``

    Parameters
    ----------
    Pr : float or array_like
        Prandtl number.
    Le : float or array_like
        Lewis number.
    rho : float or array_like
        Density.
    mu : float or array_like
        Viscosity.
    D : float or array_like
        Mass diffusivity.

    Returns
    -------
    Sc : float or array_like

    Notes
    -----

    * Sc -> 0: the molecular transport of mass is much larger than the
      convective transport.
    * Sc -> oo: the convective transport of mass is much larger than the
      molecular transport.
    * Sc = O(1): the molecular transport of mass has the same order of
      magnitude as the convective transport.

    See Also
    --------
    Lewis, Prandtl
    """
    # eq (4.92)
    rho, mu, D = args
    return mu / (rho * D)
    # eq (4.93)
    Pr, Le = args
    return Pr / Le
