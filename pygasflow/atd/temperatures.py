
import numpy as np


def recovery_factor(Pr, laminar=True):
    """Compute the recovery factor.

    Parameters
    ----------
    Pr : float or array_like
        Prandtl number
    laminar : bool, optional
        If True, compute the recovery factor for a laminar flow. Otherwise,
        compute it for a turbulent flow.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    reference_temperature
    """
    # near eq (3.7)
    if laminar:
        return np.sqrt(Pr)
    return np.cbrt(Pr)


def recovery_temperature(T, M, r, gamma=1.4):
    """Compute the recovery temperature.

    Parameters
    ----------
    T : float or array_like
        Free stream (or boundary-layer edge) static temperature.
    M : float or array_like
        Free stream (or boundary-layer edge) Mach number
    r : float or array_like
        Recovery factor
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    out : float or array_like

    Notes
    -----

    Quotes from Section 3.1:

    Consider a wall with finite thickness and finite heat capacity, which is
    completely insulated from the surroundings, except at the surface, where it
    is exposed to the (viscous) flow. Without radiation cooling, the wall
    material will be heated up by the flow, depending on the heat amount
    penetrating the surface and the heat capacity of the material. The surface
    temperature will always be that of the gas at the wall: Tw = Tgw, apart
    from a possible temperature jump, which can be present in the slip-flow
    regime. If enough heat has entered the wall material (function of time),
    the temperature in the entire wall and at the surface will reach an
    upper limit, the recovery temperature Tw = Tr (the heat flux goes to zero).
    The surface is then called an adiabatic surface: no exchange of heat takes
    place between gas and wall material. With steady flow conditions the
    recovery (adiabatic) temperature Tr is somewhat smaller than the total
    temperature Tt, but always of the same order of magnitude.

    The total enthalpy at hypersonic flight is proportional to the flight
    velocity squared. This holds also for the total temperature, if perfect gas
    behaviour can be assumed, which is permitted for v_inf < 1.0 km/s.
    The total temperature Tt then is only a function of the total enthalpy ht,
    which can be expressed as function of the flight Mach number Minf.

    At flight velocities larger than approximately 1.0 km/s they lose their
    validity, since high-temperature real-gas effects appear. The temperature
    in thermal and chemical equilibrium becomes a function of two variables,
    for instance the enthalpy and the density. At velocities larger than
    approximately 5.0 km/s, non-equilibrium effects can play a role,
    complicating even more these relations.

    References
    ----------
    Basic of Aerothermodynamics, Ernst H. Hirschel

    See Also
    --------
    recovery_factor
    """
    # eq (3.7)
    return T * (1 + r * (gamma - 1) / 2 * M**2)


def reference_temperature(Te, Tw, Me=None, rs=None, Tr=None, gamma_e=1.4, mod=False):
    """Compute the reference temperature for compressible boundary layers.

    There are three modes of operations:

    * ``reference_temperature(Te, Tw, rs=rs, Me=Me, gamma_e=gamma_e)``
    * ``reference_temperature(Te, Tw, Prs=Prs, Me=Me, gamma_e=gamma_e)``
    * ``reference_temperature(Te, Tw, Tr=Tr)``

    Parameters
    ----------
    Te : float or array_like
        Temperature at the edge of the boundary layer.
    Tw : float or array_like
        Temperature of the gas at the wall.
    Me : None or float or array_like, optional
        Mach number at the edge of the boundary layer. If ``Me=None``, then
        ``Tr`` must be provided.
    rs : None or float or array_like, optional
        Reference recovery factor. If ``rs=None``, then ``Tr`` must be
        provided.
    Tr : None or float or array_like, optional
        Recovery temperature. If ``Tr=None``, then ``Me`` and ``rs``/``Prs``
        must be provided.
    gamma_e : float, optional
        Specific heats ratio at the edge of the boundary layer.
        Default to 1.4. Must be > 1.
    mod : bool, optional
        Use a modified formula that gives more weight to the recovery
        temperature and less on the wall temperature. Useful for computations
        at stagnation points/lines. Default to False.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    recovery_factor, recovery_temperature
    """
    if (Me is None) and (Tr is None) and (rs is None):
        raise ValueError("This function requires either `Me` or `Tr`.")
    if Tr is not None:
        if mod is False:
            # eq (7.62)
            return 0.28 * Te + 0.5 * Tw + 0.22 * Tr
        # eq (7.150)
        return 0.3 * Te + 0.1 * Tw + 0.6 * Tr

    if mod is False:
        # eq (7.70)
        return 0.5 * Te + 0.5 * Tw + 0.22 * rs * (gamma_e - 1) / 2 * Me**2 * Te
    # eq (7.150) modified
    return 0.9 * Te + 0.1 * Tw + 0.6 * rs * (gamma_e - 1) / 2 * Me**2 * Te
