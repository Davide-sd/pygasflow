import numpy as np

from pygasflow.atd.avf.thickness_fp import (
    deltas_lam_c, deltas_lam_ic, deltas_tur_c, deltas_tur_ic
)
from pygasflow.atd.avf.wall_shear_stress_fp import wss_ic, wss_c, skin_friction

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
    if laminar:
        return np.sqrt(Pr)
    return np.cbrt(Pr)


def recovery_temperature(T, M, r, gamma=1.4):
    """Compute the recovery temperature.

    Parameters
    ----------
    T : float or array_like
        Static temperature
    M : float or array_like
        Mach number
    r : float or array_like
        Recovery factor
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    recovery_factor
    """
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
