import numpy as np
from pygasflow.utils.decorators import check
from pygasflow.shockwave import rayleigh_pitot_formula

def sound_speed(*args, **kwargs):
    """Compute the sound speed.

    There are two modes of operations:

    * provide ``gamma``, ``R``, ``T``:
      ``sound_speed(gamma, R, T)``
    * provide a Cantera's ``Solution`` object, from which the parameters will
      be retrieved:
      ``sound_speed(gas)``

    Parameters
    ----------
    args :
        gamma : float or array_like
            Specific heats ratio. Default to 1.4. Must be gamma > 1.
        R : float or array_like
            Specific gas constant [J / (kg * K)]
        T : float or array_like
            Temperature of the gas [K].
        gas : ct.Solution, optional (second mode of operation)
            A Cantera's ``Solution`` object from which the quantities will
            be retrieved.

    Returns
    -------
    a : float or array_like
        Sound speed [m / s]

    Examples
    --------

    Compute the speed of sound of air at 300K. First mode of operation,
    providing gamma, R, T:

    >>> from pygasflow.common import sound_speed
    >>> sound_speed(1.4, 287, 300)
    347.18870949384285

    Compute the speed of sound of air at multiple temperatures:

    >>> sound_speed(1.4, 287, [300, 500])
    array([347.18870949, 448.21869662])

    Compute the sound speed in N2 at 300K and 1atm:

    >>> import cantera as ct
    >>> gas = ct.Solution("gri30.yaml")
    >>> gas.TPX = 300, ct.one_atm, {"N2": 1}
    >>> sound_speed(gas)
    353.1256637274762
    """
    if len(args) == 3:
        gamma, R, T = args
        to_np = lambda x: x if not hasattr(x, "__iter__") else np.array(x)
        gamma = to_np(gamma)
        R = to_np(R)
        T = to_np(T)
    elif len(args) == 1:
        import cantera as ct
        gas = args[0]
        if not isinstance(gas, ct.Solution):
            raise TypeError("`gas` must be an instance of `ct.Solution`.")
        gamma = gas.cp / gas.cv
        R = ct.gas_constant / gas.mean_molecular_weight
        T = gas.T
    else:
        raise ValueError(
            "This function accepts either 1 argument (a Cantera's `Solution` "
            "object, or 3 arguments (gamma, R, T).\n"
            "Received: %s arguments" % len(args))
    return np.sqrt(gamma * R * T)


@check([0])
def pressure_coefficient(Mfs, param_name="pressure", param_value=None, stagnation=False, gamma=1.4):
    """Compute the pressure coefficient of a compressible flow.
    For supersonic flows, the pressure coefficient downstream of the
    shockwave is returned.

    Parameters
    ----------
    Mfs : float or array_like
        Free stream Mach number. If float, list, tuple is given as input,
        a conversion will be attempted. Must be Mfs > 0.
        If Mfs > 1 (supersonic case), it corresponds to the Mach number
        upstream of the shock wave.
    param_name : str, optional
        Name of the ratio. It can be:

        * ``'m'``: specify the local Mach number.
        * ``'velocity'``: specify the ratio between the local speed and the
          free stream speed, u / ufs.
        * ``'pressure'``: specify the ratio between the local pressure and the
          total pressure, p / pt.
        * ``'pressure_fs'``: specify the ratio between the local pressure and
          the free stream pressure, p / pfs.

        Default to ``'pressure'``.
    param_value : float or None, optional
        The value of the parameter. For `Mfs < 1` either ``param_value`` must
        be different than ``None``, or ``stagnation=True`` must be set.
    stagnation : bool, optional

        * ``False``: (default value) compute the local pressure coefficient.
          For subsonic Mfs, ``param_name`` and ``param_value`` must be
          provided.
        * ``True``: compute the pressure coefficient at the stagnation point.
          In this case, only ``Mfs`` is required. If `Mfs > 1`, isentropic
          flow is assumed from just downstream of the shockwave to the
          stagnation point.
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    cp : ndarray
        Pressure coefficient.

    Examples
    --------

    Pressure coefficients at the stagnation point:

    >>> from pygasflow.common import pressure_coefficient
    >>> Minf = [0.01, 0.1, 0.5, 1, 5, 10]
    >>> pressure_coefficient(Minf, stagnation=True)
    array([1.000025  , 1.0025025 , 1.06407222, 1.27561308, 1.80876996,
           1.83167098])

    Pressure coefficients at the stagnation point, by specifying a parameter:

    >>> pressure_coefficient(Minf, "velocity", 0)
    array([1.000025  , 1.0025025 , 1.06407222, 1.27561308, 1.80876996,
           1.83167098])

    References
    ----------

    * "Basic of Aerothermodynamics", by Ernst H. Hirschel
    * "Hypersonic Aerothermodynamics" by John J. Bertin

    """

    if not isinstance(param_name, str):
        raise ValueError("param_name must be a string")
    param_name = param_name.lower()
    available_pnames = ['m', 'pressure', 'pressure_fs', 'velocity']
    if param_name not in available_pnames:
        raise ValueError("param_name not recognized. Must be one of the following:\n{}".format(available_pnames))

    idx = Mfs < 1
    if any(idx) and (param_value is None) and (stagnation is False):
        raise ValueError("To compute the local pressure coefficient in the "
            "subsonic case, a parameter is required.")
    elif any(idx) and (param_value is None) and (stagnation is True):
        param_name = "pressure"
        param_value = 1

    results = np.zeros_like(Mfs, dtype=float)

    # subsonic case (isentropic flow)
    if param_name == "velocity":
        # eq (6.33)
        results[idx] = 2 / (gamma * Mfs[idx]**2) * ((1 + (gamma - 1) / 2 * Mfs[idx]**2 * (1 - param_value**2))**(gamma / (gamma - 1)) - 1)
    elif param_name == "m":
        # eq (6.34)
        results[idx] = 2 / (gamma * Mfs[idx]**2) * (((1 + (gamma - 1) / 2 * Mfs[idx]**2) / (1 + (gamma - 1) / 2 * param_value**2))**(gamma / (gamma - 1)) - 1)
    elif param_name == "pressure_fs":
        # eq (6.32)
        results[idx] = 2 / (gamma * Mfs[idx]**2) * (param_value - 1)
    else:
        # eq (6.35)
        results[idx] = 2 / (gamma * Mfs[idx]**2) * (param_value * (1 + (gamma - 1) / 2 * Mfs[idx]**2)**(gamma / (gamma - 1)) - 1)

    # NOTE: now let's deal with supersonic case
    idx = np.invert(idx)
    if stagnation or ((param_name in ["velocity", "pressure_fs"]) and (param_value == 0)):
        # Exercise 6.1 from "Hypersonic Aerothermodynamics", or a variation of
        # eq (6.63) from "Basic of Aerothermodynamics"
        pt2_p1 = rayleigh_pitot_formula(Mfs[idx], gamma)
        results[idx] = (pt2_p1 - 1) * (2 / (gamma * Mfs[idx]**2))
    else:
        # eq (6.62)
        results[idx] = 4 * (Mfs[idx]**2 - 1) / ((gamma + 1) * Mfs[idx]**2)

    return results
