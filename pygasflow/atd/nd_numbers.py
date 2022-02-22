import numpy as np
import cantera as ct

def Prandtl(*args):
    """Compute the Prandtl number.

    There are 3 modes of operations:

    * ``Prandtl(mu, cp, k)``
    * ``Prandtl(gas)`` where ``gas`` is a Cantera's ``Solution`` object.
    * ``Prandtl(gamma)`` which is a good approximation for both monoatomic and
      polyatomic gases. It is derived from Eucken's formula for thermal
      conductivity.

    Parameters
    ----------
    mu : float or array_like
        Viscosity of the gas.
    cp : float or array_like
        Specific heat at constant pressure.
    k : float or array_like
        Thermal conductivity of the gas.
    gamma : float
        Specific heats ratio. Default to None. Must be gamma > 1.
    gas : ct.Solution
        A Cantera's ``Solution`` object.

    Returns
    -------
    Pr : float or array_like
        Prandtl number.

    Examples
    --------

    Compute the Prandtl number of air with specific heat ratio of 1.4:

    >>> from pygasflow.atd.numbers import Prandtl
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

    """
    if len(args) == 1:
        if isinstance(args[0], (int, float, np.ndarray)):
            gamma = args[0]
            # eq (4.19)
            return 4 * gamma / (9 * gamma - 5)
        elif isinstance(args[0], ct.Solution):
            gas = args[0]
            return gas.viscosity * gas.cp_mass / gas.thermal_conductivity
        raise ValueError(
            "When 1 argument is provided, it must be an instance of int, or "
            "float, or np.ndarray or ct.Solution"
        )
    elif len(args) == 3:
        mu, cp, k = args
        return mu * cp / k
    else:
        raise ValueError(
            "This function accepts 1 or 3 arguments. "
            "Please, read the documentation.")
