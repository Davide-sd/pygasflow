import os
import numpy as np
import pandas as pd
from scipy import interpolate
from pygasflow.utils.decorators import check_T
from pygasflow.utils.common import (
    _is_pint_quantity,
    _parse_pint_units,
    _check_mix_of_units_and_dimensionless
)


@check_T
def thermal_conductivity_chapman_enskog(T, gas="O"):
    """Compute the thermal conductivity of pure monoatomic gases.

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]
    gas : str, optional
        Possible values are: ``['N', 'O', 'Ar', 'He']``

    Returns
    -------
    k : float or array_like
        Thermal conductivity of the gas [W / (m * K)]

    Examples
    --------

    >>> import numpy as np
    >>> from pygasflow.atd import thermal_conductivity_chapman_enskog
    >>> thermal_conductivity_chapman_enskog(300, gas="O")
    np.float64(0.049434309555779335)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> thermal_conductivity_chapman_enskog(300 * ureg.K)
    <Quantity(0.0494343096, 'watt / meter / kelvin')>

    Comparison of the thermal conductivity of different gases over a 
    range of temperatures.

    .. plot::
       :context: close-figs
       :include-source: True

       import numpy as np
       import matplotlib.pyplot as plt
       from pygasflow.atd import thermal_conductivity_chapman_enskog
       T = np.linspace(200, 1000)
       gases = ['N', 'O', 'Ar', 'He']
       fig, ax = plt.subplots()
       for gas in gases:
           ax.plot(T, thermal_conductivity_chapman_enskog(T, gas=gas), label=gas)
       ax.legend()
       ax.set_xlabel("Temperature [K]")
       ax.set_ylabel("Thermal Conductivity [W / (m * K)]")
       plt.show()
    
    References
    ----------

    * "Basic of aerothermodynamics" by Ernst Heinrich, Table 13.1
    * "Transport Phenomena" by R. Byron Bird, Warren E. Stewart,
      Edwing N. Lightfoot, Table E2
    
    """
    available_gases = ["O", "N", "Ar", "He"]
    if gas not in available_gases:
        raise ValueError(
            f"`gas` must be one of the following: {available_gases}.\n"
            f"Instead, '{gas}' was provided."
        )

    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K").magnitude

    # path of the folder containing this file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # path of the folder containing the data of the plot
    data_dir = os.path.join(current_dir, "data")
    df1 = pd.read_csv(os.path.join(data_dir, "Table-13_1.csv"))
    df2 = pd.read_csv(os.path.join(data_dir, "Table-E2.csv"))
    if gas not in df1["Gas"].values:
        raise KeyError(
            f"The provided gas, '{gas}', is not included in the dataframe."
            " Please, read the documentation of this function to"
            " understand which gases are supported."

        )
    df1 = df1[df1["Gas"] == gas]
    M = df1["M"].values[0]
    sigma = df1["sigma"].values[0] * 1e10
    eps_kappa = df1["eps_kappa"].values[0]
    kappaT_eps = T / eps_kappa
    spline = interpolate.InterpolatedUnivariateSpline(
        df2["kT_eps"], df2["visc_thermal_cond"])
    Sigma_mu = spline(kappaT_eps)

    # eq (4.16)
    k = 8.3225e-02 * np.sqrt(T / M) / (sigma**2 * Sigma_mu)

    if is_pint:
        k *= _parse_pint_units("W / (m * K)")
    return k


def thermal_conductivity_eucken(cp, R, mu):
    """Compute the thermal conductivity of gases with the semi-empirical
    Eucken formula.

    Parameters
    ----------
    cp : float or array_like
        Specific heat at constant pressure. [J / (kg * K)]
    R : float or array_like
        Specific gas constant, which is equal to `R0 / M` with M the molecular
        weight.
    mu : float or array_like
        Viscosity [kg / (m * s)].

    Returns
    -------
    k : float or array_like
        Thermal conductivity of the gas [W / (m * K)]
    
    Examples
    --------

    >>> import numpy as np
    >>> from pygasflow.atd import thermal_conductivity_eucken
    >>> cp, R, mu = 1010, 288, 1.863e-05
    >>> thermal_conductivity_eucken(cp, R, mu)
    np.float64(0.0255231)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> cp = 1010 * ureg.J / (ureg.kg * ureg.K)
    >>> R = 288 * ureg.J / (ureg.kg * ureg.K)
    >>> mu = 1.863e-05 * ureg.kg / (ureg.m * ureg.s)
    >>> thermal_conductivity_eucken(cp, R, mu)
    <Quantity(0.0255231, 'watt / meter / kelvin')>

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
    _check_mix_of_units_and_dimensionless([cp, R, mu])
    is_pint = any(_is_pint_quantity(q) for q in [cp, R, mu])
    if is_pint:
        cp = cp.to("J / kg / K").magnitude
        mu = mu.to("kg / m / s").magnitude
        R = R.to("J / (kg * K)").magnitude

    # eq (4.18)
    k = (cp + 5.0 / 4.0 * R) * mu

    if is_pint:
        k *= _parse_pint_units("W / (m * K)")
    return k


@check_T
def thermal_conductivity_hansen(T):
    """Compute air's thermal conductivity.

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]

    Returns
    -------
    k : float or array_like
        Thermal conductivity of the gas [W / (m * K)]
    
    Examples
    --------

    >>> import numpy as np
    >>> from pygasflow.atd import thermal_conductivity_hansen
    >>> thermal_conductivity_hansen(300)
    np.float64(0.0251357567)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> thermal_conductivity_hansen(300 * ureg.K)
    <Quantity(0.0251357567, 'watt / meter / kelvin')>

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K").magnitude

    # eq (4.20)
    k = 1.993e-03 * T**1.5 / (T + 112.0)

    if is_pint:
        k *= _parse_pint_units("W / (m * K)")
    return k


@check_T
def thermal_conductivity_power_law(T):
    """Compute air's thermal conductivity.

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]

    Returns
    -------
    k : float or array_like
        Thermal conductivity of the gas [W / (m * K)]
    
    Examples
    --------

    >>> import numpy as np
    >>> from pygasflow.atd import thermal_conductivity_power_law
    >>> thermal_conductivity_power_law(300)
    np.float64(0.0251985236)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> thermal_conductivity_power_law(300 * ureg.K)
    <Quantity(0.0251985236, 'watt / meter / kelvin')>

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K").magnitude

    is_scalar = False
    if not isinstance(T, np.ndarray):
        is_scalar = True
        T = np.array([T], dtype=float)

    # eq (4.21) - (4.22)
    v = np.zeros_like(T)
    idx = T <= 200
    v[idx] = 9.572e-05 * T[idx]
    v[~idx] = 34.957e-05 * T[~idx]**0.75
    if is_scalar:
        return v[0]

    if is_pint:
        v *= _parse_pint_units("W / (m * K)")
    return v
