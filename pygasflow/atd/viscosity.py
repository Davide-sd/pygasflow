
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
def viscosity_air_power_law(T):
    """Compute air's viscosity using a power law:

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]

    Returns
    -------
    mu : float or array_like
        Viscosity [kg / (m * s)]

    Notes
    -----
    The following equation is being used:

    * mu(T) = 0.702e-07 * T for T <= 200
    * mu(T) = 0.04644e-05 * T**0.65 for T > 200

    Examples
    --------

    Compute air's viscosity at T=50K:

    >>> from pygasflow.atd.viscosity import viscosity_air_power_law
    >>> viscosity_air_power_law(50)
    np.float64(3.5100000000000003e-06)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> viscosity_air_power_law(50 * ureg.K)
    <Quantity(3.51e-06, 'kilogram / meter / second')>

    """
    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K")
        T = T.magnitude

    v = np.zeros_like(T)
    idx = T <= 200
    v[idx] = 0.702e-07 * T[idx]
    v[~idx] = 0.04644e-05 * T[~idx]**0.65

    if is_pint:
        v *= _parse_pint_units("kg / (m * s)")
    return v


@check_T
def viscosity_air_southerland(T):
    """Compute the viscosity of air with Southerland equation.

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]

    Returns
    -------
    mu : float or array_like
        Viscosity [kg / (m * s)]

    Examples
    --------

    Compute air's viscosity at T=50K:

    >>> from pygasflow.atd.viscosity import viscosity_air_southerland
    >>> viscosity_air_southerland(50)
    np.float64(3.2137209693578125e-06)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> viscosity_air_southerland(50 * ureg.K)
    <Quantity(3.21372097e-06, 'kilogram / meter / second')>

    """
    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K")
        T = T.magnitude

    mu = 1.458e-06 * T**1.5 / (T + 110.4)

    if is_pint:
        mu *= _parse_pint_units("kg / (m * s)")
    return mu


@check_T
def viscosity_chapman_enskog(T, gas="air", M=None, sigma=None, Sigma_mu=None):
    """Compute the viscosity of pure motoatomic or polyatomic gases using
    Chapman-Enskog theory.

    There are two mode of operation:

    1. by providing the ``gas`` keyword argument, the algorithm will load
       pre-computed values of ``M``, ``sigma`` and ``Sigma_mu``.
       ``viscosity_chapman_enskog(T, gas="air" [optional])``
    2. by providing ``M``, ``sigma`` and ``Sigma_mu``. This is going to
       disregard the value of ``gas``, and computes the viscosity of air.
       ``viscosity_chapman_enskog(T, M=M, sigma=sigma, Sigma_mu=Sigma_mu)``

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]
    gas : str, optional
        Possible values are: ``['air', 'N2', 'O2', 'NO', 'N', 'O', 'Ar', 'He']``
    M : float or None, optional
        Molecular weigth [kg / kmole]
    sigma : float or None, optional
        Collision parameter (first Lennard-Jones parameter) [m]
    Sigma_mu : float or None, optional
        Dimensionless collision integral

    Returns
    -------
    mu : float or array_like
        Viscosity [kg / (m * s)]

    Examples
    --------

    Compute air's viscosity at T=50K:

    >>> from pygasflow.atd.viscosity import viscosity_chapman_enskog
    >>> viscosity_chapman_enskog(50)
    np.float64(3.4452054654966263e-06)

    Compute the viscosity of molecular oxygen at T=300K:

    >>> viscosity_chapman_enskog(300, gas="O2")
    np.float64(2.069427983599303e-05)

    Pint quantities can be used as well:

    >>> import pint
    >>> import pygasflow
    >>> ureg = pint.UnitRegistry()
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> viscosity_chapman_enskog(50 * ureg.K)
    <Quantity(3.44520547e-06, 'kilogram / meter / second')>

    Comparison of the viscosity of different gases over a
    range of temperatures.

    .. plot::
       :context: close-figs
       :include-source: True

       import numpy as np
       import matplotlib.pyplot as plt
       from pygasflow.atd import viscosity_chapman_enskog
       T = np.linspace(200, 1000)
       gases = ['air', 'N2', 'O2', 'NO', 'N', 'O', 'Ar', 'He']
       fig, ax = plt.subplots()
       for gas in gases:
           ax.plot(T, viscosity_chapman_enskog(T, gas=gas), label=gas)
       ax.legend()
       ax.set_xlabel("Temperature [K]")
       ax.set_ylabel("Viscosity [kg / (m * s)]")
       plt.show()

    References
    ----------

    * "Basic of aerothermodynamics" by Ernst Heinrich, Table 13.1
    * "Transport Phenomena" by R. Byron Bird, Warren E. Stewart,
      Edwing N. Lightfoot, Table E2

    """

    if all(t is not None for t in [M, sigma, Sigma_mu]):
        # SECOND MODE OF OPERATION
        # Quick dimensional check: don't allow mix of units and unitless
        # quantities, as it is dangerous for the user.
        # Don't look at Sigma_mu, it will be checked later
        _check_mix_of_units_and_dimensionless([T, M, sigma])

    is_pint = _is_pint_quantity(T)
    if is_pint:
        T = T.to("K").magnitude
    if _is_pint_quantity(M):
        M = M.to("kg / kmol").magnitude
    if _is_pint_quantity(sigma):
        sigma = sigma.to("m").magnitude
    if _is_pint_quantity(Sigma_mu):
        if not Sigma_mu.unitless:
            raise ValueError(
                "`Sigma_mu` must be dimensionless. Instead, this dimension"
                f" was provided {Sigma_mu.units}"
            )

    if any([t is None for t in [M, sigma, Sigma_mu]]):
        # FIRST MODE OF OPERATION

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

    mu = 2.6693e-06 * np.sqrt(M * T) / (sigma**2 * Sigma_mu)

    if is_pint:
        mu *= _parse_pint_units("kg / (m * s)")
    return mu
