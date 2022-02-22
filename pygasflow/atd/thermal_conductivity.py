import os
import numpy as np
import pandas as pd
from scipy import interpolate
from pygasflow.utils.decorators import check_T


@check_T
def thermal_conductivity_chapman_enskog(T, gas="O"):
    """Compute the thermal conductivity of pure monoatomic gases.

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]
    gas : str, optional
        Possible values are:``'N'``, ``'O'``, ``'Ar'``, ``'He'``

    Returns
    -------
    k : float or array_like
        Thermal conductivity of the gas [W / (m * K)]

    References
    ----------

    * "Basic of aerothermodynamics" by Ernst Heinrich, Table 13.1
    * "Transport Phenomena" by R. Byron Bird, Warren E. Stewart,
      Edwing N. Lightfoot, Table E2
    """
    # path of the folder containing this file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # path of the folder containing the data of the plot
    data_dir = os.path.join(current_dir, "data")
    df1 = pd.read_csv(os.path.join(data_dir, "Table-13_1.csv"))
    df2 = pd.read_csv(os.path.join(data_dir, "Table-E2.csv"))
    df1[df1["Gas"] == gas]
    M = df1["M"].values[0]
    sigma = df1["sigma"].values[0] * 1e10
    eps_kappa = df1["eps_kappa"].values[0]
    kappaT_eps = T / eps_kappa
    spline = interpolate.InterpolatedUnivariateSpline(
        df2["kT_eps"], df2["visc_thermal_cond"])
    Sigma_mu = spline(kappaT_eps)
    # eq (4.16)
    return 8.3225e-02 * np.sqrt(T / M) / (sigma**2 * Sigma_mu)


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

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
    # eq (4.18)
    return (cp + 5.0 / 4.0 * R) * mu


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

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
    # eq (4.20)
    return 1.993e-03 * T**1.5 / (T + 112.0)


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

    References
    ----------

    "Basic of Aerothermodynamics", by Ernst H. Hirschel
    """
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
    return v
