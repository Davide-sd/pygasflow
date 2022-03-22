
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from pygasflow.utils.decorators import check_T


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
    3.5100000000000003e-06

    """
    v = np.zeros_like(T)
    idx = T <= 200
    v[idx] = 0.702e-07 * T[idx]
    v[~idx] = 0.04644e-05 * T[~idx]**0.65
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
    3.2137209693578125e-06

    """
    return 1.458e-06 * T**1.5 / (T + 110.4)


@check_T
def viscosity_chapman_enskog(T, gas="air", M=None, sigma=None, Sigma_mu=None):
    """Compute the viscosity of pure motoatomic or polyatomic gases using
    Chapman-Enskog theory.

    There are two mode of operation:

    1. by providing the ``gas`` keyword argument, the algorithm will load
       pre-computed values of ``M``, ``sigma`` and ``Sigma_mu``.
       ``viscosity_chapman_enskog(T, gas="air" [optional])``
    2. by providing ``M``, ``sigma`` and ``Sigma_mu``. This is going to
       disregard the value of ``gas``.
       ``viscosity_chapman_enskog(T, M=M, sigma=sigma, Sigma_mu=Sigma_mu)``

    Parameters
    ----------
    T : float or array_like
        Temperature of the air in [K]
    gas : str, optional
        Possible values are: ``'air'``, ``'N2'``, ``'O2'``, ``'NO'``, ``'N'``,
        ``'O'``, ``'Ar'``, ``'He'``
    M : float or None, optional
        Molecular weigth [kg / kmole]
    sigma : float or None, optional
        Collision parameter (first Lennard-Jones parameter) [1e10 m]
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
    3.4452054654966263e-06

    Compute the viscosity of molecular oxygen at T=300K:

    >>> viscosity_chapman_enskog(300, gas="O2")
    1.8423646870376057e-05

    References
    ----------

    * "Basic of aerothermodynamics" by Ernst Heinrich, Table 13.1
    * "Transport Phenomena" by R. Byron Bird, Warren E. Stewart,
      Edwing N. Lightfoot, Table E2

    """
    if any([t is None for t in [M, sigma, Sigma_mu]]):
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
    return 2.6693e-06 * np.sqrt(M * T) / (sigma**2 * Sigma_mu)
