import numpy as np
import cantera as ct
from pygasflow.atd.nd_numbers import Prandtl
from pygasflow.atd.thermal_conductivity import thermal_conductivity_hansen
from pygasflow.atd.viscosity import viscosity_air_southerland

def test_Prandtl_gamma():
    assert np.isclose(Prandtl(1.4), 0.7368421052631579)
    assert np.allclose(
        Prandtl(np.array([1.1, 1.2, 1.3, 1.4, 1.5])),
        np.array([0.89795918, 0.82758621, 0.7761194, 0.73684211, 0.70588235]))

def test_Prandtl_mu_cp_k():
    T = 350
    cp = 1004
    mu = viscosity_air_southerland(T)
    k = thermal_conductivity_hansen(T)
    assert np.isclose(Prandtl(mu, cp, k), 0.7370392202421769)

def test_Prandtl_cantera():
    air = ct.Solution("gri30.yaml")
    air.TPX = 350, ct.one_atm, {"N2": 0.79, "O2": 0.21}
    assert np.isclose(Prandtl(air), 0.7139365242266411)
