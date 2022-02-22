import numpy as np
from pygasflow.atd.thermal_conductivity import (
    thermal_conductivity_power_law, thermal_conductivity_hansen,
    thermal_conductivity_chapman_enskog, thermal_conductivity_eucken
)


def test_thermal_conductivity_power_law():
    assert np.isclose(
        thermal_conductivity_power_law(50), 0.004786)
    assert np.isclose(
        thermal_conductivity_power_law(350), 0.0282868891002169)
    assert np.allclose(
        thermal_conductivity_power_law([50, 100, 200, 300, 500, 1000]),
        np.array([0.004786  , 0.009572  , 0.019144  , 0.02519852,
            0.03696253, 0.06216331]))


def test_thermal_conductivity_hansen():
    assert np.isclose(thermal_conductivity_hansen(50), 0.004349579675632066)
    assert np.isclose(thermal_conductivity_hansen(350), 0.028246678681213885)
    assert np.allclose(
        thermal_conductivity_hansen([50, 100, 200, 300, 500, 1000]),
        np.array([0.00434958, 0.00940094, 0.01806748, 0.02513576,
            0.03640918, 0.05667643]))


def test_thermal_conductivity_chapman_enskog():
    assert np.isclose(
        thermal_conductivity_chapman_enskog(50), 0.003708574505160782)
    assert np.isclose(
        thermal_conductivity_chapman_enskog(350), 0.022242098627654833)
    assert np.isclose(
        thermal_conductivity_chapman_enskog(300, gas="O2"),
        0.019832044201669334)
    assert np.allclose(
        thermal_conductivity_chapman_enskog([50, 100, 200, 300, 500, 1000]),
        np.array([0.00370857, 0.00752377, 0.01434195, 0.01983204,
            0.02863668, 0.04547823]))

def test_thermal_conductivity_eucken():
    assert np.isclose(
        thermal_conductivity_eucken(1010, 288, 1.863e-05), 0.0255231)
