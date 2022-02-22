import numpy as np
from pygasflow.atd.viscosity import (
    viscosity_air_power_law, viscosity_air_southerland,
    viscosity_chapman_enskog
)


def test_viscosity_air_power_law():
    assert np.isclose(viscosity_air_power_law(50), 3.5100000000000003e-06)
    assert np.isclose(viscosity_air_power_law(350), 2.091878420800692e-05)
    assert np.allclose(
        viscosity_air_power_law([50, 100, 200, 300, 500, 1000]),
        np.array([3.51000000e-06, 7.02000000e-06, 1.40400000e-05,
            1.89243524e-05, 2.63768369e-05, 4.13896936e-05]))


def test_viscosity_air_southerland():
    assert np.isclose(viscosity_air_southerland(50), 3.2137209693578125e-06)
    assert np.isclose(viscosity_air_southerland(350), 2.073596616497331e-05)
    assert np.allclose(
        viscosity_air_southerland([50, 100, 200, 300, 500, 1000]),
        np.array([3.21372097e-06, 6.92965779e-06, 1.32855887e-05,
            1.84600152e-05, 2.67053335e-05, 4.15219815e-05]))


def test_viscosity_chapman_enskog():
    assert np.isclose(viscosity_chapman_enskog(50), 3.4452054654966263e-06)
    assert np.isclose(viscosity_chapman_enskog(350), 2.0662548278179808e-05)
    assert np.isclose(
        viscosity_chapman_enskog(300, gas="O2"), 1.8423646870376057e-05)
    assert np.allclose(
        viscosity_chapman_enskog([50, 100, 200, 300, 500, 1000]),
        np.array([3.44520547e-06, 6.98946374e-06, 1.33234370e-05,
            1.84236469e-05, 2.66030087e-05, 4.22485402e-05]))
