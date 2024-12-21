import numpy as np
from pygasflow.isentropic import (
    sonic_density_ratio,
    sonic_pressure_ratio,
    sonic_temperature_ratio,
    sonic_sound_speed_ratio
)


def test_sonic_conditions():
    g = 1.4
    assert np.isclose(sonic_density_ratio(g), 1.5774409656148785)
    assert np.isclose(sonic_pressure_ratio(g), 1.892929158737854)
    assert np.isclose(sonic_temperature_ratio(g), 1.2)
    assert np.isclose(sonic_sound_speed_ratio(g), 1.0954451150103321)

    gammas = [1.4, 1.5]
    assert np.allclose(sonic_density_ratio(gammas), [1.57744097, 1.5625])
    assert np.allclose(sonic_pressure_ratio(gammas), [1.89292916, 1.953125])
    assert np.allclose(sonic_temperature_ratio(gammas), [1.2, 1.25])
    assert np.allclose(sonic_sound_speed_ratio(gammas), [1.09544512, 1.11803399])
