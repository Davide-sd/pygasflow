import numpy as np
from pygasflow.isentropic import (
    sonic_density_ratio,
    sonic_pressure_ratio,
    sonic_temperature_ratio,
    sonic_sound_speed_ratio,
    critical_velocity_ratio,
)
import pytest


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


@pytest.mark.parametrize("m, gamma, expected_vel_ratio", [
    (2, 1.4, 1.63299316),
    ([2, 3], 1.4, [1.63299316, 1.96396101]),
    (0.5, 1.4, 0.5345224838248488),
    (1, 1.4, 1),
    (0, 1.4, 0),
    (2, 1.2, 1.77281052),
    ([2, 3], 1.2, [1.77281052, 2.28265773]),
    (0.5, 1.2, 0.5179697702828123),
    (1, 1.2, 1),
    (0, 1.2, 0),
])
def test_critical_velocity_ratio(m, gamma, expected_vel_ratio):
    vr = critical_velocity_ratio(m, gamma)
    assert np.allclose(vr, expected_vel_ratio)
