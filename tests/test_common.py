
import cantera as ct
import numpy as np
from pygasflow.common import pressure_coefficient, sound_speed


def test_pressure_coefficient_single_value_stagnation():
    g = 1.4

    assert np.isclose(pressure_coefficient(0.5, stagnation=True), 1.06407222)
    assert np.isclose(pressure_coefficient(5, stagnation=True), 1.80876996)


def test_pressure_coefficient_multiple_values_stagnation():
    g = 1.4
    Minf = [0.01, 0.1, 0.5, 1, 5, 10]
    assert np.allclose(
        pressure_coefficient(Minf, stagnation=True),
        np.array([1.000025  , 1.0025025 , 1.06407222, 1.27561308,
            1.80876996, 1.83167098])
    )


def test_pressure_coefficient_trivial_cases():
    g = 1.4

    # eq (6.32)
    assert np.isclose(pressure_coefficient(0.5, "pressure_fs", 1), 0)
    # eq (6.33)
    assert np.isclose(pressure_coefficient(0.5, "velocity", 1), 0)
    # eq (6.34)
    assert np.isclose(pressure_coefficient(0.5, "m", 0.5), 0)
    # eq (6.62)
    assert np.isclose(pressure_coefficient(1), 0)


def test_sound_speed_first_mode_operation():
    g = 1.4
    R = 287
    assert np.isclose(sound_speed(g, R, 0), 0)
    assert np.isclose(sound_speed(g, R, 300), 347.18870949384285)
    assert np.allclose(sound_speed(g, R, [300, 500]), np.array([347.18870949, 448.21869662]))


def test_sound_speed_second_mode_operation():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 300, ct.one_atm, {"N2": 1}
    assert np.isclose(sound_speed(gas), 353.1256637274762)
