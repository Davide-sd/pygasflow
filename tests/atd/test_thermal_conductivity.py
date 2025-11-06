import numpy as np
from pygasflow.atd.thermal_conductivity import (
    thermal_conductivity_power_law, thermal_conductivity_hansen,
    thermal_conductivity_chapman_enskog, thermal_conductivity_eucken
)
import pint
import pytest
import pygasflow


ureg = pint.UnitRegistry()
pygasflow.defaults.pint_ureg = ureg
W = ureg.W
m = ureg.m
s = ureg.s
K = ureg.K
J = ureg.J
kg = ureg.kg


@pytest.mark.parametrize("use_pint", [False, True])
def test_thermal_conductivity_power_law(use_pint):
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])

    expected_r1 = 0.004786
    expected_r2 = 0.0282868891002169
    expected_r3 = np.array([0.004786  , 0.009572  , 0.019144  , 0.02519852,
            0.03696253, 0.06216331])

    if use_pint:
        T1, T2, T3 = [t * K for t in [T1, T2, T3]]

    r1 = thermal_conductivity_power_law(T1)
    r2 = thermal_conductivity_power_law(T2)
    r3 = thermal_conductivity_power_law(T3)

    if use_pint:
        units = set(r.units for r in [r1, r2, r3])
        assert len(units) == 1
        assert units.pop() == W / (m * K)
        r1, r2, r3 = [r.magnitude for r in [r1, r2, r3]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("use_pint", [False, True])
def test_thermal_conductivity_hansen(use_pint):
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])

    expected_r1 = 0.004349579675632066
    expected_r2 = 0.028246678681213885
    expected_r3 = np.array([0.00434958, 0.00940094, 0.01806748, 0.02513576,
            0.03640918, 0.05667643])

    if use_pint:
        T1, T2, T3 = [t * K for t in [T1, T2, T3]]

    r1 = thermal_conductivity_hansen(T1)
    r2 = thermal_conductivity_hansen(T2)
    r3 = thermal_conductivity_hansen(T3)

    if use_pint:
        units = set(r.units for r in [r1, r2, r3])
        assert len(units) == 1
        assert units.pop() == W / (m * K)
        r1, r2, r3 = [r.magnitude for r in [r1, r2, r3]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("use_pint", [False, True])
def test_thermal_conductivity_chapman_enskog(use_pint):
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])
    T4 = 300

    expected_r1 = 0.008809307530789567
    expected_r2 = 0.05695432760614933
    expected_r3 = np.array([0.00880931, 0.01639032, 0.03317209, 0.04943431,
        0.077221 , 0.129541])
    expected_r4 = 0.047112985675833156

    if use_pint:
        T1, T2, T3, T4 = [t * K for t in [T1, T2, T3, T4]]

    r1 = thermal_conductivity_chapman_enskog(T1)
    r2 = thermal_conductivity_chapman_enskog(T2)
    r3 = thermal_conductivity_chapman_enskog(T3)
    r4 = thermal_conductivity_chapman_enskog(T4, gas="N")

    if use_pint:
        units = set(r.units for r in [r1, r2, r3, r4])
        assert len(units) == 1
        assert units.pop() == W / (m * K)
        r1, r2, r3, r4 = [r.magnitude for r in [r1, r2, r3, r4]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)
    assert np.isclose(r4, expected_r4, rtol=1e-6, atol=1e-10)



def test_thermal_conductivity_chapman_enskog_gases():
    pytest.raises(
        ValueError,
        lambda : thermal_conductivity_chapman_enskog(50, gas="air")
    )
    pytest.raises(
        ValueError,
        lambda : thermal_conductivity_chapman_enskog(50, gas="H2")
    )

    # verify that the data table doesn't contain redundant data, specifically:
    # 1. the same numeric values are not used for different gases.
    # 2. these gases are supported.
    gases = ['N', 'O', 'Ar', 'He']
    T = np.linspace(200, 1000, 10)
    results = {g: thermal_conductivity_chapman_enskog(T, gas=g) for g in gases}

    gases = set(gases)
    for gas in gases:
        other_gases = gases.difference([gas])
        for k in other_gases:
            assert not np.allclose(results[gas], results[k])


@pytest.mark.parametrize("use_pint", [False, True])
def test_thermal_conductivity_eucken(use_pint):
    cp, R, mu = 1010, 288, 1.863e-05

    if use_pint:
        cp *= J / (kg * K)
        mu *= kg / (m * s)
        R *= J / (kg * K)

    k = thermal_conductivity_eucken(cp, R, mu)

    if use_pint:
        assert k.units == W / (m * K)
        k = k.magnitude

    assert np.isclose(k, 0.0255231)

def test_thermal_conductivity_eucken_errors():
    # mix of units and unitless quantities -> error

    cp, R, mu = 1010, 288, 1.863e-05

    # mu is unitless
    pytest.raises(
        ValueError,
        lambda: thermal_conductivity_eucken(
            cp * J / (kg * K),
            R * J / (kg * K),
            mu
        )
    )
    # cp is unitless
    pytest.raises(
        ValueError,
        lambda: thermal_conductivity_eucken(
            cp,
            R * J / (kg * K),
            mu * kg / (m * s)
        )
    )
    # R is unitless
    pytest.raises(
        ValueError,
        lambda: thermal_conductivity_eucken(
            cp * J / (kg * K),
            R,
            mu * kg / (m * s)
        )
    )

