import numpy as np
from pygasflow.atd.viscosity import (
    viscosity_air_power_law, viscosity_air_southerland,
    viscosity_chapman_enskog
)
import pytest


@pytest.mark.parametrize("use_pint", [False, True])
def test_viscosity_air_power_law(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])

    expected_r1 = 3.5100000000000003e-06
    expected_r2 = 2.091878420800692e-05
    expected_r3 = np.array([3.51000000e-06, 7.02000000e-06, 1.40400000e-05,
            1.89243524e-05, 2.63768369e-05, 4.13896936e-05])

    if use_pint:
        T1, T2, T3 = [t * K for t in [T1, T2, T3]]

    r1 = viscosity_air_power_law(T1)
    r2 = viscosity_air_power_law(T2)
    r3 = viscosity_air_power_law(T3)

    if use_pint:
        units = set(r.units for r in [r1, r2, r3])
        assert len(units) == 1
        assert units.pop() == kg / (m * s)
        r1, r2, r3 = [r.magnitude for r in [r1, r2, r3]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("use_pint", [False, True])
def test_viscosity_air_southerland(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])

    expected_r1 = 3.2137209693578125e-06
    expected_r2 = 2.073596616497331e-05
    expected_r3 = np.array([3.21372097e-06, 6.92965779e-06, 1.32855887e-05,
            1.84600152e-05, 2.67053335e-05, 4.15219815e-05])

    if use_pint:
        T1, T2, T3 = [t * K for t in [T1, T2, T3]]

    r1 = viscosity_air_southerland(T1)
    r2 = viscosity_air_southerland(T2)
    r3 = viscosity_air_southerland(T3)

    if use_pint:
        units = set(r.units for r in [r1, r2, r3])
        assert len(units) == 1
        assert units.pop() == kg / (m * s)
        r1, r2, r3 = [r.magnitude for r in [r1, r2, r3]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("use_pint", [False, True])
def test_viscosity_chapman_enskog_mode_1(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    T1 = 50
    T2 = 350
    T3 = np.array([50, 100, 200, 300, 500, 1000])
    T4 = 300

    expected_r1 = 3.4452054654966263e-06
    expected_r2 = 2.0662548278179808e-05
    expected_r3 = np.array([3.44520547e-06, 6.98946374e-06, 1.33234370e-05,
            1.84236469e-05, 2.66030087e-05, 4.22485402e-05])
    expected_r4 = 2.069427983599303e-05

    if use_pint:
        T1, T2, T3, T4 = [t * K for t in [T1, T2, T3, T4]]

    r1 = viscosity_chapman_enskog(T1)
    r2 = viscosity_chapman_enskog(T2)
    r3 = viscosity_chapman_enskog(T3)
    r4 = viscosity_chapman_enskog(T4, gas="O2")

    if use_pint:
        units = set(r.units for r in [r1, r2, r3, r4])
        assert len(units) == 1
        assert units.pop() == kg / (m * s)
        r1, r2, r3, r4 = [r.magnitude for r in [r1, r2, r3, r4]]

    assert np.isclose(r1, expected_r1, rtol=1e-6, atol=1e-10)
    assert np.isclose(r2, expected_r2, rtol=1e-6, atol=1e-10)
    assert np.allclose(r3, expected_r3, rtol=1e-6, atol=1e-10)
    assert np.allclose(r4, expected_r4, rtol=1e-6, atol=1e-10)


def test_viscosity_chapman_enskog_mode_1_gases():
    # verify that the data table doesn't contain redundant data, specifically:
    # 1. the same numeric values are not used for different gases.
    # 2. these gases are supported.

    gases = ['air', 'N2', 'O2', 'NO', 'N', 'O', 'Ar', 'He']
    T = np.linspace(200, 1000, 10)
    results = {g: viscosity_chapman_enskog(T, gas=g) for g in gases}

    gases = set(gases)
    for gas in gases:
        other_gases = gases.difference([gas])
        for k in other_gases:
            assert not np.allclose(results[gas], results[k])


def test_viscosity_chapman_enskog_mode_1_errors():
    # because 'test' is not a valid gas
    pytest.raises(KeyError, lambda: viscosity_chapman_enskog(300, gas="test"))


def test_viscosity_chapman_enskog_mode_2():
    # from table 13.1, these are the values for air
    # M, sigma = 28.9644, 3.62E-10 * 1e10
    # which computes Sigma_mu = 1.03061757
    # Let's change these values a little bit:

    M, sigma = 28.9, 3.7E-10 * 1e10
    Sigma_mu = 1.1
    T = 300
    res_mode_1 = viscosity_chapman_enskog(T, gas="air")
    res_mode_2 = viscosity_chapman_enskog(T, M=M, sigma=sigma, Sigma_mu=Sigma_mu)
    assert not np.isclose(res_mode_1, res_mode_2)


def test_viscosity_chapman_enskog_mode_2_errors(setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    T, M, sigma, Sigma_mu = 300, 28.9, 3.7E-10 * 1e10, 1.1

    # everything is fine if quantities have appropriate dimensions
    viscosity_chapman_enskog(
        T*K,
        M=M*kg/kmol,
        sigma=sigma*m,
        Sigma_mu=Sigma_mu
    )

    # because Sigma_mu has dimensions
    pytest.raises(
        ValueError,
        lambda: viscosity_chapman_enskog(
            T*K,
            M=M*kg/kmol,
            sigma=sigma*m,
            Sigma_mu=Sigma_mu*s
        )
    )
    # because sigma doesn't have dimensions
    pytest.raises(
        ValueError,
        lambda: viscosity_chapman_enskog(
            T*K,
            M=M*kg/kmol,
            sigma=sigma,
            Sigma_mu=Sigma_mu*s
        )
    )
    # because M doesn't have dimensions
    pytest.raises(
        ValueError,
        lambda: viscosity_chapman_enskog(
            T*K,
            M=M,
            sigma=sigma*m,
            Sigma_mu=Sigma_mu*s
        )
    )

    # because T doesn't have dimensions
    pytest.raises(
        ValueError,
        lambda: viscosity_chapman_enskog(
            T,
            M=M*kg/kmol,
            sigma=sigma*m,
            Sigma_mu=Sigma_mu*s
        )
    )
