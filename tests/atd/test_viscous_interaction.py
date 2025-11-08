import numpy as np
from pygasflow.atd import (
    chapman_rubesin,
    interaction_parameter,
    rarefaction_parameter,
    critical_distance
)
import pytest


def setup_Problem_9_4(use_pint, units):
    K, m, s, kg, J, W, Q_, kmol, atm = units

    # Values from Problem 9.4 and Table 9.6
    mu_argon = lambda T: T**0.75
    Minf = 12.66
    Tinf = 64.5
    Tw = 285
    Reinf_u = 985.45 # cm^-1
    x = 1
    Lsfr = 6.676 # cm

    if use_pint:
        Tinf *= K
        Tw *= K
        Reinf_u /= (m / 100)
        Lsfr *= (m / 100)

    Re_inf = Reinf_u * x * Lsfr
    Cinf = chapman_rubesin(Tw, Tinf, func=mu_argon)
    Chi = interaction_parameter(Minf, Re_inf, Cinf, laminar=True)
    V = rarefaction_parameter(Minf, Re_inf, Cinf)
    Chi_u = Chi * np.sqrt(x * Lsfr)
    x_crit_cold = critical_distance(Chi_u, weak=False, cold_wall=True)
    x_crit_hot = critical_distance(Chi_u, weak=False, cold_wall=False)
    return Cinf, Chi, V, x_crit_cold, x_crit_hot


def setup_Problem_9_5(use_pint, units):
    K, m, s, kg, J, W, Q_, kmol, atm = units

    # Values from Problem 9.5 and Table 7.9
    Tw = 1500
    mu_air = lambda T: T**0.65
    Minf = 6.8
    Tinf = 231.5
    Reinf_u = 1.5e06
    L = 55
    x = 1 # m

    if use_pint:
        x *= m
        L *= m
        Tinf *= K
        Tw *= K
        Reinf_u /= m

    Cinf = chapman_rubesin(Tw, Tinf, func=mu_air)
    Chi = interaction_parameter(Minf, Reinf_u * x, Cinf)
    V = rarefaction_parameter(Minf, Reinf_u * x, Cinf)
    Chi_u = Chi * np.sqrt(x)
    x_crit = critical_distance(Chi_u, weak=False, cold_wall=False)
    return Cinf, Chi, V, x_crit


@pytest.mark.parametrize("use_pint", [False, True])
def test_chapman_rubesin(use_pint, setup_pint_registry):
    Cinf, _, _, _, _ = setup_Problem_9_4(use_pint, setup_pint_registry)
    assert np.isclose(Cinf, 0.6897293607595334)
    assert isinstance(Cinf, float) is (not use_pint)
    if use_pint:
        assert Cinf.unitless

    Cinf, _, _, _ = setup_Problem_9_5(use_pint, setup_pint_registry)
    assert np.isclose(Cinf, 0.5199491920651366)
    assert isinstance(Cinf, float) is (not use_pint)
    if use_pint:
        assert Cinf.unitless


@pytest.mark.parametrize("use_pint", [False, True])
def test_interaction_parameter(use_pint, setup_pint_registry):
    _, Chi, _, _, _ = setup_Problem_9_4(use_pint, setup_pint_registry)
    assert np.isclose(Chi, 20.776147167529803)
    assert isinstance(Chi, float) is (not use_pint)
    if use_pint:
        assert Chi.unitless

    _, Chi, _, _ = setup_Problem_9_5(use_pint, setup_pint_registry)
    assert np.isclose(Chi, 0.18512350420167742)
    assert isinstance(Chi, float) is (not use_pint)
    if use_pint:
        assert Chi.unitless


@pytest.mark.parametrize("use_pint", [False, True])
def test_rarefaction_parameter(use_pint, setup_pint_registry):
    _, _, V, _, _ = setup_Problem_9_4(use_pint, setup_pint_registry)
    assert np.isclose(V, 0.1296276361937176)
    assert isinstance(V, float) is (not use_pint)
    if use_pint:
        assert V.unitless

    _, _, V, _ = setup_Problem_9_5(use_pint, setup_pint_registry)
    assert np.isclose(V, 0.004003535990520706)
    assert isinstance(V, float) is (not use_pint)
    if use_pint:
        assert V.unitless


@pytest.mark.parametrize("use_pint", [False, True])
def test_critical_distance(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    expected_d1 = 17.051384565460914 # cm
    expected_d2 = 180.10524947268095 # cm
    expected_d = 0.00214191948799428 # m

    if use_pint:
        expected_d1 *= (m / 100)
        expected_d2 *= (m / 100)
        expected_d *= m

    _, _, _, d1, d2 = setup_Problem_9_4(use_pint, setup_pint_registry)
    assert np.isclose(d1, expected_d1)
    assert isinstance(d1, float) is (not use_pint)
    assert np.isclose(d2, expected_d2)
    assert isinstance(d2, float) is (not use_pint)

    _, _, _, d = setup_Problem_9_5(use_pint, setup_pint_registry)
    assert np.isclose(d, expected_d)
    assert isinstance(d, float) is (not use_pint)
