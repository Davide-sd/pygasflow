import numpy as np
from pygasflow.atd.viscous_interaction import (
    chapman_rubesin, interaction_parameter, rarefaction_parameter,
    critical_distance
)


def setup_9_4():
    # Problem 9.4
    mu_argon = lambda T: T**0.75
    Minf = 12.66
    Tinf = 64.5
    Tw = 285
    Reinf_u = 985.45 # cm^-1
    x = 1
    Lsfr = 6.676 # cm
    Re_inf = Reinf_u * x * Lsfr
    Cinf = chapman_rubesin(Tw, Tinf, func=mu_argon)
    Chi = interaction_parameter(Minf, Re_inf, Cinf, laminar=True)
    V = rarefaction_parameter(Minf, Re_inf, Cinf)
    Chi_u = Chi * np.sqrt(x * Lsfr)
    x_crit_cold = critical_distance(Chi_u, weak=False, cold_wall=True)
    x_crit_hot = critical_distance(Chi_u, weak=False, cold_wall=False)
    return Cinf, Chi, V, x_crit_cold, x_crit_hot


def setup_9_5():
    # Problem 9.5
    Tw = 1500
    mu_air = lambda T: T**0.65
    Minf = 6.8
    Tinf = 231.5
    Reinf_u = 1.5e06
    L = 55
    x = 1 # m
    Cinf = chapman_rubesin(Tw, Tinf, func=mu_air)
    Chi = interaction_parameter(Minf, Reinf_u * x, Cinf)
    V = rarefaction_parameter(Minf, Reinf_u * x, Cinf)
    Chi_u = Chi * np.sqrt(x)
    x_crit = critical_distance(Chi_u, weak=False, cold_wall=False)
    return Cinf, Chi, V, x_crit


def test_chapman_rubesin():
    Cinf, _, _, _, _ = setup_9_4()
    assert np.isclose(Cinf, 0.6897293607595334)
    Cinf, _, _, _ = setup_9_5()
    assert np.isclose(Cinf, 0.5199491920651366)


def test_interaction_parameter():
    _, Chi, _, _, _ = setup_9_4()
    assert np.isclose(Chi, 20.776147167529803)
    _, Chi, _, _ = setup_9_5()
    assert np.isclose(Chi, 0.18512350420167742)


def test_rarefaction_parameter():
    _, _, V, _, _ = setup_9_4()
    assert np.isclose(V, 0.1296276361937176)
    _, _, V, _ = setup_9_5()
    assert np.isclose(V, 0.004003535990520706)


def test_critical_distance():
    _, _, _, d1, d2 = setup_9_4()
    assert np.isclose(d1, 17.051384565460914)
    assert np.isclose(d2, 180.10524947268095)
    _, _, _, d = setup_9_5()
    assert np.isclose(d, 0.00214191948799428)
