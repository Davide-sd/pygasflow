import numpy as np
from pygasflow.atd.temperatures import (
    recovery_factor,
    recovery_temperature,
    reference_temperature,
)


# NOTE: the following values come from Problem 7.8


def test_recovery_factor():
    assert np.isclose(
        recovery_factor(0.74, laminar=True),
        0.8602325267042626)
    assert np.isclose(
        recovery_factor(0.74, laminar=False),
        0.9045041696510274)


def test_recovery_temperature_no_pint():
    gamma = 1.4
    Prs = 0.74
    rs = recovery_factor(Prs, laminar=False)
    Mw = 5.182
    Tw = 291.52
    assert np.isclose(
        recovery_temperature(Tw, Mw, rs, gamma),
        1707.6520161554038)

    Ml = 6.997
    Tl = 172.12
    assert np.isclose(
        recovery_temperature(Tl, Ml, rs, gamma),
        1696.50846613263)


def test_recovery_temperature_with_pint(setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    gamma = 1.4
    Prs = 0.74
    rs = recovery_factor(Prs, laminar=False)
    Mw = 5.182
    Tw = 291.52 * K
    assert np.isclose(
        recovery_temperature(Tw, Mw, rs, gamma),
        1707.6520161554038 * K)

    Ml = 6.997
    Tl = 172.12 * K
    assert np.isclose(
        recovery_temperature(Tl, Ml, rs, gamma),
        1696.50846613263 * K)


def test_reference_temperature_no_pint():
    gamma = 1.4
    Prs = 0.74
    rs = recovery_factor(Prs, laminar=False)

    Mw = 5.182
    Tw = 291.52
    Tww = 1000
    Tr_w = recovery_temperature(Tw, Mw, rs, gamma)
    assert np.isclose(
        reference_temperature(Tw, Tww, Tr=Tr_w),
        957.3090435541887)

    Ml = 6.997
    Tl = 172.12
    Twl = 800
    Tr_l = recovery_temperature(Tl, Ml, rs, gamma)
    assert np.isclose(
        reference_temperature(Tl, Twl, Tr=Tr_l),
        821.4254625491786)


def test_reference_temperature_with_pint(setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    gamma = 1.4
    Prs = 0.74
    rs = recovery_factor(Prs, laminar=False)

    Mw = 5.182
    Tw = 291.52 * K
    Tww = 1000 * K
    Tr_w = recovery_temperature(Tw, Mw, rs, gamma)
    assert np.isclose(
        reference_temperature(Tw, Tww, Tr=Tr_w),
        957.3090435541887 * K)

    Ml = 6.997
    Tl = 172.12 * K
    Twl = 800 * K
    Tr_l = recovery_temperature(Tl, Ml, rs, gamma)
    assert np.isclose(
        reference_temperature(Tl, Twl, Tr=Tr_l),
        821.4254625491786 * K)
