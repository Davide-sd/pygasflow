import numpy as np
from pygasflow.atd.avf.thickness_fp import (
    deltas_lam_ic, deltas_tur_ic, deltas_lam_c, deltas_tur_c
)
import pytest


@pytest.mark.parametrize("use_pint", [False, True])
def test_deltas_lam_ic(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # from Problem 7.1 (Hirschel)
    x = 1
    Re = 1e06

    expected_delta = 0.005
    expected_delta_1 = 0.0017208000000000002
    expected_delta_2 = 0.0006641
    expected_H12 = 2.5911760277066707

    if use_pint:
        x *= m
        expected_delta *= m
        expected_delta_1 *= m
        expected_delta_2 *= m

    r = deltas_lam_ic(x, Re, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 4
    assert hasattr(r, "show")
    r = deltas_lam_ic(x, Re, to_dict=True)
    assert hasattr(r, "show")
    assert isinstance(r, dict) and len(r) == 4
    assert np.isclose(r["delta"], expected_delta)
    assert np.isclose(r["delta_1"], expected_delta_1)
    assert np.isclose(r["delta_2"], expected_delta_2)
    assert np.isclose(r["H12"], expected_H12)


@pytest.mark.parametrize("use_pint", [False, True])
def test_deltas_tur_ic(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # from Problem 7.1 (Hirschel)
    x = 1
    Re = 1e06

    expected_delta = 0.023345421745767148
    expected_delta_vs = 0.00011568994376284626
    expected_delta_sc = 0.0005353769204133639
    expected_delta_1 = 0.0029213325049432947
    expected_delta_2 = 0.002271446440128695
    expected_H12 = 1.2861111111111112

    if use_pint:
        x *= m
        expected_delta *= m
        expected_delta_1 *= m
        expected_delta_2 *= m
        expected_delta_vs *= m
        expected_delta_sc *= m

    r = deltas_tur_ic(x, Re, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 6
    assert hasattr(r, "show")
    r = deltas_tur_ic(x, Re, to_dict=True)
    assert hasattr(r, "show")
    assert isinstance(r, dict) and len(r) == 6
    assert np.isclose(r["delta"], expected_delta)
    assert np.isclose(r["delta_vs"], expected_delta_vs)
    assert np.isclose(r["delta_sc"], expected_delta_sc)
    assert np.isclose(r["delta_1"], expected_delta_1)
    assert np.isclose(r["delta_2"], expected_delta_2)
    assert np.isclose(r["H12"], expected_H12)


@pytest.mark.parametrize("use_pint", [False, True])
def test_deltas_lam_c(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # TODO: find solved examples to verify the numerical values
    x = 1
    Re = 1e06

    expected_delta = 0.00581158
    expected_delta_1 = 0.00806313
    expected_delta_2 = 0.00064325
    expected_H12 = 12.53507315

    if use_pint:
        x *= m
        expected_delta *= m
        expected_delta_1 *= m
        expected_delta_2 *= m

    r = deltas_lam_c(x, Re, 2.1, 1.2, 6, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 4
    assert hasattr(r, "show")
    r = deltas_lam_c(x, Re, 2.1, 1.2, 6, to_dict=True)
    assert isinstance(r, dict) and len(r) == 4
    assert hasattr(r, "show")
    assert np.isclose(r["delta"], expected_delta)
    assert np.isclose(r["delta_1"], expected_delta_1)
    assert np.isclose(r["delta_2"], expected_delta_2)
    assert np.isclose(r["H12"], expected_H12)


@pytest.mark.parametrize("use_pint", [False, True])
def test_deltas_tur_c(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # TODO: find solved examples to verify the numerical values
    x = 1
    Re = 1e06

    expected_delta = 0.02479314
    expected_delta_vs = 0.00015166
    expected_delta_sc = 0.00068105
    expected_delta_1 = 0.01864145
    expected_delta_2 = 0.00201025
    expected_H12 = 9.27318000

    if use_pint:
        x *= m
        expected_delta *= m
        expected_delta_1 *= m
        expected_delta_2 *= m
        expected_delta_vs *= m
        expected_delta_sc *= m

    r = deltas_tur_c(x, Re, 2.1, 1.2, 6, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 6
    assert hasattr(r, "show")
    r = deltas_tur_c(x, Re, 2.1, 1.2, 6, to_dict=True)
    assert isinstance(r, dict) and len(r) == 6
    assert hasattr(r, "show")
    assert np.isclose(r["delta"], expected_delta)
    assert np.isclose(r["delta_vs"], expected_delta_vs)
    assert np.isclose(r["delta_sc"], expected_delta_sc)
    assert np.isclose(r["delta_1"], expected_delta_1)
    assert np.isclose(r["delta_2"], expected_delta_2)
    assert np.isclose(r["H12"], expected_H12)
