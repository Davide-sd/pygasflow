import numpy as np
from pygasflow.atd.avf.thickness_fp import (
    deltas_lam_ic, deltas_tur_ic, deltas_lam_c, deltas_tur_c
)


def test_deltas_lam_ic():
    x = 1
    Re = 1e06
    r = deltas_lam_ic(x, Re, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 4
    r = deltas_lam_ic(x, Re, to_dict=True)
    assert isinstance(r, dict) and len(r) == 4
    assert np.isclose(r["delta"], 0.005)
    assert np.isclose(r["delta_1"], 0.0017208000000000002)
    assert np.isclose(r["delta_2"], 0.0006641)
    assert np.isclose(r["H12"], 2.5911760277066707)


def test_deltas_tur_ic():
    x = 1
    Re = 1e06
    r = deltas_tur_ic(x, Re, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 6
    r = deltas_tur_ic(x, Re, to_dict=True)
    assert isinstance(r, dict) and len(r) == 6
    assert np.isclose(r["delta"], 0.023345421745767148)
    assert np.isclose(r["delta_vs"], 0.00011568994376284626)
    assert np.isclose(r["delta_sc"], 0.0005353769204133639)
    assert np.isclose(r["delta_1"], 0.0029213325049432947)
    assert np.isclose(r["delta_2"], 0.002271446440128695)
    assert np.isclose(r["H12"], 1.2861111111111112)


def test_deltas_lam_c():
    x = 1
    Re = 1e06
    r = deltas_lam_c(x, Re, 2.1, 1.2, 6, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 4
    r = deltas_lam_c(x, Re, 2.1, 1.2, 6, to_dict=True)
    assert isinstance(r, dict) and len(r) == 4


def test_deltas_tur_c():
    x = 1
    Re = 1e06
    r = deltas_tur_c(x, Re, 2.1, 1.2, 6, to_dict=False)
    assert isinstance(r, (tuple, list)) and len(r) == 6
    r = deltas_tur_c(x, Re, 2.1, 1.2, 6, to_dict=True)
    assert isinstance(r, dict) and len(r) == 6
