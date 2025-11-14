import numpy as np
from pygasflow.atd.temperatures import reference_temperature
from pygasflow.atd.avf.wall_shear_stress_fp import *
from pygasflow.atd.viscosity import viscosity_air_power_law
import pytest


# NOTE: here I test specifically for wss_lam_c, wss_tur_c. This indirectly
# tests also for wss_lam_ic, wss_tur_ic.


def setup(use_pint, units):
    K, m, s, kg, J, W, Q_, kmol, atm = units

    # the following values comes from Problem 7.05, 7.06 (Hirschel)
    gamma = 1.4
    Minf = 6
    Lref = 80
    Tinf = 226.50908361133006
    rhoinf = 0.01841010086243616
    Prs = 0.74  # reference Prandtl number
    Tw1 = 1000
    Tw2 = 2000
    R = 287

    if use_pint:
        Lref *= m
        Tinf *= K
        rhoinf *= kg / m**3
        Tw1 *= K
        Tw2 *= K
        R *= J / kg / K

    muinf = viscosity_air_power_law(Tinf)
    ainf = np.sqrt(gamma * R * Tinf)
    uinf = Minf * ainf
    Reinf_L = rhoinf * uinf / muinf * Lref

    rs_lam = np.sqrt(Prs)
    rs_tur = np.cbrt(Prs)
    Me, Te = Minf, Tinf

    Ts1 = reference_temperature(Te, Tw1, rs=rs_lam, Me=Me, gamma_e=1.4)
    Ts2 = reference_temperature(Te, Tw1, rs=rs_tur, Me=Me, gamma_e=1.4)

    Ts3 = reference_temperature(Te, Tw2, rs=rs_lam, Me=Me, gamma_e=1.4)
    Ts4 = reference_temperature(Te, Tw2, rs=rs_tur, Me=Me, gamma_e=1.4)

    return rhoinf, uinf, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4


@pytest.mark.parametrize("use_pint", [False, True])
def test_reference_temperature(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    _, _, _, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)

    expected_Ts1 = 921.8977042109084
    expected_Ts2 = 937.7819441806704
    expected_Ts3 = 1421.8977042109084
    expected_Ts4 = 1437.7819441806705

    if use_pint:
        expected_Ts1 *= K
        expected_Ts2 *= K
        expected_Ts3 *= K
        expected_Ts4 *= K

    assert np.isclose(Ts1, expected_Ts1)
    assert np.isclose(Ts2, expected_Ts2)
    assert np.isclose(Ts3, expected_Ts3)
    assert np.isclose(Ts4, expected_Ts4)


@pytest.mark.parametrize("use_pint", [False, True])
def test_friction_drag_lam_c(use_pint, setup_pint_registry):
    _, _, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    assert np.isclose(
        friction_drag_lam_c(Reinf_L, Ts1 / Tinf),
        7.988126288687875e-05)
    assert np.isclose(
        friction_drag_tur_c(Reinf_L, Ts2 / Tinf),
        0.0006459536745846517)
    assert np.isclose(
        friction_drag_lam_c(Reinf_L, Ts3 / Tinf),
        7.404784755397658e-05)
    assert np.isclose(
        friction_drag_tur_c(Reinf_L, Ts4 / Tinf),
        0.0004851268331035941)


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_lam_c(use_pint, setup_pint_registry):
    rhoinf, uinf, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    qinf = 0.5 * rhoinf * uinf**2

    cf = wss_lam_c(rhoinf, uinf, Reinf_L, Ts1 / Tinf)
    CDf = friction_drag_lam_c(Reinf_L, Ts1 / Tinf)
    assert np.isclose(cf / CDf, 0.5 * qinf)

    cf = wss_lam_c(rhoinf, uinf, Reinf_L, Ts2 / Tinf)
    CDf = friction_drag_lam_c(Reinf_L, Ts2 / Tinf)
    assert np.isclose(cf / CDf, 0.5 * qinf)


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_tur_c(use_pint, setup_pint_registry):
    rhoinf, uinf, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    qinf = 0.5 * rhoinf * uinf**2

    cf = wss_tur_c(rhoinf, uinf, Reinf_L, Ts3 / Tinf)
    CDf = friction_drag_tur_c(Reinf_L, Ts3 / Tinf)
    assert np.isclose(cf / CDf, 0.8 * qinf)

    cf = wss_tur_c(rhoinf, uinf, Reinf_L, Ts4 / Tinf)
    CDf = friction_drag_tur_c(Reinf_L, Ts4 / Tinf)
    assert np.isclose(cf / CDf, 0.8 * qinf)


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_lam_ic(use_pint, setup_pint_registry):
    rhoinf, uinf, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    qinf = 0.5 * rhoinf * uinf**2

    cf = wss_lam_ic(rhoinf, uinf, Reinf_L)
    CDf = friction_drag_lam_ic(Reinf_L)
    assert np.isclose(cf / CDf, 0.5 * qinf)


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_tur_ic(use_pint, setup_pint_registry):
    rhoinf, uinf, Reinf_L, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    qinf = 0.5 * rhoinf * uinf**2

    cf = wss_tur_ic(rhoinf, uinf, Reinf_L)
    CDf = friction_drag_tur_ic(Reinf_L)
    assert np.isclose(cf / CDf, 0.8 * qinf)


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_ic(use_pint, setup_pint_registry):
    rho, u, Re, _, _, _, _, _ = setup(
        use_pint, setup_pint_registry)
    res = wss_ic(rho, u, Re, laminar=True, to_dict=False)
    assert isinstance(res, (tuple, list)) and (len(res) == 3)
    assert hasattr(res, "show")
    res = wss_ic(rho, u, Re, laminar=True, to_dict=True)
    assert isinstance(res, dict) and (len(res) == 3)
    assert hasattr(res, "show")
    res = wss_ic(rho, u, Re, laminar=False, to_dict=False)
    assert isinstance(res, (tuple, list)) and (len(res) == 3)
    assert hasattr(res, "show")
    res = wss_ic(rho, u, Re, laminar=False, to_dict=True)
    assert isinstance(res, dict) and (len(res) == 3)
    assert hasattr(res, "show")


@pytest.mark.parametrize("use_pint", [False, True])
def test_wss_c(use_pint, setup_pint_registry):
    rhoinf, uinf, Reinf, Tinf, Ts1, Ts2, Ts3, Ts4 = setup(
        use_pint, setup_pint_registry)
    res = wss_c(rhoinf, uinf, Reinf, Ts1 / Tinf, laminar=True, to_dict=False)
    assert isinstance(res, (tuple, list)) and (len(res) == 3)
    assert hasattr(res, "show")
    res = wss_c(rhoinf, uinf, Reinf, Ts1 / Tinf, laminar=True, to_dict=True)
    assert isinstance(res, dict) and (len(res) == 3)
    assert hasattr(res, "show")
    assert np.isclose(res["CDf"], 7.988126288687875e-05)
    res = wss_c(rhoinf, uinf, Reinf, Ts1 / Tinf, laminar=False, to_dict=False)
    assert isinstance(res, (tuple, list)) and (len(res) == 3)
    assert hasattr(res, "show")
    res = wss_c(rhoinf, uinf, Reinf, Ts4 / Tinf, laminar=False, to_dict=True)
    assert isinstance(res, dict) and (len(res) == 3)
    assert hasattr(res, "show")
    assert np.isclose(res["CDf"], 0.0004851268331035941)
