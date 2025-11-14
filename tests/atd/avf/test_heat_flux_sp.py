import numpy as np
from pygasflow.atd.avf.heat_flux_sp import (
    heat_flux_fay_riddell,
    heat_flux_scott,
    heat_flux_detra,
    heat_flux_radiation_martin
)
from pygasflow import canonicalize_pint_dimensions
import pytest


@pytest.mark.parametrize("use_pint", [False, True])
def test_heat_flux_scott(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # from Exercise 5.4, Hypersonic Aerothermodynamics
    R = 0.3
    u_inf = 4000
    rho_inf = 0.0019662686791414754
    expected_q_dot = 90.5721895119818

    if use_pint:
        R *= m
        u_inf *= m / s
        rho_inf *= kg / m**3
        cm = m / 100
        expected_q_dot *= W / cm**2

    q_dot = heat_flux_scott(R, u_inf, rho_inf)

    assert np.isclose(q_dot, expected_q_dot)


@pytest.mark.parametrize("use_pint", [False, True])
def test_heat_flux_detra(use_pint, setup_pint_registry, ureg):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    # from Exercise 5.4, Hypersonic Aerothermodynamics

    # metric values
    R = 0.3
    u_inf = 4000
    rho_inf = 0.0019662686791414754
    u_co = 7950
    rho_sl = 1.225000018124288
    expected_q_dot_metric = 92.74510742790949

    # imperial values
    R2 = 0.984251968503937
    rho_inf2 = 3.815191096273104e-06
    rho_sl2 = 0.0023768924418420753
    u_inf2 = 13123.359580052494
    u_co2 = 26082.677165354333
    expected_q_dot_imperial = 81.6667781158476

    if use_pint:
        R *= m
        u_inf *= m / s
        u_co *= m / s
        rho_inf *= kg / m**3
        rho_sl *= kg / m**3
        cm = m / 100
        expected_q_dot_metric *= W / cm**2

        R2 *= ureg.ft
        u_inf2 *= ureg.ft / s
        u_co2 *= ureg.ft / s
        rho_inf2 *= ureg.lbf * ureg.s**2 / ureg.ft**4
        rho_sl2 *= ureg.lbf * ureg.s**2 / ureg.ft**4
        expected_q_dot_imperial *= ureg.Btu / ureg.ft**2 / ureg.s

    q_dot_metric = heat_flux_detra(R, u_inf, rho_inf, u_co, rho_sl)
    q_dot_imperial = heat_flux_detra(
        R2, u_inf2, rho_inf2, u_co2, rho_sl2, metric=False)
    assert np.isclose(q_dot_metric, expected_q_dot_metric)
    assert np.isclose(q_dot_imperial, expected_q_dot_imperial)
    if use_pint:
        assert np.isclose(
            expected_q_dot_imperial.to("W / cm**2"), q_dot_metric)


@pytest.mark.parametrize("use_pint", [False, True])
def test_heat_flux_fay_riddell(use_pint, setup_pint_imperial_registry):
    R, Btu, lbf, lbm, ft, s, Q_ = setup_pint_imperial_registry

    # from Exercise 5.2, Hypersonic Aerothermodynamics
    u_grad = 12871.540335275073
    Pr = 0.7368421052631579
    rho_w = 1.2611943627968788e-05
    mu_w = 1.0512765233552152e-06
    rho_e = 6.525428485981234e-07
    mu_e = 4.9686546490717815e-06
    h_t2 = 11586.824574050748
    h_w = 599.5031167908519
    expected_q_dot = 2.368078016743907

    if use_pint:
        u_grad *= 1 / s
        rho_w *= lbf * s**2 / ft**4
        rho_e *= lbf * s**2 / ft**4
        mu_w *= lbf * s / ft**2
        mu_e *= lbf * s / ft**2
        h_t2 *= Btu / lbm
        h_w *= Btu / lbm
        expected_q_dot *= Btu * lbf * s / (ft**3 * lbm)

    q_dot = heat_flux_fay_riddell(u_grad, Pr, rho_w, mu_w, rho_e, mu_e, h_t2, h_w, sphere=True)
    q_dot = canonicalize_pint_dimensions(q_dot)
    assert np.isclose(q_dot, expected_q_dot)


@pytest.mark.parametrize("use_pint", [False, True])
def test_heat_flux_radiation_martin(use_pint, ureg):
    kg, m, cm, s, ft, lbf = ureg.kg, ureg.m, ureg.cm, ureg.s, ureg.ft, ureg.lbf
    W, Btu = ureg.W, ureg.Btu

    # These values come from Problem 5.4 of "Hypersonic Aerothermodynamics"

    rho_inf_imp = np.array([
        1.02108992e-07, 1.58392674e-07, 6.45765510e-08, 2.40761506e-07,
        5.32793906e-07, 3.45574826e-06])
    rho_sl_imp = 0.0023768924418420753
    u_inf_imp = np.array([36000., 32000., 27000., 26000., 16000.,  8000.])
    Rn_imp = 2
    expected_q_dot_imp = np.array([
        1.10277875e+00, 8.18023363e-01, 4.59331783e-02, 2.73670186e-01,
        1.57372701e-02, 8.65659289e-04])

    rho_inf_metric = np.array([
        5.26248119e-05, 8.16322290e-05, 3.32813865e-05, 1.24083380e-04,
        2.74590694e-04, 1.78101945e-03])
    rho_sl_metric = 1.2250000181242882
    u_inf_metric = np.array([
        10972.8,  9753.6,  8229.6,  7924.8,  4876.8,  2438.4])
    Rn_metric = 0.6096
    expected_q_dot_metric = np.array([
        1.25237381e+00, 9.28990545e-01, 5.21641437e-02, 3.10794320e-01,
        1.78720753e-02, 9.83088419e-04])

    if use_pint:
        Rn_imp *= ft
        u_inf_imp *= ft / s
        rho_inf_imp *= lbf * s**2 / ft**4
        rho_sl_imp *= lbf * s**2 / ft**4
        expected_q_dot_imp *= Btu / ft**2 / s

        Rn_metric *= m
        u_inf_metric *= m / s
        rho_inf_metric *= kg / m**3
        rho_sl_metric *= kg / m**3
        expected_q_dot_metric *= W / cm**2

    q_dot_imp = heat_flux_radiation_martin(
        Rn_imp, u_inf_imp, rho_inf_imp, rho_sl_imp, metric=False)
    q_dot_metric = heat_flux_radiation_martin(
        Rn_metric, u_inf_metric, rho_inf_metric, rho_sl_metric, metric=True)
    assert np.allclose(q_dot_imp, expected_q_dot_imp)
    assert np.allclose(q_dot_metric, expected_q_dot_metric)
    if use_pint:
        assert np.allclose(q_dot_metric, q_dot_imp.to("W / cm**2"))
