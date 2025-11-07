import numpy as np
import cantera as ct
from pygasflow.atd import (
    Prandtl,
    Knudsen,
    Stanton,
    Strouhal,
    Reynolds,
    Peclet,
    Lewis,
    Eckert,
    Schmidt,
)
from pygasflow.atd.thermal_conductivity import thermal_conductivity_hansen
from pygasflow.atd.viscosity import viscosity_air_southerland
import pytest


@pytest.mark.parametrize("use_pint", [False, True])
def test_Prandtl_gamma(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    gamma_float = 1.4
    gamma_array = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    if use_pint:
        gamma_float = Q_(gamma_float)
        gamma_array = Q_(gamma_array)
    r1 = Prandtl(gamma_float)
    r2 = Prandtl(gamma_array)
    assert np.isclose(r1, 0.7368421052631579)
    assert np.allclose(
        r2,
        np.array([0.89795918, 0.82758621, 0.7761194, 0.73684211, 0.70588235]))

    if use_pint:
        assert r1.unitless and r2.unitless


@pytest.mark.parametrize("use_pint", [False, True])
def test_Prandtl_mu_cp_k(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    T = 350
    cp = 1004
    if use_pint:
        T *= K
        cp *= J / (kg * K)
    mu = viscosity_air_southerland(T)
    k = thermal_conductivity_hansen(T)
    assert np.isclose(Prandtl(mu, cp, k), 0.7370392202421769)


def test_Prandtl_cantera():
    air = ct.Solution("gri30.yaml")
    air.TPX = 350, ct.one_atm, {"N2": 0.79, "O2": 0.21}
    assert np.isclose(Prandtl(air), 0.7139365242266411)


@pytest.mark.parametrize(
    "Pe, Re, expected",
    [
        (7000, 10000, 0.7),
        (500, 1000, 0.5),
    ],
)
def test_prandtl_from_Pe_Re(Pe, Re, expected):
    assert np.isclose(Prandtl(Pe=Pe, Re=Re), expected)


@pytest.mark.parametrize(
    "Le, Sc, expected",
    [
        (1.0, 1.4, 1.4),
        (0.67, 1.0, 0.67),
        (1.5, 1.4, 2.0999999999999996),
    ],
)
def test_prandtl_from_Le_Sc(Le, Sc, expected):
    assert np.isclose(Prandtl(Le=Le, Sc=Sc), expected)


@pytest.mark.parametrize("use_pint, lambda_, L, expected", [
    (False, 0.01, 2, 0.005),
    (True, 0.01, 2, 0.005),
    (False, [0.01, 0.1], [2, 4], [0.005, 0.025]),
    (True, [0.01, 0.1], [2, 4], [0.005, 0.025]),
])
def test_Knudsen_lambda_L(use_pint, lambda_, L, expected, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry
    if use_pint:
        lambda_ *= m
        L *= m
    assert np.allclose(Knudsen(lambda_, L), expected)


@pytest.mark.parametrize("Minf, Rinf_L, gamma, expected", [
    (5, 20000, 1.4, 0.00037762789755305186),
])
def test_Knudsen_lambda_Minf_Reinf_gamma(Minf, Rinf_L, gamma, expected):
    assert np.allclose(Knudsen(Minf, Rinf_L, gamma), expected)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Stanton_q_gw_q_inf(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    q_gw = 919579342
    cp = 1004
    v_inf = 2000
    # at 10000m
    T_inf = 223
    rho_inf = 0.4135

    if use_pint:
        T_inf *= K
        cp *= J / (kg * K)
        v_inf *= m / s
        rho_inf *= kg / m**3
        q_gw *= J / (m**2 * s)

    h_inf = cp * T_inf
    h_t = h_inf + v_inf**2 / 2
    q_inf = rho_inf * v_inf * h_t

    assert np.isclose(Stanton(q_gw, q_inf), 0.5)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Stanton_q_gw_rho_inf_v_inf(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    q_gw = 919579342
    cp = 1004
    v_inf = 2000
    # at 10000m
    T_inf = 223
    rho_inf = 0.4135

    if use_pint:
        T_inf *= K
        cp *= J / (kg * K)
        v_inf *= m / s
        rho_inf *= kg / m**3
        q_gw *= J / (m**2 * s)

    assert np.isclose(Stanton(q_gw, rho_inf, v_inf), 0.555973)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Stanton_q_gw_rho_inf_v_inf(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    q_gw = 919579342
    cp = 1004
    v_inf = 2000
    # at 10000m
    T_inf = 223
    rho_inf = 0.4135

    if use_pint:
        T_inf *= K
        cp *= J / (kg * K)
        v_inf *= m / s
        rho_inf *= kg / m**3
        q_gw *= J / (m**2 * s)

    assert np.isclose(Stanton(q_gw, rho_inf, v_inf), 0.555973)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Stanton_q_gw_rho_inf_v_inf_delta_h(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    q_gw = 919579342
    cp = 1004
    v_inf = 2000
    # at 10000m
    T_inf = 223
    rho_inf = 0.4135
    delta_h = 5e06

    if use_pint:
        T_inf *= K
        cp *= J / (kg * K)
        v_inf *= m / s
        rho_inf *= kg / m**3
        q_gw *= J / (m**2 * s)
        delta_h *= J / kg

    assert np.isclose(Stanton(q_gw, rho_inf, v_inf, delta_h), 0.2223892)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Strouhal_t_res_t_ref(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    t_ref = 10
    t_res = 5
    if use_pint:
        t_ref *= s
        t_res *= s

    assert np.isclose(Strouhal(t_res, t_ref), 0.5)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Strouhal_L_ref_ref_v_ref(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    t_ref = 5
    L_ref = 10
    v_ref = 10
    if use_pint:
        t_ref *= s
        L_ref *= m
        v_ref *= m / s

    assert np.isclose(Strouhal(L_ref, t_ref, v_ref), 0.2)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Reynolds(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    rho = 0.8
    u = 200
    mu = 2e-05
    L1 = 1
    L2 = 50

    if use_pint:
        rho *= kg / m**3
        u *= m / s
        mu *= kg / (m * s)
        L1 *= m
        L2 *= m

    assert np.isclose(Reynolds(rho, u, mu, L1), 8000000.0)
    assert np.isclose(Reynolds(rho, u, mu, L2), 4e08)

    if use_pint:
        # raise error because rho, u, mu have dimensions,
        # but L (optional argument which is not provided) is dimensionless
        pytest.raises(
            ValueError,
            lambda: Reynolds(rho, u, mu)
        )


@pytest.mark.parametrize("use_pint", [False, True])
def test_Peclet(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    rho = 0.8
    cp = 1004
    u = 200
    L = 10
    k = 3e-02

    if use_pint:
        rho *= kg / m**3
        cp *= J / (kg * K)
        u *= m / s
        L *= m
        k *= W / (m * K)

    assert np.isclose(Peclet(rho, u, cp, L, k), 53546666.7)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Lewis(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    rho = 0.8
    cp = 1004
    D = 200
    k = 3e-02

    if use_pint:
        rho *= kg / m**3
        cp *= J / (kg * K)
        D *= m**2 / s
        k *= W / (m * K)

    assert np.isclose(Lewis(rho, D, cp, k), 5354666.67)



@pytest.mark.parametrize("use_pint", [False, True])
def test_Eckert(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    M = 2
    gamma = 1.4

    if use_pint:
        M = Q_(M)
        gamma = Q_(gamma)

    assert np.isclose(Eckert(M, gamma), 1.6)
    assert np.isclose(Eckert(2*M, gamma), 6.4)
    assert np.isclose(Eckert(M, 1.2), 0.8)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Schmidt_Pr_Le(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    Pr = 5
    Le = 2

    if use_pint:
        Pr = Q_(Pr)
        Le = Q_(Le)

    assert np.isclose(Schmidt(Pr, Le), 2.5)


@pytest.mark.parametrize("use_pint", [False, True])
def test_Schmidt_rho_mu_D(use_pint, setup_pint_registry):
    K, m, s, kg, J, W, Q_, kmol, atm = setup_pint_registry

    rho = 0.8
    mu = 2e-05
    D = 200

    if use_pint:
        rho *= kg / m**3
        mu *= kg / (m * s)
        D *= m**2 / s

    assert np.isclose(Schmidt(rho, mu, D), 1.25e-07)
