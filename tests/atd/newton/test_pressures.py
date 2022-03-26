import numpy as np
from pygasflow.atd.newton.pressures import (
    pressure_coefficient, shadow_region, modified_newtonian_pressure_ratio,
    pressure_coefficient_tangent_cone,
    pressure_coefficient_tangent_wedge
)
from pygasflow.shockwave import rayleigh_pitot_formula


def test_pressure_coefficient_newton():
    assert np.isclose(pressure_coefficient(0), 0)
    assert np.isclose(pressure_coefficient(np.pi / 2), 2)


def test_pressure_coefficient_newton_modified():
    assert np.isclose(pressure_coefficient(0, Mfs=20, gamma=1.4), 0)
    assert np.isclose(
        pressure_coefficient(np.pi / 2, Mfs=20, gamma=1.4),
        1.83744294512549)


def test_pressure_coefficient_newton_axisymmetric():
    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(np.deg2rad(15), alpha=np.deg2rad(10), beta=beta),
        np.array([0.35721239, 0.27899092, 0.12993477, 0.03714616, 0.01519225])
    )


def test_pressure_coefficient_newton_modified_axisymmetric():
    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(np.deg2rad(15), alpha=np.deg2rad(10),
            beta=beta, Mfs=14.9, gamma=5.0/3.0),
        np.array([0.31427852, 0.2454586 , 0.11431772, 0.03268151, 0.01336627])
    )


def test_shadow_region():
    alpha = np.deg2rad(35)
    beta = np.deg2rad(0)
    theta_c = np.deg2rad(9)
    phi_i, phi_f, func = shadow_region(alpha, theta_c, beta)
    assert np.isclose(phi_i, 1.342625208348352)
    assert np.isclose(phi_f, 4.940560098831234)
    assert np.isclose(func(alpha, theta_c, beta, phi_i), 0)
    assert np.isclose(func(alpha, theta_c, beta, phi_f), 0)


def test_modified_newtonian_pressure_ratio():
    Mfs = 10
    g = 1.4
    assert np.isclose(
        modified_newtonian_pressure_ratio(Mfs, 0, gamma=g),
        1 / rayleigh_pitot_formula(Mfs, gamma=g))

    theta_b = np.deg2rad([90, 60, 30, 0])
    assert np.allclose(
        modified_newtonian_pressure_ratio(Mfs, theta_b, gamma=g),
        np.array([1.        , 0.75193473, 0.25580419, 0.00773892]))

    assert np.allclose(
        modified_newtonian_pressure_ratio(10, theta_b, alpha=np.deg2rad(33)),
        np.array([0.70566393, 0.99728214, 0.79548767, 0.30207499]))


def test_pressure_coefficient_tangent_cone():
    theta_c = np.deg2rad(10)
    assert np.isclose(
        pressure_coefficient_tangent_cone(theta_c, 1.4),
        0.06344098329442194)

    # as gamma -> 1, the expression reduces to the Newtonian value
    assert np.isclose(
        pressure_coefficient_tangent_cone(theta_c, 1+1e-06) / theta_c**2,
        2)


def test_pressure_coefficient_tangent_wedge():
    theta_c = np.deg2rad(10)
    assert np.isclose(
        pressure_coefficient_tangent_wedge(theta_c, 1.4),
        0.07310818074881005)

    # as gamma -> 1, the expression reduces to the Newtonian value
    assert np.isclose(
        pressure_coefficient_tangent_wedge(theta_c, 1+1e-06) / theta_c**2,
        2)
