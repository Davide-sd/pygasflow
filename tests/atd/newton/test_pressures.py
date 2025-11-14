import numpy as np
from pygasflow.atd.newton.pressures import (
    pressure_coefficient, shadow_region, modified_newtonian_pressure_ratio,
    pressure_coefficient_tangent_cone,
    pressure_coefficient_tangent_wedge
)
from pygasflow.shockwave import rayleigh_pitot_formula
import pytest


def _prepare_angles(angles, ureg):
    return [
        ureg.Quantity(*a) if isinstance(a, tuple) else a
        for a in angles
    ]


@pytest.mark.parametrize("theta_b, cp", [
    (0, 0),
    ((0, "deg"), 0),
    ((0, "rad"), 0),
    (np.pi / 2, 2),
    ((90, "deg"), 2),
    ((np.pi / 2, "rad"), 2),
])
def test_pressure_coefficient_newton(theta_b, cp, ureg):
    theta_b, = _prepare_angles([theta_b], ureg)
    assert np.isclose(pressure_coefficient(theta_b), cp)


@pytest.mark.parametrize("theta_b, Mfs, gamma, cp", [
    (0, 20, 1.4, 0),
    ((0, "deg"), 20, 1.4, 0),
    ((0, "rad"), 20, 1.4, 0),
    (np.pi / 2, 20, 1.4, 1.83744294512549),
    ((90, "deg"), 20, 1.4, 1.83744294512549),
    ((np.pi / 2, "rad"), 20, 1.4, 1.83744294512549),
])
def test_pressure_coefficient_newton_modified(theta_b, Mfs, gamma, cp, ureg):
    theta_b, = _prepare_angles([theta_b], ureg)
    assert np.isclose(
        pressure_coefficient(theta_b, Mfs=Mfs, gamma=gamma),
        cp
    )


@pytest.mark.parametrize("theta_b, alpha", [
    (np.deg2rad(15), np.deg2rad(10)),
    ((15, "deg"), (10, "deg"))
])
def test_pressure_coefficient_newton_axisymmetric(theta_b, alpha, ureg):
    theta_b, alpha = _prepare_angles([theta_b, alpha], ureg)

    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(theta_b, alpha=alpha, beta=beta),
        np.array([0.35721239, 0.27899092, 0.12993477, 0.03714616, 0.01519225])
    )


@pytest.mark.parametrize("theta_b, alpha", [
    (np.deg2rad(15), np.deg2rad(10)),
    ((15, "deg"), (10, "deg"))
])
def test_pressure_coefficient_newton_modified_axisymmetric(theta_b, alpha, ureg):
    theta_b, alpha = _prepare_angles([theta_b, alpha], ureg)

    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(theta_b, alpha=alpha,
            beta=beta, Mfs=14.9, gamma=5.0/3.0),
        np.array([0.31427852, 0.2454586 , 0.11431772, 0.03268151, 0.01336627])
    )


@pytest.mark.parametrize("alpha, beta, theta_c", [
    (np.deg2rad(35), np.deg2rad(0), np.deg2rad(9)),
    ((35, "deg"), (0, "deg"), (9, "deg"))
])
def test_shadow_region(alpha, beta, theta_c, ureg):
    alpha, beta, theta_c = _prepare_angles([alpha, beta, theta_c], ureg)

    phi_i, phi_f, func = shadow_region(alpha, theta_c, beta)
    assert np.isclose(phi_i, 1.342625208348352 * ureg.radian)
    assert np.isclose(phi_f, 4.940560098831234 * ureg.radian)
    assert np.isclose(func(alpha, theta_c, beta, phi_i), 0)
    assert np.isclose(func(alpha, theta_c, beta, phi_f), 0)


def test_modified_newtonian_pressure_ratio(ureg):
    Mfs = 10
    g = 1.4
    assert np.isclose(
        modified_newtonian_pressure_ratio(Mfs, 0, gamma=g),
        1 / rayleigh_pitot_formula(Mfs, gamma=g))

    theta_b = np.deg2rad([90, 60, 30, 0])
    assert np.allclose(
        modified_newtonian_pressure_ratio(Mfs, theta_b, gamma=g),
        np.array([1.        , 0.75193473, 0.25580419, 0.00773892]))

    theta_b = np.array([90, 60, 30, 0]) * ureg.deg
    assert np.allclose(
        modified_newtonian_pressure_ratio(Mfs, theta_b, gamma=g),
        np.array([1.        , 0.75193473, 0.25580419, 0.00773892]))

    alpha = np.deg2rad(33)
    assert np.allclose(
        modified_newtonian_pressure_ratio(10, theta_b, alpha=alpha),
        np.array([0.70566393, 0.99728214, 0.79548767, 0.30207499]))

    alpha = 33 * ureg.deg
    assert np.allclose(
        modified_newtonian_pressure_ratio(10, theta_b, alpha=alpha),
        np.array([0.70566393, 0.99728214, 0.79548767, 0.30207499]))


@pytest.mark.parametrize("theta_c", [
    np.deg2rad(10),
    (10, "deg"),
])
def test_pressure_coefficient_tangent_cone(theta_c, ureg):
    theta_c, = _prepare_angles([theta_c], ureg)

    assert np.isclose(
        pressure_coefficient_tangent_cone(theta_c, 1.4),
        0.06344098329442194)

    # as gamma -> 1, the expression reduces to the Newtonian value
    assert np.isclose(
        pressure_coefficient_tangent_cone(theta_c, 1+1e-06) / theta_c**2,
        2)


@pytest.mark.parametrize("theta_c", [
    np.deg2rad(10),
    (10, "deg"),
])
def test_pressure_coefficient_tangent_wedge(theta_c, ureg):
    theta_c, = _prepare_angles([theta_c], ureg)

    assert np.isclose(
        pressure_coefficient_tangent_wedge(theta_c, 1.4),
        0.07310818074881005)

    # as gamma -> 1, the expression reduces to the Newtonian value
    assert np.isclose(
        pressure_coefficient_tangent_wedge(theta_c, 1+1e-06) / theta_c**2,
        2)
