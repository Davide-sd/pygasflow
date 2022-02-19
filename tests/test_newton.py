import numpy as np
from pygasflow.newton import pressure_coefficient

def test_pressure_coefficient_newton():
    assert np.isclose(pressure_coefficient(0), 0)
    assert np.isclose(pressure_coefficient(np.pi / 2), 2)

def test_pressure_coefficient_newton_modified():
    assert np.isclose(pressure_coefficient(0, Mfs=20), 0)
    assert np.isclose(pressure_coefficient(np.pi / 2, Mfs=20), 1.83744294512549)

def test_pressure_coefficient_newton_axisymmetric():
    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(np.deg2rad(10), beta=beta, theta=np.deg2rad(15)),
        np.array([0.35721239, 0.27899092, 0.12993477, 0.03714616, 0.01519225])
    )

def test_pressure_coefficient_newton_modified_axisymmetric():
    beta = np.linspace(0, np.pi, 5)
    assert np.allclose(
        pressure_coefficient(np.deg2rad(10), beta=beta, theta=np.deg2rad(15), Mfs=14.9, gamma=5.0/3.0),
        np.array([0.31427852, 0.2454586 , 0.11431772, 0.03268151, 0.01336627])
    )
