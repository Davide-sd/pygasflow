import pint
import pygasflow
import pytest


@pytest.fixture
def ureg():
    ureg = pint.UnitRegistry()
    ureg.define("pound_mass = 0.45359237 kg = lbm")
    pygasflow.defaults.pint_ureg = ureg
    return ureg


@pytest.fixture
def setup_pint_registry(ureg):
    K = ureg.K
    m = ureg.m
    s = ureg.s
    kg = ureg.kg
    J = ureg.J
    W = ureg.W
    Q_ = ureg.Quantity
    kmol = ureg.kmol
    atm = ureg.atm
    return K, m, s, kg, J, W, Q_, kmol, atm


@pytest.fixture
def setup_pint_imperial_registry(ureg):
    R = ureg.degR
    Btu = ureg.Btu
    lbf = ureg.lbf
    lbm = ureg.lbm
    ft = ureg.feet
    s = ureg.s
    Q_ = ureg.Quantity
    return R, Btu, lbf, lbm, ft, s, Q_
