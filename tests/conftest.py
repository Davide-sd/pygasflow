import pint
import pygasflow
import pytest


@pytest.fixture
def setup_pint_registry():
    ureg = pint.UnitRegistry()
    pygasflow.defaults.pint_ureg = ureg
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
