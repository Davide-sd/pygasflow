import numpy as np
from pygasflow import (
    isentropic_solver,
    normal_shockwave_solver,
    sound_speed,
)
from pygasflow.atd.avf.wall_shear_stress_sp import (
    velocity_gradient,
)
import pint
import pytest
import pygasflow


@pytest.mark.parametrize("use_pint", [False, True])
def test_velocity_gradient(use_pint):
    # From Exercise 5.2, Hypersonic Aerothermodynamics, J. Bertin
    ureg = pint.UnitRegistry()
    ureg.define("pound_mass = 0.45359237 kg = lbm")
    pygasflow.defaults.pint_ureg = ureg

    R = (287.05 * ureg.J / (ureg.kg * ureg.K)).to("Btu / lbm / degR").magnitude
    r = 1
    u1 = 24e03
    T1 = 381.61885288502907
    p1 = 0.0668887801071935
    rho1 = 3.2852596810182865
    H = 240e03
    Tw = 250
    gamma = 1.4

    expected_u_grad = 12871.540335275073

    if use_pint:
        R = ureg.Quantity(R, "Btu / lbm / degR")
        r *= ureg.feet
        u1 *= ureg.feet / ureg.s
        H *= ureg.feet
        Tw *= ureg.degR
        T1 *= ureg.degR
        p1 *= ureg.lbf / ureg.feet**2
        rho1 *= ureg.lbm / ureg.feet**3
        expected_u_grad /= ureg.s

    a1 = sound_speed(gamma, R, T1)
    if use_pint:
        a1 = a1.to("feet / s")
    else:
        a1 = ureg.Quantity(a1, "Btu**0.5 / lbm**0.5").to("feet / s").magnitude
    M1 = u1 / a1
    res1 = isentropic_solver("m", M1, gamma=gamma, to_dict=True)
    Tt1 = (1 / res1["tr"]) * T1
    pt1 = (1 / res1["pr"]) * p1
    shock = normal_shockwave_solver("mu", M1, gamma=gamma, to_dict=True)
    # stagnation quantities
    pt2 = shock["tpr"] * pt1
    Tt2 = Tt1
    rhot2 = (pt2 / (R * Tt2))
    if use_pint:
        rhot2 = rhot2.to("lbf * s**2 / feet**4")
    else:
        rhot2 = ureg.Quantity(rhot2, "lbf * lbm / Btu / ft**2")
        rhot2 = rhot2.to("lbf * s**2 / feet**4").magnitude

    u_grad = velocity_gradient(r, p1, pt2, rhot2, k=1)
    assert np.isclose(u_grad, expected_u_grad)
