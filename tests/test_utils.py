
import pint
import pytest
import pygasflow
import numpy as np
from pygasflow.utils.common import canonicalize_pint_dimensions
from pygasflow.atd.avf.heat_flux_sp import heat_flux_fay_riddell

def test_canonicalize_pint_dimensions():
    ureg = pint.UnitRegistry()
    ureg.formatter.default_format = "~"
    ureg.define("pound_mass = 0.45359237 kg = lbm")
    pygasflow.defaults.pint_ureg = ureg
    lbf, lbm, Btu, ft, s = ureg.lbf, ureg.lbm, ureg.Btu, ureg.ft, ureg.s
    Pr = 0.7368421052631579
    u_grad = 12871.540335275073 * 1 / s
    rho_w = 1.2611943627968788e-05 * lbf * s ** 2 / ft ** 4
    rho_e = 6.525428485981234e-07 * lbf * s ** 2 / ft ** 4
    mu_w = 1.0512765233552152e-06 * lbf * s / ft ** 2
    mu_e = 4.9686546490717815e-06 * lbf * s / ft ** 2
    h_t2 = 11586.824574050748 * Btu / lbm
    h_w = 599.5031167908519 * Btu / lbm
    q = heat_flux_fay_riddell(u_grad, Pr, rho_w, mu_w, rho_e, mu_e, h_t2, h_w, sphere=True)
    assert np.isclose(q.magnitude, 2.368078016743907)
    # conversion fails
    pytest.raises(
        pint.DimensionalityError,
        lambda: q.to("Btu / ft**2 / s")
    )

    unit_dict1 = dict(q.unit_items())

    new_q = canonicalize_pint_dimensions(q)
    assert np.isclose(q.magnitude, new_q.magnitude)

    unit_dict2 = dict(new_q.unit_items())

    assert len(unit_dict1) == len(unit_dict2)
    assert len(set(unit_dict1.keys()).difference(unit_dict2.keys())) == 0
    diff = {k: abs(unit_dict1[k] - unit_dict2[k]) for k in unit_dict1}
    assert any(v > 0 for v in diff.values())
    assert any(v == 0 for v in diff.values())

    # conversion succeeds
    new_q = new_q.to("Btu / ft**2 / s")
    assert np.isclose(new_q.magnitude, 76.19065709613399)
