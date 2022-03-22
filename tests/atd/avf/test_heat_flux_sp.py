import numpy as np
from pygasflow.atd.avf.heat_flux_sp import (
    heat_flux_fay_riddell, heat_flux_scott, heat_flux_detra
)

def test_heat_flux_scott():
    # from Exercise 5.4, Hypersonic Aerothermodynamics
    R = 0.3
    u_inf = 4000
    rho_inf = 0.0019662686791414754
    assert np.isclose(
        heat_flux_scott(R, u_inf, rho_inf),
        90.5721895119818)


def test_heat_flux_detra():
    # from Exercise 5.4, Hypersonic Aerothermodynamics
    R = 0.3
    u_inf = 4000
    rho_inf = 0.0019662686791414754
    u_co = 7950
    rho_sl = 1.225000018124288
    assert np.isclose(
        heat_flux_detra(R, u_inf, rho_inf, u_co, rho_sl),
        92.70449353689641)

def test_heat_flux_fay_riddell():
    # from Exercise 5.2, Hypersonic Aerothermodynamics
    u_grad = 12871.532867838932
    Pr = 0.7368421052631579
    rho_w = 0.006502187739466109
    mu_w = 5.0335415297500225e-05
    rho_e = 0.00033636578171017867
    mu_e = 0.00023792119826279134
    ht2 = 26955615.929438587
    hw = 1394445.56
    assert np.isclose(
        heat_flux_fay_riddell(u_grad, Pr, rho_w, mu_w, rho_e, mu_e, ht2, hw, sphere=True),
        865539.7987264636)
