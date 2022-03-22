import numpy as np
from pygasflow.atd.avf.wall_shear_stress_sp import (
    velocity_gradient, 
)


def test_velocity_gradient():
    # From Exercise 5.2, Hypersonic Aerothermodynamics, J. Bertin
    R = 0.3048
    pinf = 3.2026521144111544
    pt2 = 2591.8463529570536
    rhot2 = 0.00033636578171017867
    assert np.isclose(
        velocity_gradient(R, pinf, pt2, rhot2, k=1), 
        12871.532867838932)

