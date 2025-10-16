# NOTE:
#
# 1. The expected results comes from:
#    http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
#
# 2. Instead of testing every single function in the isentropic.py module,
# I only test the isentropic_solver function from the solvers.isentropic module,
# which is going to call (almost) every function implemented in the
# isentropic.py module. Hence I can easily compare the result with known values.

# TODO:
# 1. The web calculator linked above doesn't provide values for the critical
# velocity ratio, hence that parameter is currently tested against itself.
# Not ideal.

import numpy as np
from pygasflow.solvers.isentropic import (
    isentropic_solver as ise,
    print_isentropic_results,
)
import pytest


def test_input_mach():
    gamma = 1.4
    expected_res = [
        [2, 3],                     # M
        [0.12780452, 0.02722368],   # Pressure Ratio
        [0.23004814, 0.07622631],   # Density Ratio
        [0.55555555, 0.35714285],   # Temperature Ratio
        [0.24192491, 0.05153250],   # Critical Pressure Ratio
        [0.36288736, 0.12024251],   # Critical Density Ratio
        [0.66666666, 0.42857142],   # Critical Temperature Ratio
        [1.63299316, 1.96396101],   # Critical Velocity Ratio
        [1.68749999, 4.23456790],   # Critical Area Ratio
        [29.9999999, 19.4712206],   # Mach Angle
        [26.3797608, 49.7573467],   # Prandtl-Meyer Angle
    ]

    # single scalar mach
    M = 2
    r = ise("m", M, gamma)
    assert len(r) == 11
    for e in r:
        assert not isinstance(e, np.ndarray)
    assert np.allclose(r, [e[0] for e in expected_res])

    M = [2, 3]
    r = ise("m", M, gamma)
    assert len(r) == 11
    for e in r:
        assert isinstance(e, np.ndarray)
        assert len(e) == 2
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", 3),
    ("pressure", 0.02722368),
    ("temperature", 0.35714285),
    ("density", 0.07622631),
    ("crit_area_super", 4.23456790),
    ("prandtl_meyer", 49.7573467),
    ("mach_angle", 19.4712206),
])
def test_input_parameters_supersonic(param_name, value):
    gamma = 1.4
    expected_res = [
        3,          # Mach number
        0.02722368, # Pressure Ratio P/P0
        0.07622631, # Density Ratio rho/rho0
        0.35714285, # Temperature Ratio T/T0
        0.05153250, # Critical Pressure Ratio P/P*
        0.12024251, # Critical Density Ratio rho/rho*
        0.42857142, # Critical Temperature Ratio T/T*
        1.96396101, # Critical Velocity Ratio U/U*
        4.23456790, # Critical Area Ratio A/A*
        19.4712206, # Mach Angle
        49.7573467, # Prandtl-Meyer Angle
    ]

    r = ise(param_name, value, gamma)
    assert len(r) == 11
    for e in r:
        assert not isinstance(e, np.ndarray)
    assert np.allclose(r, expected_res, equal_nan=True)


@pytest.mark.parametrize("param_name, value", [
    ("m", 0.5),
    ("pressure", 0.84301917),
    ("temperature", 0.95238095),
    ("density", 0.88517013),
    ("crit_area_sub", 1.33984375),
])
def test_input_parameters_subsonic(param_name, value):
    gamma = 1.4
    expected_res = [
        0.5,        # Mach number
        0.84301917, # Pressure Ratio P/P0
        0.88517013, # Density Ratio rho/rho0
        0.95238095, # Temperature Ratio T/T0
        1.59577557, # Critical Pressure Ratio P/P*
        1.39630363, # Critical Density Ratio rho/rho*
        1.14285714, # Critical Temperature Ratio T/T*
        0.5345224838248488, # Critical Velocity Ratio U/U*
        1.33984375, # Critical Area Ratio A/A*
        np.nan, # Mach Angle
        np.nan, # Prandtl-Meyer Angle
    ]

    r = ise(param_name, value, gamma)
    assert len(r) == 11
    for e in r:
        assert not isinstance(e, np.ndarray)
    assert np.allclose(r, expected_res, equal_nan=True)


def test_to_dict():
    gamma = 1.4

    # single scalar mach
    M = 2
    r1 = ise("m", M, gamma, to_dict=False)
    assert len(r1) == 11
    for e in r1:
        assert not isinstance(e, np.ndarray)

    r2 = ise("m", M, gamma, to_dict=True)
    assert len(r2) == 11
    assert isinstance(r2, dict)

    assert np.isclose(r2["m"], r1[0])
    assert np.isclose(r2["pr"], r1[1])
    assert np.isclose(r2["dr"], r1[2])
    assert np.isclose(r2["tr"], r1[3])
    assert np.isclose(r2["prs"], r1[4])
    assert np.isclose(r2["drs"], r1[5])
    assert np.isclose(r2["trs"], r1[6])
    assert np.isclose(r2["urs"], r1[7])
    assert np.isclose(r2["ars"],  r1[8])
    assert np.isclose(r2["ma"], r1[9])
    assert np.isclose(r2["pm"], r1[10])


def test_error_for_multiple_gamma():
    error_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=error_msg):
        ise("m", [2, 4, 6], gamma=np.array([1.2, 1.3]))

    with pytest.raises(ValueError, match=error_msg):
        ise("m", [2, 4, 6], gamma=[1.2, 1.3])


@pytest.mark.parametrize("to_dict", [True, False])
def test_print_isentropic_results(to_dict):
    res1 = ise("m", 4, to_dict=to_dict)
    print_isentropic_results(res1)
    print_isentropic_results(res1, "{:.3f}")
