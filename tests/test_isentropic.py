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
# velocity ratio, hence that parameter is currently untested.

import numpy as np
from pygasflow.solvers import isentropic_solver as ise
from pygasflow.isentropic import critical_pressure_ratio

def check_val(v1, v2, tol=1e-05):
    assert abs(v1 - v2) < tol

def test_input_mach():
    tol = 1e-05
    gamma = 1.4

    # single scalar mach
    M = 2
    r = ise("m", M, gamma)
    assert len(r) == 11
    for e in r:
        assert not isinstance(e, np.ndarray)

    check_val(r[0], 2, tol)
    check_val(r[1], 0.12780452, tol)
    check_val(r[2], 0.23004814, tol)
    check_val(r[3], 0.55555555, tol)
    check_val(r[4], 0.24192491, tol)
    check_val(r[5], 0.36288736, tol)
    check_val(r[6], 0.66666666, tol)
    check_val(r[7], r[7], tol)
    check_val(r[8],  1.68749999, tol)
    check_val(r[9], 29.9999999, tol)
    check_val(r[10], 26.3797608, tol)

    M = [2, 3]
    r = ise("m", M, gamma)
    assert len(r) == 11
    for e in r:
        assert isinstance(e, np.ndarray)
        assert len(e) == 2

    check_val(r[0][1], 3, tol)
    check_val(r[1][1], 0.02722368, tol)
    check_val(r[2][1], 0.07622631, tol)
    check_val(r[3][1], 0.35714285, tol)
    check_val(r[4][1], 0.05153250, tol)
    check_val(r[5][1], 0.12024251, tol)
    check_val(r[6][1], 0.42857142, tol)
    check_val(r[7][1], r[7][1], tol)
    check_val(r[8][1],   4.23456790, tol)
    check_val(r[9][1], 19.4712206, tol)
    check_val(r[10][1], 49.7573467, tol)


def test_input_parameters():
    tol = 1e-05
    gamma = 1.4

    def do_test(param_name, value, er):
        r = ise(param_name, value, gamma)
        assert len(r) == 11
        for e in r:
            assert not isinstance(e, np.ndarray)
        check_val(r[0], er[0], tol)
        check_val(r[1], er[1], tol)
        check_val(r[2], er[2], tol)
        check_val(r[3], er[3], tol)
        check_val(r[4], er[4], tol)
        check_val(r[5], er[5], tol)
        check_val(r[6], er[6], tol)
        check_val(r[7], r[7], tol)
        check_val(r[8], er[8], tol)
        if not np.isnan(r[9]):
            check_val(r[9], er[9], tol)
        if not np.isnan(r[10]):
            check_val(r[10], er[10], tol)

    # supersonic case
    expected_res = [
        3,          # Mach number
        0.02722368, # Pressure Ratio P/P0
        0.07622631, # Density Ratio rho/rho0
        0.35714285, # Temperature Ratio T/T0
        0.05153250, # Critical Pressure Ratio P/P*
        0.12024251, # Critical Density Ratio rho/rho*
        0.42857142, # Critical Temperature Ratio T/T*
        None,       # Critical Velocity Ratio U/U*
        4.23456790, # Critical Area Ratio A/A*
        19.4712206, # Mach Angle
        49.7573467, # Prandtl-Meyer Angle
    ]
    do_test("m", 3, expected_res)
    do_test("pressure", 0.02722368, expected_res)
    do_test("temperature", 0.35714285, expected_res)
    do_test("density", 0.07622631, expected_res)
    do_test("crit_area_super", 4.23456790, expected_res)
    do_test("prandtl_meyer", 49.7573467, expected_res)
    do_test("mach_angle", 19.4712206, expected_res)

    # subsonic case
    expected_res = [
        0.5,          # Mach number
        0.84301917, # Pressure Ratio P/P0
        0.88517013, # Density Ratio rho/rho0
        0.95238095, # Temperature Ratio T/T0
        1.59577557, # Critical Pressure Ratio P/P*
        1.39630363, # Critical Density Ratio rho/rho*
        1.14285714, # Critical Temperature Ratio T/T*
        None,       # Critical Velocity Ratio U/U*
        1.33984375, # Critical Area Ratio A/A*
        np.nan, # Mach Angle
        np.nan, # Prandtl-Meyer Angle
    ]
    do_test("m", 0.5, expected_res)
    do_test("pressure", 0.84301917, expected_res)
    do_test("temperature", 0.95238095, expected_res)
    do_test("density", 0.88517013, expected_res)
    do_test("crit_area_sub", 1.33984375, expected_res)
    # no prandtl_meyer-angle and mach-angle for subsonic case
    # do_test("prandtl_meyer", np.nan, expected_res)
    # do_test("mach_angle", np.nan, expected_res)


def test_to_dict():
    tol = 1e-05
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

    check_val(r2["m"], r1[0], tol)
    check_val(r2["pr"], r1[1], tol)
    check_val(r2["dr"], r1[2], tol)
    check_val(r2["tr"], r1[3], tol)
    check_val(r2["prs"], r1[4], tol)
    check_val(r2["drs"], r1[5], tol)
    check_val(r2["trs"], r1[6], tol)
    check_val(r2["urs"], r1[7], tol)
    check_val(r2["ars"],  r1[8], tol)
    check_val(r2["ma"], r1[9], tol)
    check_val(r2["pm"], r1[10], tol)
