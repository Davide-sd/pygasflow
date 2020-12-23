# NOTE:
# 
# 1. The expected results comes from:
#    http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
#
# 2. Instead of testing every single function in the fanno.py module,
# I only test the fanno_solver function from the solvers.fanno module,
# which is going to call (almost) every function implemented in the
# fanno.py module. Hence I can easily compare the result with known values.

# TODO: 
# 1. The web calculator linked above doesn't provide values for the critical
# density ratio, hence that parameter is currently untested.

import numpy
import pytest
from pygasflow.solvers.fanno import fanno_solver

# TODO: test for critical density ratio

def check_val(v1, v2, tol=1e-05):
    assert abs(v1 - v2) < tol

tol = 1e-05
gamma = 1.4

def test_1(param_name, value, er, gamma=1.4):
    r = fanno_solver(param_name, value, gamma)
    assert len(r) == len(er)
    assert len(r) == 8
    check_val(r[0], er[0], tol)
    check_val(r[1], er[1], tol)
    check_val(r[2], er[2], tol)
    check_val(r[3], er[3], tol)
    check_val(r[4], er[4], tol)
    check_val(r[5], er[5], tol)
    check_val(r[6], er[6], tol)
    check_val(r[7], er[7], tol)

expected_res = [
    1, # M
    1, # Critical Pressure Ratio P/P*
    1, # Critical Density Ratio rho/rho*
    1, # Critical Temperature Ratio T/T*
    1, # Critical Total Pressure Ratio P0/P0*
    1, # Critical Velocity Ratio U/U*
    0, # Critical Friction Parameter 4fL*/D
    0, # Critical Entropy Ratio (s*-s)/R
]
test_1("m", 1, expected_res, gamma)
test_1("pressure", 1, expected_res, gamma)
test_1("density", 1, expected_res, gamma)
test_1("temperature", 1, expected_res, gamma)
test_1("total_pressure_sub", 1, expected_res, gamma)
test_1("total_pressure_super", 1, expected_res, gamma)
test_1("velocity", 1, expected_res, gamma)
test_1("friction_sub", 0, expected_res, gamma)
test_1("friction_super", 0, expected_res, gamma)
test_1("entropy_sub", 0, expected_res, gamma)
test_1("entropy_super", 0, expected_res, gamma)

# test subsonic-supersonic and multiple mach numbers
def test_2(param_name, value, er, gamma=1.4):
    r = fanno_solver(param_name, value, gamma)
    assert len(r) == len(er)
    assert len(r) == 8
    for i, (e, f) in enumerate(zip(r, er)):
        assert len(e) == len(f)
        if i != 2:
            for g, h in zip(e, f):
                check_val(g, h, tol)

# subsonic
expected_res = [
    [0.5, 1],        # M
    [2.13808993, 1], # Critical Pressure Ratio P/P*
    [None, None],    # Critical Density Ratio rho/rho*
    [1.14285714, 1], # Critical Temperature Ratio T/T*
    [1.33984375, 1], # Critical Total Pressure Ratio P0/P0*
    [0.53452248, 1], # Critical Velocity Ratio U/U*
    [1.06906031, 0], # Critical Friction Parameter 4fL*/D
    [0.29255300, 0], # Critical Entropy Ratio (s*-s)/R
]

test_2("m", [0.5, 1], expected_res, gamma)
test_2("pressure", [2.13808993, 1], expected_res, gamma)
# test_2("density", 1, expected_res, gamma)
test_2("temperature", [1.14285714, 1], expected_res, gamma)
test_2("total_pressure_sub", [1.33984375, 1], expected_res, gamma)
test_2("velocity", [0.53452248, 1], expected_res, gamma)
test_2("friction_sub", [1.06906031, 0], expected_res, gamma)
test_2("entropy_sub", [0.29255300, 0], expected_res, gamma)

# supersonic
expected_res = [
    [1, 2.5],        # M
    [1, 0.29211869], # Critical Pressure Ratio P/P*
    [None, None],    # Critical Density Ratio rho/rho*
    [1, 0.53333333], # Critical Temperature Ratio T/T*
    [1, 2.63671875], # Critical Total Pressure Ratio P0/P0*
    [1, 1.82574185], # Critical Velocity Ratio U/U*
    [0, 0.43197668], # Critical Friction Parameter 4fL*/D
    [0, 0.96953524], # Critical Entropy Ratio (s*-s)/R
]
test_2("m", [1, 2.5], expected_res, gamma)
test_2("pressure", [1, 0.29211869], expected_res, gamma)
# test_2("density", 1, expected_res, gamma)
test_2("temperature", [1, 0.53333333], expected_res, gamma)
test_2("total_pressure_super", [1, 2.63671875], expected_res, gamma)
test_2("velocity", [1, 1.82574185], expected_res, gamma)
test_2("friction_super", [0, 0.43197668], expected_res, gamma)
test_2("entropy_super", [0, 0.96953524], expected_res, gamma)
