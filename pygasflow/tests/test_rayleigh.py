# NOTE:
# 
# 1. The expected results comes from:
#    http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
#
# 2. Instead of testing every single function in the rayleigh.py module,
# I only test the rayleigh_solver function from the solvers.rayleigh module,
# which is going to call (almost) every function implemented in the
# rayleigh.py module. Hence I can easily compare the result with known values.

# TODO: 
# 1. The web calculator linked above doesn't provide values for the critical
# density ratio, hence that parameter is currently untested.
#
# 2. test_1("temperature_sub", 1, expected_res, gamma) return wrong results
#    (expected M=1). Investigate why that happens...

import numpy
import pytest
from pygasflow.solvers.rayleigh import rayleigh_solver

# TODO: test for critical density ratio

def check_val(v1, v2, tol=1e-05):
    assert abs(v1 - v2) < tol

tol = 1e-05
gamma = 1.4

def test_1(param_name, value, er, gamma=1.4):
    r = rayleigh_solver(param_name, value, gamma)
    # print(r)
    assert len(r) == len(er)
    assert len(r) == 8
    check_val(r[0], er[0], tol)
    check_val(r[1], er[1], tol)
    # check_val(r[2], er[2], tol)
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
    1, # Critical Total Temperature Ratio T0/T0*
    1, # Critical Velocity Ratio U/U*
    0, # Critical Entropy Ratio (s*-s)/R
]
test_1("m", 1, expected_res, gamma)
test_1("pressure", 1, expected_res, gamma)
test_1("density", 1, expected_res, gamma)
test_1("velocity", 1, expected_res, gamma)
# test_1("temperature_sub", 0.99999, expected_res, gamma)
test_1("temperature_super", 1, expected_res, gamma)
test_1("total_pressure_sub", 1, expected_res, gamma)
test_1("total_pressure_super", 1, expected_res, gamma)
test_1("total_temperature_sub", 1, expected_res, gamma)
test_1("total_temperature_super", 1, expected_res, gamma)
test_1("entropy_sub", 0, expected_res, gamma)
test_1("entropy_super", 0, expected_res, gamma)

# subsonic case
expected_res = [
    0.5, # M
    1.77777777, # Critical Pressure Ratio P/P*
    None, # Critical Density Ratio rho/rho*
    0.79012345, # Critical Temperature Ratio T/T*
    1.11405250, # Critical Total Pressure Ratio P0/P0*
    0.69135802, # Critical Total Temperature Ratio T0/T0*
    0.44444444, # Critical Velocity Ratio U/U*
    1.39984539, # Critical Entropy Ratio (s*-s)/R
]
test_1("m", 0.5, expected_res, gamma)
test_1("pressure", 1.77777777, expected_res, gamma)
# test_1("density", 1, expected_res, gamma)
test_1("velocity", 0.44444444, expected_res, gamma)
test_1("temperature_sub", 0.79012345, expected_res, gamma)
test_1("total_pressure_sub", 1.11405250, expected_res, gamma)
test_1("total_temperature_sub", 0.69135802, expected_res, gamma)
test_1("entropy_sub", 1.39984539, expected_res, gamma)

# supersonic case
expected_res = [
    2.5, # M
    0.24615384, # Critical Pressure Ratio P/P*
    None, # Critical Density Ratio rho/rho*
    0.37869822, # Critical Temperature Ratio T/T*
    2.22183128, # Critical Total Pressure Ratio P0/P0*
    0.71005917, # Critical Total Temperature Ratio T0/T0*
    1.53846153, # Critical Velocity Ratio U/U*
    1.99675616, # Critical Entropy Ratio (s*-s)/R
]
test_1("m", 2.5, expected_res, gamma)
test_1("pressure", 0.24615384, expected_res, gamma)
# test_1("density", 1, expected_res, gamma)
test_1("velocity", 1.53846153, expected_res, gamma)
test_1("temperature_super", 0.37869822, expected_res, gamma)
test_1("total_pressure_super", 2.22183128, expected_res, gamma)
test_1("total_temperature_super", 0.71005917, expected_res, gamma)
test_1("entropy_super", 1.99675616, expected_res, gamma)



def test_2(param_name, value, er, gamma=1.4):
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(er)
    assert len(r) == 8
    for i, (e, f) in enumerate(zip(r, er)):
        assert len(e) == len(f)
        if i != 2:
            for g, h in zip(e, f):
                # print(i, g, h)
                check_val(g, h, tol)
                # print("\t done")

# multiple values subsonic case
expected_res = [
    [0.5, 0.75], # M
    [1.77777777, 1.34265734], # Critical Pressure Ratio P/P*
    [None, None], # Critical Density Ratio rho/rho*
    [0.79012345, 1.01403491], # Critical Temperature Ratio T/T*
    [1.11405250, 1.03010358], # Critical Total Pressure Ratio P0/P0*
    [0.69135802, 0.94009487], # Critical Total Temperature Ratio T0/T0*
    [0.44444444, 0.75524475], # Critical Velocity Ratio U/U*
    [1.39984539, 0.24587005], # Critical Entropy Ratio (s*-s)/R
]
test_2("m", [0.5, 0.75], expected_res, gamma)
test_2("pressure", [1.77777777, 1.34265734], expected_res, gamma)
# test_2("density", [0.24615384, 0.06666666], expected_res, gamma)
test_2("velocity", [0.44444444, 0.75524475], expected_res, gamma)
test_2("temperature_sub", [0.79012345, 1.01403491], expected_res, gamma)
test_2("total_pressure_sub", [1.11405250, 1.03010358], expected_res, gamma)
test_2("total_temperature_sub", [0.69135802, 0.94009487], expected_res, gamma)
test_2("entropy_sub", [1.39984539, 0.24587005], expected_res, gamma)



# multiple values supersonic case
expected_res = [
    [2.5, 5], # M
    [0.24615384, 0.06666666], # Critical Pressure Ratio P/P*
    [None, None], # Critical Density Ratio rho/rho*
    [0.37869822, 0.11111111], # Critical Temperature Ratio T/T*
    [2.22183128, 18.6338998], # Critical Total Pressure Ratio P0/P0*
    [0.71005917, 0.55555555], # Critical Total Temperature Ratio T0/T0*
    [1.53846153, 1.66666666], # Critical Velocity Ratio U/U*
    [1.99675616, 4.98223581], # Critical Entropy Ratio (s*-s)/R
]
test_2("m", [2.5, 5], expected_res, gamma)
test_2("pressure", [0.24615384, 0.06666666], expected_res, gamma)
# test_2("density", [0.24615384, 0.06666666], expected_res, gamma)
test_2("velocity", [1.53846153, 1.66666666], expected_res, gamma)
test_2("temperature_super", [0.37869822, 0.11111111], expected_res, gamma)
test_2("total_pressure_super", [2.22183128, 18.6338998], expected_res, gamma)
test_2("total_temperature_super", [0.71005917, 0.55555555], expected_res, gamma)
test_2("entropy_super", [1.99675616, 4.98223581], expected_res, gamma)