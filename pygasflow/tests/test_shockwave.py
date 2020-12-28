# NOTE:
# 
# 1. The expected results comes from:
#    http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
#
# 2. Instead of testing every single function in the shockwave.py module,
# I only test the shockwave_solver and conical_shockwave_solver functions
# from the solvers.shockwave module, which are going to call (almost) every
# function implemented in the shockwave.py module. Hence I can easily compare 
# the result with known values.

import numpy as np
import pytest
from pytest import raises
from pygasflow.solvers.shockwave import (
    shockwave_solver as ss,
    conical_shockwave_solver as css
)

from pygasflow.shockwave import (
    shock_angle_from_mach_cone_angle,
    mach_cone_angle_from_shock_angle,
    mach_downstream
)

def check_val(v1, v2, tol=1e-05):
    assert abs(v1 - v2) < tol

def func(err, fn, *args, **kwargs):
    with raises(err):
        fn(*args, **kwargs)

gamma = 1.4
tol = 1e-05

# Raise error when:
func(ValueError, ss, "m1", 2, "asd", 4) # p2_name not in ["beta", "theta", "mn1"]
func(ValueError, ss, "m1", 2, "beta", None) # p2_value = None
func(ValueError, ss, "m1", 2, "beta", -10) # beta < 0
func(ValueError, ss, "m1", [2, 4], "beta", [20, -10]) # at least one beta < 0
func(ValueError, ss, "m1", 2, "beta", 100) # beta > 90
func(ValueError, ss, "m1", [2, 4], "beta", [20, 100]) # at least one beta > 90
func(ValueError, ss, "m1", 2, "theta", -10) # beta < 0
func(ValueError, ss, "m1", [2, 4], "theta", [20, -10]) # at least one beta < 0
func(ValueError, ss, "m1", 2, "theta", 100) # beta > 90
func(ValueError, ss, "m1", [2, 4], "theta", [20, 100]) # at least one beta > 90
func(ValueError, ss, "asd", 2, "beta", 4) # p1_name not in available_p1names
func(ValueError, ss, "mn1", 2, "mn1", 4) # p1_name = p2_name
func(ValueError, ss, "theta", 2) # p1_name = theta and beta=None
func(ValueError, ss, "theta", -10, "beta", 4) # p1_name = theta and theta < 0
func(ValueError, ss, "theta", 100, "beta", 4) # p1_name = theta and theta > 90
func(ValueError, ss, "beta", 2) # p1_name = beta and theta=None
func(ValueError, ss, "beta", -10, "theta", 4) # p1_name = theta and theta < 0
func(ValueError, ss, "beta", 100, "theta", 4) # p1_name = theta and theta > 90
func(ValueError, ss, "beta", 60, "theta", 45) # detachment
func(ValueError, ss, "beta", 89, "theta", 10) # detachment
func(ValueError, ss, "beta", 5, "theta", 10) # detachment


# Normal Shock Wave: beta=90deg
def test_1(param_name, value, er, gamma=1.4):
    r = ss(param_name, value, gamma=gamma)
    assert len(r) == len(er)
    assert len(r) == 10
    check_val(r[0], er[0], tol)
    check_val(r[1], er[1], tol)
    check_val(r[2], er[2], tol)
    check_val(r[3], er[3], tol)
    check_val(r[4], er[4], tol)
    check_val(r[5], er[5], tol)
    check_val(r[6], er[6], tol)
    check_val(r[7], er[7], tol)
    check_val(r[8], er[8], tol)
    check_val(r[9], er[9], tol)

expected_res = [
    2,          # M1
    2,          # MN1
    0.57735026, # M2
    0.57735026, # MN2
    90,         # beta
    0,          # theta
    4.5,        # Pressure Ratio, P2/P1
    2.66666666, # Density Ratio, RHO2/RHO1
    1.6875,     # Temperature Ratio, T2/T1
    0.72087386, # Total Pressure Ratio, P02/P01
]
test_1("m1", 2, expected_res, 1.4)
test_1("mn2", 0.57735026, expected_res, 1.4)
test_1("pressure", 4.5, expected_res, 1.4)
test_1("temperature", 1.6875, expected_res, 1.4)
test_1("density", 2.66666666, expected_res, 1.4)
test_1("total_pressure", 0.72087386, expected_res, 1.4)

expected_res = [
    2,          # M1
    2,          # MN1
    0.52522573, # M2
    0.52522573, # MN2
    90,         # beta
    0,          # theta
    4.14285714, # Pressure Ratio, P2/P1
    3.5,        # Density Ratio, RHO2/RHO1
    1.18367346, # Temperature Ratio, T2/T1
    0.64825952, # Total Pressure Ratio, P02/P01
]
test_1("m1", 2, expected_res, 1.1)
test_1("mn2", 0.52522573, expected_res, 1.1)
test_1("pressure", 4.14285714, expected_res, 1.1)
test_1("temperature", 1.18367346, expected_res, 1.1)
test_1("density", 3.5, expected_res, 1.1)
test_1("total_pressure", 0.64825952, expected_res, 1.1)


# Oblique Shock Wave
def test_2(param_name, value, angle_name, angle_value, er, gamma=1.4, flag="weak"):
    r = ss(param_name, value, angle_name, angle_value, gamma=gamma, flag=flag)
    assert len(r) == len(er)
    assert len(r) == 10
    check_val(r[0], er[0], tol)
    check_val(r[1], er[1], tol)
    check_val(r[2], er[2], tol)
    check_val(r[3], er[3], tol)
    check_val(r[4], er[4], tol)
    check_val(r[5], er[5], tol)
    check_val(r[6], er[6], tol)
    check_val(r[7], er[7], tol)
    check_val(r[8], er[8], tol)
    check_val(r[9], er[9], tol)

# weak oblique shock
expected_res = [
    5,          # M1
    2.48493913, # MN1
    3.02215162, # M2
    0.51444650, # MN2
    29.8009155, # beta
    20,         # theta
    7.03740958, # Pressure Ratio, P2/P1
    3.31541762, # Density Ratio, RHO2/RHO1
    2.12263141, # Temperature Ratio, T2/T1
    0.50507023, # Total Pressure Ratio, P02/P01
]
test_2("m1", 5, "theta", 20, expected_res, 1.4, flag="weak")
test_2("m1", 5, "beta", 29.8009155, expected_res, 1.4, flag="weak")
test_2("m1", 5, "mn1", 2.48493913, expected_res, 1.4, flag="weak")

# strong oblique shock
expected_res = [
    5,          # M1
    4.97744911, # MN1
    0.46018704, # M2
    0.41555237, # MN2
    84.5562548, # beta
    20,         # theta
    28.7374996, # Pressure Ratio, P2/P1
    4.99244331, # Density Ratio, RHO2/RHO1
    5.75619948, # Temperature Ratio, T2/T1
    0.06280201, # Total Pressure Ratio, P02/P01
]
test_2("m1", 5, "theta", 20, expected_res, 1.4, flag="strong")
test_2("m1", 5, "beta", 84.5562548, expected_res, 1.4, flag="strong")
test_2("m1", 5, "mn1", 4.97744911, expected_res, 1.4, flag="strong")

# Conical Shock Wave
# Raise error when:
func(ValueError, css, 2, "mcc", 1.5) # wrong parameter name
func(ValueError, css, 2, "mc", 2) # Mc = M1
func(ValueError, css, 2, "mc", 3) # Mc > M1
func(ValueError, css, 2, "beta", -30) # beta < 0
func(ValueError, css, 2, "beta", 100) # beta > 90
func(ValueError, css, 2, "theta_c", -30) # theta_c < 0
func(ValueError, css, 2, "theta_c", 100) # theta_c > 90
func(ValueError, css, 2, "theta_c", 45) # detachment
func(ValueError, css, 2, "beta", 20) # detachment


def test_3(m1, param_name, value, er, gamma=1.4, tol=1e-05):
    r = css(m1, param_name, value, gamma)
    assert len(r) == len(er)
    assert len(r) == 12
    check_val(r[0], er[0], tol)
    check_val(r[1], er[1], tol)
    check_val(r[2], er[2], tol)
    check_val(r[3], er[3], tol)
    check_val(r[4], er[4], tol)
    check_val(r[5], er[5], tol)
    check_val(r[6], er[6], tol)
    check_val(r[7], er[7], tol)
    check_val(r[8], er[8], tol)
    check_val(r[9], er[9], tol)
    check_val(r[10], er[10], tol)
    check_val(r[11], er[11], tol)


expected_res = [
    5,          # M1
    3.37200575, # Mc
    20,         # theta_c
    24.9785489, # beta
    15.6245656, # delta
    5.03431828, # Pressure Ratio, P2/P1
    2.82807772, # Density Ratio, RHO2/RHO1
    1.78012020, # Temperature Ratio, RHO2/RHO1
    0.66891050, # Total Pressure Ratio, P0c/P01
    5.57291949, # Pc/P1
    3.04103495, # RHOc/RHO1
    1.83257331, # Tc/T1
]

# very high tolerances since the online calculator is using a fixed-step 
# iterative procedure, where I'm using bisection
test_3(5, "theta_c", 20, expected_res, 1.4, 1e-01)
test_3(5, "beta", 24.9785489, expected_res, 1.4, 1e-01)
test_3(5, "mc", 3.37200575, expected_res, 1.4, 1e-01)

# Test multiple Mach numbers on Normal Shock Wave
def test_4(param_name, value, er, gamma=1.4):
    r = ss(param_name, value, gamma=gamma)
    assert len(r) == len(er)
    assert len(r) == 10
    for e, f in zip(r, er):
        assert len(e) == len(f)
        for g, h in zip(e, f):
            check_val(g, h, tol)

expected_res = [
    [2, 5],                     # M1
    [2, 5],                     # MN1
    [0.57735026, 0.41522739],   # M2
    [0.57735026, 0.41522739],   # MN2
    [90, 90],                   # beta
    [0, 0],                     # theta
    [4.5, 29],                  # Pressure Ratio, P2/P1
    [2.66666666, 5],            # Density Ratio, RHO2/RHO1
    [1.6875, 5.79999999],       # Temperature Ratio, T2/T1
    [0.72087386, 0.06171631],   # Total Pressure Ratio, P02/P01
]
test_4("m1", [2, 5], expected_res, 1.4)
test_4("mn2", [0.57735026, 0.41522739], expected_res, 1.4)
test_4("pressure", [4.5, 29], expected_res, 1.4)
test_4("temperature", [1.6875, 5.79999999], expected_res, 1.4)
test_4("density", [2.66666666, 5], expected_res, 1.4)
test_4("total_pressure", [0.72087386, 0.06171631], expected_res, 1.4)

# Test multiple Mach numbers on Weak Oblique Shock Wave
def test_5(param_name, value, angle_name, angle_value, er, gamma=1.4, flag="weak"):
    r = ss(param_name, value, angle_name, angle_value, gamma=gamma, flag=flag)
    assert len(r) == len(er)
    assert len(r) == 10
    for e, f in zip(r, er):
        assert len(e) == len(f)
        for g, h in zip(e, f):
            check_val(g, h, tol)

expected_res = [
    [2, 5],                   # M1
    [1.60611226, 2.48493913], # MN1
    [1.21021838, 3.02215162], # M2
    [0.66660639, 0.51444650], # MN2
    [53.4229405, 29.8009155], # beta
    [20, 20],                 # theta
    [2.84286270, 7.03740958], # Pressure Ratio, P2/P1
    [2.04200572, 3.31541762], # Density Ratio, RHO2/RHO1
    [1.39219135, 2.12263141], # Temperature Ratio, T2/T1
    [0.89291398, 0.50507023], # Total Pressure Ratio, P02/P01
]
# also test for multiple angle values and input normal mach
test_5("m1", [2, 5], "theta", [20, 20], expected_res, 1.4, flag="weak")
test_5("m1", [2, 5], "beta", [53.4229405, 29.8009155], expected_res, 1.4, flag="weak")
test_5("m1", [2, 5], "mn1", [1.60611226, 2.48493913], expected_res, 1.4, flag="weak")

# Test multiple Mach numbers on Strong Oblique Shock Wave
expected_res = [
    [2, 5],                   # M1
    [1.92510115, 4.97744911], # MN1
    [0.72778853, 0.46018704], # M2
    [0.59080365, 0.41555237], # MN2
    [74.2701370, 84.5562548], # beta
    [20, 20],                 # theta
    [4.15701686, 28.7374996], # Pressure Ratio, P2/P1
    [2.55410634, 4.99244331], # Density Ratio, RHO2/RHO1
    [1.62758174, 5.75619948], # Temperature Ratio, T2/T1
    [0.75575659, 0.06280201], # Total Pressure Ratio, P02/P01
]
# also test for multiple angle values and input normal mach
test_5("m1", [2, 5], "theta", [20, 20], expected_res, 1.4, flag="strong")
test_5("m1", [2, 5], "beta", [74.2701370, 84.5562548], expected_res, 1.4, flag="strong")
test_5("m1", [2, 5], "mn1", [1.92510115, 4.97744911], expected_res, 1.4, flag="strong")
