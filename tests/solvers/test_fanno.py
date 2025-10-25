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
# density ratio, hence that parameter is currently tested against itself.
# Not ideal.

import numpy as np
from pygasflow.solvers.fanno import fanno_solver, print_fanno_results
import pytest
from contextlib import redirect_stdout
import io


@pytest.mark.parametrize("param_name, value", [
    ("m", 1),
    ("pressure", 1),
    ("density", 1),
    ("temperature", 1),
    ("total_pressure_sub", 1),
    ("total_pressure_super", 1),
    ("velocity", 1),
    ("friction_sub", 0),
    ("friction_super", 0),
    ("entropy_sub", 0),
    ("entropy_super", 0),
])
def test_single_parameter(param_name, value):
    gamma = 1.4
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

    r = fanno_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", [0.5, 1]),
    ("pressure", [2.13808993, 1]),
    ("density", [1.87082869, 1]),
    ("temperature", [1.14285714, 1]),
    ("total_pressure_sub", [1.33984375, 1]),
    ("velocity", [0.53452248, 1]),
    ("friction_sub", [1.06906031, 0]),
    ("entropy_sub", [0.29255300, 0]),
])
def test_multiple_parameters_subsonic(param_name, value):
    gamma = 1.4
    expected_res = [
        [0.5, 1],        # M
        [2.13808993, 1], # Critical Pressure Ratio P/P*
        [1.87082869, 1], # Critical Density Ratio rho/rho*
        [1.14285714, 1], # Critical Temperature Ratio T/T*
        [1.33984375, 1], # Critical Total Pressure Ratio P0/P0*
        [0.53452248, 1], # Critical Velocity Ratio U/U*
        [1.06906031, 0], # Critical Friction Parameter 4fL*/D
        [0.29255300, 0], # Critical Entropy Ratio (s*-s)/R
    ]

    r = fanno_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", [1, 2.5]),
    ("pressure", [1, 0.29211869]),
    ("density", [1, 0.54772256]),
    ("temperature", [1, 0.53333333]),
    ("total_pressure_super", [1, 2.63671875]),
    ("velocity", [1, 1.82574185]),
    ("friction_super", [0, 0.43197668]),
    ("entropy_super", [0, 0.96953524]),
])
def test_multiple_parameters_supersonic(param_name, value):
    gamma = 1.4
    expected_res = [
        [1, 2.5],        # M
        [1, 0.29211869], # Critical Pressure Ratio P/P*
        [1, 0.54772256], # Critical Density Ratio rho/rho*
        [1, 0.53333333], # Critical Temperature Ratio T/T*
        [1, 2.63671875], # Critical Total Pressure Ratio P0/P0*
        [1, 1.82574185], # Critical Velocity Ratio U/U*
        [0, 0.43197668], # Critical Friction Parameter 4fL*/D
        [0, 0.96953524], # Critical Entropy Ratio (s*-s)/R
    ]

    r = fanno_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


def test_to_dict():
    gamma = 1.4
    M = 2.5
    r1 = fanno_solver("m", M, gamma, to_dict=False)
    assert len(r1) == 8
    for e in r1:
        assert not isinstance(e, np.ndarray)

    r2 = fanno_solver("m", M, gamma, to_dict=True)
    assert len(r2) == 8
    assert isinstance(r2, dict)

    assert np.isclose(r2["m"], r1[0])
    assert np.isclose(r2["prs"], r1[1])
    assert np.isclose(r2["drs"], r1[2])
    assert np.isclose(r2["trs"], r1[3])
    assert np.isclose(r2["tprs"], r1[4])
    assert np.isclose(r2["urs"], r1[5])
    assert np.isclose(r2["fps"], r1[6])
    assert np.isclose(r2["eps"], r1[7])


def test_error_for_multiple_gamma():
    error_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=error_msg):
        fanno_solver("m", [2, 4, 6], gamma=np.array([1.2, 1.3]))

    with pytest.raises(ValueError, match=error_msg):
        fanno_solver("m", [2, 4, 6], gamma=[1.2, 1.3])


def test_print_fanno_results_number_formatter():
    res = fanno_solver("m", 4, to_dict=True)

    f1 = io.StringIO()
    with redirect_stdout(f1):
        print_fanno_results(res)
    output1 = f1.getvalue()

    f2 = io.StringIO()
    with redirect_stdout(f2):
        print_fanno_results(res, "{:.3f}")
    output2 = f2.getvalue()

    assert output1 != output2


@pytest.mark.parametrize("to_dict, expected", [
    (
        True,
        """key     quantity     
---------------------
m       M                 4.00000000
prs     P / P*            0.13363062
drs     rho / rho*        0.46770717
trs     T / T*            0.28571429
tprs    P0 / P0*         10.71875000
urs     U / U*            2.13808994
fps     4fL* / D          0.63306493
eps     (s*-s) / R        2.37199454
"""
    ),
    (
        False,
        """idx   quantity     
-------------------
0     M                 4.00000000
1     P / P*            0.13363062
2     rho / rho*        0.46770717
3     T / T*            0.28571429
4     P0 / P0*         10.71875000
5     U / U*            2.13808994
6     4fL* / D          0.63306493
7     (s*-s) / R        2.37199454
"""
    )
])
def test_show_isentropic_results(to_dict, expected):
    res = fanno_solver("m", 4, to_dict=to_dict)

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected
