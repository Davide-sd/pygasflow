# NOTE:
#
# 1. The expected results comes from:
#    http://www.dept.aoe.vt.edu/~devenpor/aoe3114/calc.html
#
# 2. Instead of testing every single function in the rayleigh.py module,
# I only test the rayleigh_solver function from the solvers.rayleigh module,
# which is going to call (almost) every function implemented in the
# rayleigh.py module. Hence I can easily compare the result with known values.

import numpy as np
from pygasflow.solvers.rayleigh import (
    rayleigh_solver,
    print_rayleigh_results,
    specific_heat_solver,
)
import pytest
from contextlib import redirect_stdout
import io


@pytest.mark.parametrize("param_name, value", [
    ("m", 1),
    ("pressure", 1),
    ("density", 1),
    ("velocity", 1),
    ("temperature_sub", 1),
    ("temperature_super", 1),
    ("total_pressure_sub", 1),
    ("total_pressure_super", 1),
    ("total_temperature_sub", 1),
    ("total_temperature_super", 1),
    ("entropy_sub", 0),
    ("entropy_super", 0),
])
def test_single_value_input_sonic_case(param_name, value):
    gamma = 1.4

    if param_name != "temperature_sub":
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
    else:
        # NOTE: look at RayleighDiagram. The curve T/T* has a maximum > 1
        # before M=1. So, the name "temperature_sub", "temperature_super" are
        # somehow misleading, because by providing values to
        # "temperature_super" it is still possible to compute M<1.
        # To compute M=1, "temperature_super=1" must be provided.
        expected_res = [
            0.714285714285714,
            1.4000000000000006,
            1.4000000000000006,
            1.0,
            1.039167495429634,
            0.9183673469387753,
            0.714285714285714,
            0.3364722366212134
        ]
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", 0.5),
    ("pressure", 1.77777777),
    ("density", 2.25000000),
    ("velocity", 0.44444444),
    ("temperature_sub", 0.79012345),
    ("total_pressure_sub", 1.11405250),
    ("total_temperature_sub", 0.69135802),
    ("entropy_sub", 1.39984539),
])
def test_single_value_input_subsonic_case(param_name, value):
    gamma = 1.4
    expected_res = [
        0.5, # M
        1.77777777, # Critical Pressure Ratio P/P*
        2.25000000, # Critical Density Ratio rho/rho*
        0.79012345, # Critical Temperature Ratio T/T*
        1.11405250, # Critical Total Pressure Ratio P0/P0*
        0.69135802, # Critical Total Temperature Ratio T0/T0*
        0.44444444, # Critical Velocity Ratio U/U*
        1.39984539, # Critical Entropy Ratio (s*-s)/R
    ]
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", 2.5),
    ("pressure", 0.24615384),
    ("density", 0.65),
    ("velocity", 1.53846153),
    ("temperature_super", 0.37869822),
    ("total_pressure_super", 2.22183128),
    ("total_temperature_super", 0.71005917),
    ("entropy_super", 1.99675616),
])
def test_single_value_input_supersonic_case(param_name, value):
    gamma = 1.4
    expected_res = [
        2.5, # M
        0.24615384, # Critical Pressure Ratio P/P*
        0.65, # Critical Density Ratio rho/rho*
        0.37869822, # Critical Temperature Ratio T/T*
        2.22183128, # Critical Total Pressure Ratio P0/P0*
        0.71005917, # Critical Total Temperature Ratio T0/T0*
        1.53846153, # Critical Velocity Ratio U/U*
        1.99675616, # Critical Entropy Ratio (s*-s)/R
    ]
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", [0.5, 0.75]),
    ("pressure", [1.77777777, 1.34265734]),
    ("density", [2.25      , 1.32407407]),
    ("velocity", [0.44444444, 0.75524475]),
    ("temperature_sub", [0.79012345, 1.01403491]),
    ("total_pressure_sub", [1.11405250, 1.03010358]),
    ("total_temperature_sub", [0.69135802, 0.94009487]),
    ("entropy_sub", [1.39984539, 0.24587005]),
])
def test_multiple_values_input_subsonic_case(param_name, value):
    gamma = 1.4
    expected_res = [
        [0.5, 0.75], # M
        [1.77777777, 1.34265734], # Critical Pressure Ratio P/P*
        [2.25      , 1.32407407], # Critical Density Ratio rho/rho*
        [0.79012345, 1.01403491], # Critical Temperature Ratio T/T*
        [1.11405250, 1.03010358], # Critical Total Pressure Ratio P0/P0*
        [0.69135802, 0.94009487], # Critical Total Temperature Ratio T0/T0*
        [0.44444444, 0.75524475], # Critical Velocity Ratio U/U*
        [1.39984539, 0.24587005], # Critical Entropy Ratio (s*-s)/R
    ]
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


@pytest.mark.parametrize("param_name, value", [
    ("m", [2.5, 5]),
    ("pressure", [0.24615384, 0.06666666]),
    ("density", [0.65, 0.6]),
    ("velocity", [1.53846153, 1.66666666]),
    ("temperature_super", [0.37869822, 0.11111111]),
    ("total_pressure_super", [2.22183128, 18.6338998]),
    ("total_temperature_super", [0.71005917, 0.55555555]),
    ("entropy_super", [1.99675616, 4.98223581]),
])
def test_multiple_values_input_supersonic_case(param_name, value):
    gamma = 1.4
    expected_res = [
        [2.5, 5], # M
        [0.24615384, 0.06666666], # Critical Pressure Ratio P/P*
        [0.65, 0.6], # Critical Density Ratio rho/rho*
        [0.37869822, 0.11111111], # Critical Temperature Ratio T/T*
        [2.22183128, 18.6338998], # Critical Total Pressure Ratio P0/P0*
        [0.71005917, 0.55555555], # Critical Total Temperature Ratio T0/T0*
        [1.53846153, 1.66666666], # Critical Velocity Ratio U/U*
        [1.99675616, 4.98223581], # Critical Entropy Ratio (s*-s)/R
    ]
    r = rayleigh_solver(param_name, value, gamma)
    assert len(r) == len(expected_res)
    assert len(r) == 8
    assert np.allclose(r, expected_res)


def test_to_dict():
    gamma = 1.4
    M = 0.5
    r1 = rayleigh_solver("m", M, gamma, to_dict=False)
    assert len(r1) == 8
    for e in r1:
        assert not isinstance(e, np.ndarray)

    r2 = rayleigh_solver("m", M, gamma, to_dict=True)
    assert len(r2) == 8
    assert isinstance(r2, dict)

    assert np.isclose(r2["m"], r1[0])
    assert np.isclose(r2["prs"], r1[1])
    assert np.isclose(r2["drs"], r1[2])
    assert np.isclose(r2["trs"], r1[3])
    assert np.isclose(r2["tprs"], r1[4])
    assert np.isclose(r2["ttrs"], r1[5])
    assert np.isclose(r2["urs"], r1[6])
    assert np.isclose(r2["eps"], r1[7])


def test_error_for_multiple_gamma():
    error_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=error_msg):
        rayleigh_solver("m", [2, 4, 6], gamma=np.array([1.2, 1.3]))

    with pytest.raises(ValueError, match=error_msg):
        rayleigh_solver("m", [2, 4, 6], gamma=[1.2, 1.3])


def test_print_rayleigh_results_number_formatter():
    res = rayleigh_solver("m", 4, to_dict=True)

    f1 = io.StringIO()
    with redirect_stdout(f1):
        print_rayleigh_results(res)
    output1 = f1.getvalue()

    f2 = io.StringIO()
    with redirect_stdout(f2):
        print_rayleigh_results(res, "{:.3f}")
    output2 = f2.getvalue()

    assert output1 != output2


@pytest.mark.parametrize("to_dict, expected", [
    (
        True,
        """key     quantity     
---------------------
m       M                 4.00000000
prs     P / P*            0.10256410
drs     rho / rho*        0.60937500
trs     T / T*            0.16831032
tprs    P0 / P0*          8.22684925
ttrs    T0 / T0*          0.58908613
urs     U / U*            1.64102564
eps     (s*-s) / R        3.95954318
"""
    ),
    (
        False,
        """idx   quantity     
-------------------
0     M                 4.00000000
1     P / P*            0.10256410
2     rho / rho*        0.60937500
3     T / T*            0.16831032
4     P0 / P0*          8.22684925
5     T0 / T0*          0.58908613
6     U / U*            1.64102564
7     (s*-s) / R        3.95954318
"""
    )
])
def test_show_rayleigh_results(to_dict, expected):
    res = rayleigh_solver("m", 4, to_dict=to_dict)

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected


@pytest.mark.parametrize("params", [
    {"Cp": 3, "T01": 2, "T02": 10},
    {"Cp": 3, "T01": 2, "DeltaT0": 8},
    {"Cp": 3, "T02": 10, "DeltaT0": 8},
    {"Cp": 3, "q": 24, "DeltaT0": 8, "T01": 2},
    {"Cp": 3, "q": 24, "DeltaT0": 8, "T02": 10},
    {"q": 24, "Cp": 3, "T01": 2},
    {"q": 24, "Cp": 3, "T02": 10},
    {"q": 24, "T01": 2, "T02": 10},

])
def test_specific_heat_solver_scalar_1(params):
    res = specific_heat_solver(**params)
    assert isinstance(res, dict)
    assert len(res) == 6
    assert len(set(res.keys()).difference(
        ["q", "Cp", "T01", "T02", "DeltaT0", "q_Cp"])) == 0
    assert np.isclose(res["q"], 24)
    assert np.isclose(res["Cp"], 3)
    assert np.isclose(res["T02"], 10)
    assert np.isclose(res["T01"], 2)
    assert np.isclose(res["DeltaT0"], 8)
    assert np.isclose(res["q_Cp"], 8)


@pytest.mark.parametrize("params, none_keys", [
    ({"Cp": 3, "DeltaT0": 8}, ["T01", "T02"]),
    ({"q_Cp": 8, "DeltaT0": 8}, ["q", "Cp", "T01", "T02"]),
    ({"DeltaT0": 8, "T01": 2}, ["q", "Cp"]),
    ({"q_Cp": 8, "T01": 2}, ["q", "Cp"]),
])
def test_specific_heat_solver_scalar_2(params, none_keys):
    res = specific_heat_solver(**params)
    assert isinstance(res, dict)
    assert len(res) == 6
    assert len(set(res.keys()).difference(
        ["q", "Cp", "T01", "T02", "DeltaT0", "q_Cp"])) == 0
    expected = {
        "q": 24,
        "Cp": 3,
        "T02": 10,
        "T01": 2,
        "DeltaT0": 8,
        "q_Cp": 8
    }
    for k in res.keys():
        if k in none_keys:
            assert res[k] is None
        else:
            assert np.isclose(res[k], expected[k])


def test_show_specific_heat_results():
    res = specific_heat_solver(Cp=3, T01=2, T02=10)
    expected = """key      quantity     
----------------------
q        q                24.00000000
Cp       Cp                3.00000000
T01      T01               2.00000000
T02      T02              10.00000000
DeltaT0  ΔT0               8.00000000
q_Cp     q / Cp            8.00000000
"""

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected
