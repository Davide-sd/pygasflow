import numpy as np
from pygasflow.solvers.gas import (
    gas_solver,
    ideal_gas_solver,
    sonic_condition,
    print_gas_results,
    print_ideal_gas_results,
    print_sonic_condition_results,
)
import pytest
from contextlib import redirect_stdout
import io


expected1 = [1.4, 287.05, 1004.6750000000002, 717.6250000000002]
expected2 = {
    'gamma': 1.4,
    'R': 287.05,
    'Cp': 1004.6750000000002,
    'Cv': 717.6250000000002
}
expected3 = [
    np.array([1.2, 1.4]),
    np.array([400, 287.05]),
    np.array([2400, 1004.6750000000002]),
    np.array([2000, 717.6250000000002])
]
expected4 = {
    'gamma': [1.2, 1.4],
    'R': [400, 287.05],
    'Cp': [2400, 1004.6750000000002],
    'Cv': [2000, 717.6250000000002]
}
expected5 = [101345.64336000002, 1.2259, 287.05, 288]
expected6 = {'p': 101345.64336000002, 'rho': 1.2259, 'R': 287.05, 'T': 288}
expected7 = [
    np.array([101345.64336, 154000.]),
    np.array([1.2259, 1.1]),
    np.array([287.05, 400.]),
    np.array([288, 350])
]
expected8 = {
    'p': np.array([101345.64336, 154000.]),
    'rho': np.array([1.2259, 1.1]),
    'R': np.array([287.05, 400.]),
    'T': np.array([288, 350])
}


@pytest.mark.parametrize("p1, v1, p2, v2, to_dict", [
    ("gamma", 1.4, "R", 287.05, False),
    ("R", 287.05, "gamma", 1.4, False),
    ("cp", 1004.6750000000002, "gamma", 1.4, False),
    ("gamma", 1.4, "cp", 1004.6750000000002, False),
    ("cp", 1004.6750000000002, "R", 287.05, False),
    ("R", 287.05, "cp", 1004.6750000000002, False),
    ("cv", 717.6250000000002, "gamma", 1.4, False),
    ("gamma", 1.4, "cv", 717.6250000000002, False),
    ("cv", 717.6250000000002, "R", 287.05, False),
    ("R", 287.05, "cv", 717.6250000000002, False),
    ("cv", 717.6250000000002, "cp", 1004.6750000000002, False),
    ("cp", 1004.6750000000002, "cv", 717.6250000000002, False),
    ("gamma", 1.4, "R", 287.05, True),
    ("R", 287.05, "gamma", 1.4, True),
    ("cp", 1004.6750000000002, "gamma", 1.4, True),
    ("cp", 1004.6750000000002, "R", 287.05, True),
    ("R", 287.05, "cp", 1004.6750000000002, True),
    ("gamma", 1.4, "cp", 1004.6750000000002, True),
    ("cv", 717.6250000000002, "gamma", 1.4, True),
    ("gamma", 1.4, "cv", 717.6250000000002, True),
    ("cv", 717.6250000000002, "R", 287.05, True),
    ("R", 287.05, "cv", 717.6250000000002, True),
    ("cv", 717.6250000000002, "cp", 1004.6750000000002, True),
    ("cp", 1004.6750000000002, "cv", 717.6250000000002, True),
])
def test_gas_solver_scalar(p1, v1, p2, v2, to_dict):
    res = gas_solver(p1, v1, p2, v2, to_dict=to_dict)
    if to_dict:
        assert len(set(res.keys()).difference(expected2.keys())) == 0
        for k in res:
            assert np.isclose(res[k], expected2[k])
    else:
        assert np.allclose(res, expected1)


@pytest.mark.parametrize("p1, v1, p2, v2, to_dict", [
    ("gamma", np.array([1.2, 1.4]), "R", np.array([400, 287.05]), False),
    ("cp", np.array([2400, 1004.6750000000002]), "gamma", np.array([1.2, 1.4]), False),
    ("cp", np.array([2400, 1004.6750000000002]), "R", np.array([400, 287.05]), False),
    ("cv", np.array([2000, 717.6250000000002]), "gamma", np.array([1.2, 1.4]), False),
    ("cv", np.array([2000, 717.6250000000002]), "R", np.array([400, 287.05]), False),
    ("cv", np.array([2000, 717.6250000000002]), "cp", np.array([2400, 1004.6750000000002]), False),
    ("gamma", np.array([1.2, 1.4]), "R", np.array([400, 287.05]), True),
    ("cp", np.array([2400, 1004.6750000000002]), "gamma", np.array([1.2, 1.4]), True),
    ("cp", np.array([2400, 1004.6750000000002]), "R", np.array([400, 287.05]), True),
    ("cv", np.array([2000, 717.6250000000002]), "gamma", np.array([1.2, 1.4]), True),
    ("cv", np.array([2000, 717.6250000000002]), "R", np.array([400, 287.05]), True),
    ("cv", np.array([2000, 717.6250000000002]), "cp", np.array([2400, 1004.6750000000002]), True),
])
def test_gas_solver_array(p1, v1, p2, v2, to_dict):
    res = gas_solver(p1, v1, p2, v2, to_dict=to_dict)
    if to_dict:
        assert len(set(res.keys()).difference(expected4.keys())) == 0
        for k in res:
            assert np.allclose(res[k], expected4[k])
    else:
        assert np.allclose(res, expected3)


def test_gas_solver_errors():
    # same parameter string
    with pytest.raises(
        ValueError,
        match="`p1_name` must be different from `p2_name`"
    ):
        gas_solver("gamma", 1.4, "gamma", 1.5)

    # wrong first parameter string
    with pytest.raises(
        ValueError,
        match="Wrong `p1_name` or `p2_name`"
    ):
        gas_solver("asd", 1.4, "gamma", 1.5)

    # wrong second parameter string
    with pytest.raises(
        ValueError,
        match="Wrong `p1_name` or `p2_name`"
    ):
        gas_solver("gamma", 1.4, "asd", 1.5)


@pytest.mark.parametrize("wanted, params, to_dict", [
    ("p", {'rho': 1.2259, 'R': 287.05, 'T': 288}, False),
    ("rho", {'p': 101345.64336000002, 'R': 287.05, 'T': 288}, False),
    ("R", {'p': 101345.64336000002, 'rho': 1.2259, 'T': 288}, False),
    ("T", {'p': 101345.64336000002, 'R': 287.05, 'rho': 1.2259}, False),
    ("p", {'rho': 1.2259, 'R': 287.05, 'T': 288}, True),
    ("rho", {'p': 101345.64336000002, 'R': 287.05, 'T': 288}, True),
    ("R", {'p': 101345.64336000002, 'rho': 1.2259, 'T': 288}, True),
    ("T", {'p': 101345.64336000002, 'R': 287.05, 'rho': 1.2259}, True),
])
def test_ideal_gas_solver_scalar(wanted, params, to_dict):
    res = ideal_gas_solver(wanted, to_dict=to_dict, **params)
    if to_dict:
        assert len(set(res.keys()).difference(expected6.keys())) == 0
        for k in res:
            assert np.isclose(res[k], expected6[k])
    else:
        assert np.allclose(res, expected5)


@pytest.mark.parametrize("wanted, params, to_dict", [
    ("p", {'rho': np.array([1.2259, 1.1]), 'R': np.array([287.05, 400]), 'T': np.array([288, 350])}, False),
    ("rho", {'p': np.array([101345.64336, 154000.]), 'R': np.array([287.05, 400]), 'T': np.array([288, 350])}, False),
    ("R", {'p': np.array([101345.64336, 154000.]), 'rho': np.array([1.2259, 1.1]), 'T': np.array([288, 350])}, False),
    ("T", {'p': np.array([101345.64336, 154000.]), 'R': np.array([287.05, 400]), 'rho': np.array([1.2259, 1.1])}, False),
    ("p", {'rho': np.array([1.2259, 1.1]), 'R': np.array([287.05, 400]), 'T': np.array([288, 350])}, True),
    ("rho", {'p': np.array([101345.64336, 154000.]), 'R': np.array([287.05, 400]), 'T': np.array([288, 350])}, True),
    ("R", {'p': np.array([101345.64336, 154000.]), 'rho': np.array([1.2259, 1.1]), 'T': np.array([288, 350])}, True),
    ("T", {'p': np.array([101345.64336, 154000.]), 'R': np.array([287.05, 400]), 'rho': np.array([1.2259, 1.1])}, True),
])
def test_ideal_gas_solver_array(wanted, params, to_dict):
    res = ideal_gas_solver(wanted, to_dict=to_dict, **params)
    if to_dict:
        assert len(set(res.keys()).difference(expected8.keys())) == 0
        for k in res:
            assert np.allclose(res[k], expected8[k])
    else:
        assert np.allclose(res, expected7)


def test_ideal_gas_solver_errors():
    # not enough parameters
    pytest.raises(ValueError, lambda: ideal_gas_solver("p"))
    pytest.raises(ValueError, lambda: ideal_gas_solver("p", rho=1))
    pytest.raises(ValueError, lambda: ideal_gas_solver("p", rho=1, T=2))

    # wrong `wanted` string
    pytest.raises(ValueError, lambda: ideal_gas_solver("asd", rho=1, T=2, p=3))

    # `wanted` is also a parameter
    pytest.raises(ValueError, lambda: ideal_gas_solver("p", rho=1, T=2, p=3))


def test_sonic_condition():
    expected_values = [
        1.5774409656148785,
        1.892929158737854,
        1.0954451150103321,
        1.2
    ]
    assert np.allclose(sonic_condition(1.4, to_dict=False), expected_values)

    gammas = [1.4, 1.5]
    expected_drs = [1.57744097, 1.5625]
    expected_prs = [1.89292916, 1.953125]
    expected_ars = [1.09544512, 1.11803399]
    expected_trs = [1.2, 1.25]
    res = sonic_condition(gammas, to_dict=True)
    assert np.allclose(res["ars"], expected_ars)
    assert np.allclose(res["drs"], expected_drs)
    assert np.allclose(res["prs"], expected_prs)
    assert np.allclose(res["trs"], expected_trs)

    assert np.allclose(sonic_condition(0.9, to_dict=False), [np.nan] * 4,
        equal_nan=True)
    res = sonic_condition([0.9] + gammas, to_dict=True)
    assert all(np.isnan(v[0]) for v in res.values())
    assert np.allclose(res["ars"][1:], expected_ars)
    assert np.allclose(res["drs"][1:], expected_drs)
    assert np.allclose(res["prs"][1:], expected_prs)
    assert np.allclose(res["trs"][1:], expected_trs)


@pytest.mark.parametrize("to_dict, expected", [
    (
        True,
        """key     quantity    
--------------------
gamma   gamma            1.40000000
R       R              287.05000000
Cp      Cp            1004.67500000
Cv      Cv             717.62500000
"""
    ),
    (
        False,
        """idx   quantity    
------------------
0     gamma            1.40000000
1     R              287.05000000
2     Cp            1004.67500000
3     Cv             717.62500000
"""
    )
])
def test_show_gas_results(to_dict, expected):
    res = gas_solver("gamma", 1.4, "R", 287.05, to_dict=to_dict)

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected


@pytest.mark.parametrize("to_dict, expected", [
    (
        True,
        """key     quantity    
--------------------
p       P           101345.64336000
rho     rho              1.22590000
R       R              287.05000000
T       T              288.00000000
"""
    ),
    (
        False,
        """idx   quantity    
------------------
0     P           101345.64336000
1     rho              1.22590000
2     R              287.05000000
3     T              288.00000000
"""
    )
])
def test_show_ideal_gas_results(to_dict, expected):
    res = ideal_gas_solver("p", rho=1.2259, R=287.05, T=288, to_dict=to_dict)

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected


@pytest.mark.parametrize("to_dict, expected", [
    (
        True,
        """key     quantity    
--------------------
drs     rho0/rho*        1.57744097
prs     P0/P*            1.89292916
ars     a0/T*            1.09544512
trs     T0/T*            1.20000000
"""
    ),
    (
        False,
        """idx   quantity    
------------------
0     rho0/rho*        1.57744097
1     P0/P*            1.89292916
2     a0/T*            1.09544512
3     T0/T*            1.20000000
"""
    )
])
def test_show_sonic_condition_results(to_dict, expected):
    res = sonic_condition(1.4, to_dict=to_dict)

    f = io.StringIO()
    with redirect_stdout(f):
        res.show()
    output = f.getvalue()

    # NOTE: for this tests to succeed, VSCode option
    # "trim trailing whitespaces in regex and strings"
    # must be disabled!
    assert output == expected


def test_print_gas_results_number_formatter():
    res = gas_solver("gamma", 1.4, "r", 287.05, to_dict=True)

    f1 = io.StringIO()
    with redirect_stdout(f1):
        print_gas_results(res)
    output1 = f1.getvalue()

    f2 = io.StringIO()
    with redirect_stdout(f2):
        print_gas_results(res, "{:.3f}")
    output2 = f2.getvalue()

    assert output1 != output2


def test_print_ideal_gas_results_number_formatter():
    res = ideal_gas_solver("p", R=287.05, T=288, rho=1.2259, to_dict=True)

    f1 = io.StringIO()
    with redirect_stdout(f1):
        print_ideal_gas_results(res)
    output1 = f1.getvalue()

    f2 = io.StringIO()
    with redirect_stdout(f2):
        print_ideal_gas_results(res, "{:.3f}")
    output2 = f2.getvalue()

    assert output1 != output2


def test_print_sonic_condition_results_number_formatter():
    res = sonic_condition(1.4, to_dict=True)

    f1 = io.StringIO()
    with redirect_stdout(f1):
        print_sonic_condition_results(res)
    output1 = f1.getvalue()

    f2 = io.StringIO()
    with redirect_stdout(f2):
        print_sonic_condition_results(res, "{:.3f}")
    output2 = f2.getvalue()

    assert output1 != output2
