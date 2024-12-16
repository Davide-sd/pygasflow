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
import os
import pytest
from pygasflow.generic import characteristic_mach_number
from pygasflow.solvers.shockwave import (
    shockwave_solver as ss,
    conical_shockwave_solver as css,
    normal_shockwave_solver as nss
)
from pygasflow.shockwave import (
    shock_angle_from_mach_cone_angle,
    mach_cone_angle_from_shock_angle,
    mach_downstream,
    oblique_mach_downstream,
    beta_from_mach_max_theta,
    load_data,
    create_mach_beta_theta_c_csv_file,
    PressureDeflectionLocus,
    max_theta_from_mach,
    shock_polar_equation,
    shock_polar,
    beta_theta_max_for_unit_mach_downstream,
    beta_theta_c_for_unit_mach_downstream
)
from tempfile import TemporaryDirectory
from numbers import Number


def check_val(v1, v2, tol=1e-05):
    assert abs(v1 - v2) < tol


def func(err, fn, *args, **kwargs):
    with pytest.raises(err):
        fn(*args, **kwargs)

def test_raises_error():
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


def test_normal_shock_wave():
    gamma = 1.4
    tol = 1e-05

    # Normal Shock Wave: beta=90deg
    def do_test(param_name, value, er, gamma=1.4):
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
    do_test("m1", 2, expected_res, 1.4)
    do_test("mn2", 0.57735026, expected_res, 1.4)
    do_test("pressure", 4.5, expected_res, 1.4)
    do_test("temperature", 1.6875, expected_res, 1.4)
    do_test("density", 2.66666666, expected_res, 1.4)
    do_test("total_pressure", 0.72087386, expected_res, 1.4)

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
    do_test("m1", 2, expected_res, 1.1)
    do_test("mn2", 0.52522573, expected_res, 1.1)
    do_test("pressure", 4.14285714, expected_res, 1.1)
    do_test("temperature", 1.18367346, expected_res, 1.1)
    do_test("density", 3.5, expected_res, 1.1)
    do_test("total_pressure", 0.64825952, expected_res, 1.1)


def test_oblique_shockwave():
    gamma = 1.4
    tol = 1e-05

    # Oblique Shock Wave
    def do_test(param_name, value, angle_name, angle_value, er, gamma=1.4, flag="weak"):
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
    do_test("m1", 5, "theta", 20, expected_res, 1.4, flag="weak")
    do_test("m1", 5, "beta", 29.8009155, expected_res, 1.4, flag="weak")
    do_test("m1", 5, "mn1", 2.48493913, expected_res, 1.4, flag="weak")

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
    do_test("m1", 5, "theta", 20, expected_res, 1.4, flag="strong")
    do_test("m1", 5, "beta", 84.5562548, expected_res, 1.4, flag="strong")
    do_test("m1", 5, "mn1", 4.97744911, expected_res, 1.4, flag="strong")


def test_conical_shockwave():
    gamma = 1.4
    tol = 1e-05

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


    def do_test(m1, param_name, value, er, gamma=1.4, tol=1e-05):
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
    do_test(5, "theta_c", 20, expected_res, 1.4, 1e-01)
    do_test(5, "beta", 24.9785489, expected_res, 1.4, 1e-01)
    do_test(5, "mc", 3.37200575, expected_res, 1.4, 1e-01)


def test_normal_shockwave_multiple_input_mach():
    gamma = 1.4
    tol = 1e-05

    # Test multiple Mach numbers on Normal Shock Wave
    def do_test(param_name, value, er, gamma=1.4):
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
    do_test("m1", [2, 5], expected_res, 1.4)
    do_test("mn2", [0.57735026, 0.41522739], expected_res, 1.4)
    do_test("pressure", [4.5, 29], expected_res, 1.4)
    do_test("temperature", [1.6875, 5.79999999], expected_res, 1.4)
    do_test("density", [2.66666666, 5], expected_res, 1.4)
    do_test("total_pressure", [0.72087386, 0.06171631], expected_res, 1.4)


def test_oblique_shockwave_multiple_input_mach():
    gamma = 1.4
    tol = 1e-05

    # Test multiple Mach numbers on Weak Oblique Shock Wave
    def do_test(param_name, value, angle_name, angle_value, er, gamma=1.4, flag="weak"):
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
    do_test("m1", [2, 5], "theta", [20, 20], expected_res, 1.4, flag="weak")
    do_test("m1", [2, 5], "beta", [53.4229405, 29.8009155], expected_res, 1.4, flag="weak")
    do_test("m1", [2, 5], "mn1", [1.60611226, 2.48493913], expected_res, 1.4, flag="weak")

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
    do_test("m1", [2, 5], "theta", [20, 20], expected_res, 1.4, flag="strong")
    do_test("m1", [2, 5], "beta", [74.2701370, 84.5562548], expected_res, 1.4, flag="strong")
    do_test("m1", [2, 5], "mn1", [1.92510115, 4.97744911], expected_res, 1.4, flag="strong")


def test_shockwave_solver_to_dict():
    tol = 1e-05
    gamma = 1.4
    M = 2
    r1 = ss("m1", M, gamma=gamma, to_dict=False)
    assert len(r1) == 10

    r2 = ss("m1", M, gamma=gamma, to_dict=True)
    assert len(r2) == 10
    assert isinstance(r2, dict)

    check_val(r2["m1"], r1[0], tol)
    check_val(r2["mn1"], r1[1], tol)
    check_val(r2["m2"], r1[2], tol)
    check_val(r2["mn2"], r1[3], tol)
    check_val(r2["beta"], r1[4], tol)
    check_val(r2["theta"], r1[5], tol)
    check_val(r2["pr"], r1[6], tol)
    check_val(r2["dr"], r1[7], tol)
    check_val(r2["tr"], r1[8], tol)
    check_val(r2["tpr"], r1[9], tol)


def test_conical_shockwave_solver_to_dict():
    tol = 1e-05
    gamma = 1.4

    r1 = css(5, "theta_c", 20, 1.4, to_dict=False)
    assert len(r1) == 12

    r2 = css(5, "theta_c", 20, 1.4, to_dict=True)
    assert len(r2) == 12
    assert isinstance(r2, dict)

    check_val(r2["m"], r1[0], tol)
    check_val(r2["mc"], r1[1], tol)
    check_val(r2["theta_c"], r1[2], tol)
    check_val(r2["beta"], r1[3], tol)
    check_val(r2["delta"], r1[4], tol)
    check_val(r2["pr"], r1[5], tol)
    check_val(r2["dr"], r1[6], tol)
    check_val(r2["tr"], r1[7], tol)
    check_val(r2["tpr"], r1[8], tol)
    check_val(r2["pc_p1"], r1[9], tol)
    check_val(r2["rhoc_rho1"], r1[10], tol)
    check_val(r2["Tc_T1"], r1[11], tol)


@pytest.mark.parametrize("beta, expected_mach_downstream", [
    (60, [1.04454822, 1.12256381]),
    ([60, 30], [1.04454822, 2.36734555])
])
def test_oblique_mach_downstream_input_beta(beta, expected_mach_downstream):
    M1 = [1.5, 3]
    solver_results = ss(
        "m1", M1, "beta", beta, to_dict=True)
    actual_mach_downstream = oblique_mach_downstream(
        M1,
        beta=beta,
    )
    assert np.allclose(solver_results["m2"], expected_mach_downstream)
    assert np.allclose(actual_mach_downstream, expected_mach_downstream)



@pytest.mark.parametrize("theta, flag, expected_mach_downstream", [
    (10, "weak", [2.50500068, 3.99916193]),
    (10, "strong", [0.48924158, 0.42546429]),
    ([10, 20], "weak", [2.50500068, 3.02215165]),
    ([10, 20], "strong", [0.48924158, 0.46018705])
])
def test_oblique_mach_downstream_input_theta(theta, flag, expected_mach_downstream):
    M1 = [3, 5]
    solver_results = ss(
        "m1", M1, "theta", theta, to_dict=True, flag=flag)
    actual_mach_downstream = oblique_mach_downstream(
        M1,
        theta=theta,
        flag=flag
    )
    assert np.allclose(solver_results["m2"], expected_mach_downstream)
    assert np.allclose(actual_mach_downstream, expected_mach_downstream)


def test_oblique_mach_downstream_errors():
    M1 = [3, 5]
    pytest.raises(ValueError, lambda: oblique_mach_downstream(
        M1,
        theta=None,
        beta=None,
    ))

    pytest.raises(ValueError, lambda: oblique_mach_downstream(
        M1,
        theta=10,
        flag="both"
    ))


def test_beta_from_mach_max_theta():
    assert np.isclose(
        beta_from_mach_max_theta(2.5, 1.4),
        64.78216996529343
    )
    assert np.allclose(
        beta_from_mach_max_theta([2.5, 3.5], 1.4),
        [64.78216997, 65.68861403]
    )


def test_error_for_multiple_gamma():
    err_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=err_msg):
        ss("m1", [2, 3], "beta", 80, gamma=[1.1, 2])

    with pytest.raises(ValueError, match=err_msg):
        nss("m1", [2, 3], gamma=[1.1, 2])

    with pytest.raises(ValueError, match=err_msg):
        css([2.5, 5], "mc", 1.5, gamma=[1.1, 2])


@pytest.mark.parametrize("g", [0.9, 1])
def test_error_gamma_less_equal_than_one(g):
    err_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=err_msg):
        ss("m1", [2, 3], "beta", 80, gamma=g)

    with pytest.raises(ValueError, match=err_msg):
        nss("m1", [2, 3], gamma=g)

    with pytest.raises(ValueError, match=err_msg):
        css([2.5, 5], "mc", 1.5, gamma=g)


@pytest.mark.parametrize("gamma, raise_error", [
    (1.05, False),
    (1.1, False),
    (1.15, False),
    (1.2, False),
    (1.25, False),
    (1.3, False),
    (1.35, False),
    (1.4, False),
    (1.45, False),
    (1.5, False),
    (1.55, False),
    (1.6, False),
    (1.65, False),
    (1.7, False),
    (1.75, False),
    (1.8, False),
    (1.85, False),
    (1.9, False),
    (1.95, False),
    (2, False),
    (2.05, True)
])
def test_load_data(gamma, raise_error):
    if not raise_error:
        data = load_data(gamma)
        assert len(data) == 3
        assert all(isinstance(d, np.ndarray) for d in data)
    else:
        with pytest.raises(FileNotFoundError):
            load_data(gamma)


def test_create_mach_beta_theta_c_csv_file():
    with pytest.raises(TypeError):
        # gamma is not iterable
        create_mach_beta_theta_c_csv_file([1, 1.1], 1.4)
    with pytest.raises(TypeError):
        # M1 is not iterable
        create_mach_beta_theta_c_csv_file(2, [1.1, 1.4])
    with pytest.raises(TypeError):
        # both M1 and gamma are not iterables
        create_mach_beta_theta_c_csv_file(2, 1.4)
    with TemporaryDirectory(prefix="pygasflow_") as tmpdir:
        create_mach_beta_theta_c_csv_file(
            [1, 1.05], [1.35, 1.4], folder=tmpdir)
        assert os.path.exists(os.path.join(
            tmpdir,
            "m-beta-theta_c-g1.35.csv.zip"
        ))
        assert os.path.exists(os.path.join(
            tmpdir,
            "m-beta-theta_c-g1.4.csv.zip"
        ))
    with TemporaryDirectory(prefix="pygasflow_") as tmpdir:
        create_mach_beta_theta_c_csv_file(
            [1.05], [1.35], folder=tmpdir, filename="test%s.csv.zip")
        assert os.path.exists(os.path.join(
            tmpdir,
            "test1.35.csv.zip"
        ))


class Test_PressureDeflectionLocus:
    def _do_test_instantion(
        self, obj, M, gamma, theta_origin, pr_to_freestream, label
    ):
        assert obj.M == M
        assert obj.gamma == gamma
        assert obj.theta_origin == theta_origin
        assert obj.pr_to_freestream == pr_to_freestream
        assert obj.label == label
        assert callable(obj.shockwave_at_theta)
        assert np.isclose(obj.theta_max, max_theta_from_mach(M, gamma))

    def test_instantiation_simple(self):
        p = PressureDeflectionLocus(M=2)
        self._do_test_instantion(p, 2, 1.4, 0, 1, "")

    def test_instantiation_M_gamma(self):
        M = 3
        gamma = 1.2
        p = PressureDeflectionLocus(M=M, gamma=gamma)
        self._do_test_instantion(p, 3, 1.2, 0, 1, "")

    def test_instantiation_advanced(self):
        p = PressureDeflectionLocus(
            M=2, gamma=1.4, theta_origin=10, pr_to_freestream=3, label="test")
        self._do_test_instantion(p, 2, 1.4, 10, 3, "test")

    def test_new_locus_from_shockwave(self):
        pdl1 = PressureDeflectionLocus(M=3, label="1")
        pdl2 = pdl1.new_locus_from_shockwave(20, label="2")
        self._do_test_instantion(
            pdl2, 1.9941316655645605, 1.4, 20, 3.771257463082658, "2")

        pdl1 = PressureDeflectionLocus(M=3, label="1", gamma=1.2)
        pdl2 = pdl1.new_locus_from_shockwave(20, label="2")
        assert np.isclose(pdl2.gamma, 1.2)

    def test_update_func_theta_max(self):
        p = PressureDeflectionLocus(M=2)
        f1 = p.shockwave_at_theta
        tm1 = p.theta_max

        p.M = 3
        f2 = p.shockwave_at_theta
        tm2 = p.theta_max
        assert not np.isclose(tm1, tm2)
        assert id(f1) != id(f2)

        p.gamma = 1.2
        f3 = p.shockwave_at_theta
        tm3 = p.theta_max
        assert not np.isclose(tm2, tm3)
        assert id(f2) != id(f3)

        p.theta_origin = 10
        f4 = p.shockwave_at_theta
        tm4 = p.theta_max
        assert np.isclose(tm3, tm4)
        assert id(f3) != id(f4)

        p.pr_to_freestream = 3
        f5 = p.shockwave_at_theta
        tm5 = p.theta_max
        assert np.isclose(tm4, tm5)
        assert id(f4) != id(f5)

    def test_intersection_between_locuses(self):
        pdl1 = PressureDeflectionLocus(M=3, label="1")
        pdl2 = pdl1.new_locus_from_shockwave(20, label="2")
        pdl3 = pdl1.new_locus_from_shockwave(-15, label="3")

        theta_inter, pr_inter = pdl2.intersection(pdl3, region="weak")
        assert np.isclose(theta_inter, 4.795958931693682)
        assert np.isclose(pr_inter, 8.352551913417367)

        theta_inter, pr_inter = pdl2.intersection(pdl3, region="strong")
        assert np.isclose(theta_inter, 1.4095306470178952)
        assert np.isclose(pr_inter, 15.852431019354935)

        theta_inter, pr_inter = pdl1.intersection(pdl2, region="weak")
        assert np.isclose(theta_inter, 32.511026902792594)
        assert np.isclose(pr_inter, 7.299330121992539)

        theta_inter, pr_inter = pdl1.intersection(pdl3, region="weak")
        assert np.isclose(theta_inter, -33.51465769303509)
        assert np.isclose(pr_inter, 7.8289804826169425)

        pdl4 = pdl2.new_locus_from_shockwave(20)
        theta_inter, pr_inter = pdl3.intersection(pdl4, region="weak")
        assert theta_inter is None
        assert pr_inter is None

    def test_create_path(self):
        theta_2 = 20
        theta_3 = -15
        pdl1 = PressureDeflectionLocus(M=3, label="1")
        pdl2 = pdl1.new_locus_from_shockwave(theta_2, label="2")
        pdl3 = pdl1.new_locus_from_shockwave(theta_3, label="3")

        # error: first element of the tuple is not an instance of PressureDeflectionLocus
        pytest.raises(
            ValueError,
            lambda: PressureDeflectionLocus.create_path((0, theta_2))
        )
        # error: second element of the tuple is not a number
        pytest.raises(
            ValueError,
            lambda: PressureDeflectionLocus.create_path((pdl1, pdl2))
        )
        # error: tuples with number of elements != 2
        pytest.raises(
            ValueError,
            lambda: PressureDeflectionLocus.create_path((pdl1, theta_2, 3))
        )
        # error: not a tuple
        pytest.raises(
            ValueError,
            lambda: PressureDeflectionLocus.create_path(pdl1, theta_2, 3)
        )

        theta, pr = PressureDeflectionLocus.create_path((pdl1, theta_2))
        assert len(theta) == len(pr) == 100

        theta, pr = PressureDeflectionLocus.create_path((pdl1, theta_2), N=5)
        assert len(theta) == len(pr) == 5
        assert np.allclose(theta, np.array([
            0.,  5., 10., 15., 20.
        ]))
        assert np.allclose(pr, np.array([
            1.        , 1.45398306, 2.05447215, 2.82156232, 3.77125746
        ]))

        phi, _ = pdl2.intersection(pdl3, region="weak")
        theta, pr = PressureDeflectionLocus.create_path(
            (pdl1, theta_2), (pdl2, phi), N=5)
        assert len(theta) == len(pr) == 10
        assert np.allclose(theta, np.array([
            0.        ,  5.        , 10.        , 15.        , 20.        ,
            20.        , 16.19898973, 12.39797947,  8.5969692 ,  4.79595893
        ]))
        assert np.allclose(pr, np.array([
            1.        , 1.45398306, 2.05447215, 2.82156232, 3.77125746,
            3.77125746, 4.64971828, 5.68480575, 6.90371604, 8.35255191
        ]))

        theta, pr = PressureDeflectionLocus.create_path(
            (pdl1, theta_3), (pdl3, phi), N=5)
        assert len(theta) == len(pr) == 10
        assert np.allclose(theta, np.array([
            -0.        ,  -3.75      ,  -7.5       , -11.25      ,
            -15.        , -15.        , -10.05101027,  -5.10202053,
            -0.1530308 ,   4.79595893
        ]))
        assert np.allclose(pr, np.array([
            1.        , 1.32765833, 1.734528  , 2.22996022, 2.82156232,
            2.82156232, 3.78869335, 4.99840531, 6.49434694, 8.35255191
        ]))

        theta, pr = PressureDeflectionLocus.create_path(
            (pdl1, theta_3), (pdl3, phi), N=5, concatenate=False)
        assert isinstance(theta, list)
        assert len(theta) == 2
        assert all(isinstance(t, np.ndarray) and (len(t) == 5) for t in theta)
        assert isinstance(pr, list)
        assert len(pr) == 2
        assert all(isinstance(t, np.ndarray) and (len(t) == 5) for t in pr)
        assert np.allclose(
            theta[0], np.array([ -0.  ,  -3.75,  -7.5 , -11.25, -15.  ]))
        assert np.allclose(
            theta[1], np.array([
                -15. , -10.05101027,  -5.10202053,  -0.1530308 , 4.79595893]))
        assert np.allclose(
            pr[0], np.array([
                1.        , 1.32765833, 1.734528  , 2.22996022, 2.82156232]))
        assert np.allclose(
            pr[1], np.array([
                2.82156232, 3.78869335, 4.99840531, 6.49434694, 8.35255191]))


class Test_PressureDeflectionLocus_pressure_deflection:
    def test_default_lengths(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t, p = locus.pressure_deflection(include_mirror=False)
        assert len(t) == len(p) == 200
        t, p = locus.pressure_deflection(include_mirror=True)
        assert len(t) == len(p) == 400

    def test_include_mirror_False(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t, p = locus.pressure_deflection(N=5, include_mirror=False)
        t_max = max_theta_from_mach(M, gamma)
        assert len(t) == len(p) == 10
        assert len([t for t in np.isclose(t - t_max, 0) if t]) == 2
        assert len([t for t in np.isclose(t - (-t_max), 0) if t]) == 0
        assert np.allclose(t, np.array([
            0, 20.67617858, 22.74379644, 22.95055823, 22.97353176,
            22.97353176, 22.95055823, 22.74379644, 20.67617858,  0
        ]))
        assert np.allclose(p, np.array([
            1, 2.95556551, 3.45277315, 3.5872617 , 3.64575071,
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5
        ]))

    def test_include_mirror_True(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t, p = locus.pressure_deflection(N=5, include_mirror=True)
        t_max = max_theta_from_mach(M, gamma)
        assert len(t) == len(p) == 20
        assert len([t for t in np.isclose(t - t_max, 0) if t]) == 2
        assert len([t for t in np.isclose(t - (-t_max), 0) if t]) == 2
        assert np.allclose(t, np.array([
            0.        ,  20.67617858,  22.74379644,  22.95055823,
            22.97353176,  22.97353176,  22.95055823,  22.74379644,
            20.67617858,   0.        ,  -0.        , -20.67617858,
        -22.74379644, -22.95055823, -22.97353176, -22.97353176,
        -22.95055823, -22.74379644, -20.67617858,  -0
        ]))
        assert np.allclose(p, np.array([
            1.        , 2.95556551, 3.45277315, 3.5872617 , 3.64575071,
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5       ,
            4.5       , 4.10913375, 3.81560199, 3.70192379, 3.64575192,
            3.64575071, 3.5872617 , 3.45277315, 2.95556551, 1.
        ]))

    def test_theta_origin_pr_to_freestream(self):
        gamma = 1.4
        M1 = 3
        theta_2 = 20
        theta_3 = -15

        locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
        locus2 = locus1.new_locus_from_shockwave(theta_2)
        locus3 = locus1.new_locus_from_shockwave(theta_3)

        t, pr = locus2.pressure_deflection(N=5, include_mirror=True)
        assert len(t) == len(pr) == 20
        assert np.allclose(t, np.array([
            20.        , 40.58502741, 42.64353015, 42.84938042, 42.87225267,
            42.87225267, 42.84938042, 42.64353015, 40.58502741, 20.        ,
            20.        , -0.58502741, -2.64353015, -2.84938042, -2.87225267,
            -2.87225267, -2.84938042, -2.64353015, -0.58502741, 20.
        ]))
        assert np.allclose(pr, np.array([
            3.77125752, 11.0844218 , 12.94381725, 13.44702261, 13.66591115,
            13.66591578, 13.8761607 , 14.30173635, 15.4012961 , 16.8675321 ,
            16.8675321 , 15.4012961 , 14.30173635, 13.8761607 , 13.66591578,
            13.66591115, 13.44702261, 12.94381725, 11.0844218 ,  3.77125752
        ]))

        t, pr = locus3.pressure_deflection(N=5, include_mirror=True)
        assert len(t) == len(pr) == 20
        assert np.allclose(t, np.array([
            -15.        ,   9.17472916,  11.59220207,  11.83394937,
            11.86081018,  11.86081018,  11.83394937,  11.59220207,
            9.17472916, -15.        , -15.        , -39.17472916,
            -41.59220207, -41.83394937, -41.86081018, -41.86081018,
            -41.83394937, -41.59220207, -39.17472916, -15.
        ]))
        assert np.allclose(pr, np.array([
            2.8215624 , 10.50413697, 12.45604464, 12.97283314, 13.19571904,
            13.19572119, 13.40862531, 13.83578023, 14.91214638, 16.26729013,
            16.26729013, 14.91214638, 13.83578023, 13.40862531, 13.19572119,
            13.19571904, 12.97283314, 12.45604464, 10.50413697,  2.8215624
        ]))


class Test_PressureDeflectionLocus_pressure_deflection_split_regions:
    def test_default_lengths(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        tw, pw, ts, ps = locus.pressure_deflection_split_regions(
            include_mirror=False)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 200
        tw, pw, ts, ps = locus.pressure_deflection_split_regions(
            include_mirror=True)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 400

    def test_include_mirror_False(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        tw, pw, ts, ps = locus.pressure_deflection_split_regions(
            N=5, include_mirror=False)
        t_max = max_theta_from_mach(M, gamma)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 10
        assert len([t for t in np.isclose(tw - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(ts - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(tw - (-t_max), 0) if t]) == 0
        assert len([t for t in np.isclose(ts - (-t_max), 0) if t]) == 0
        assert np.allclose(tw, np.array([
            0.        , 20.67617858, 22.74379644, 22.95055823, 22.97353176
        ]))
        assert np.allclose(ts, np.array([
            22.97353176, 22.95055823, 22.74379644, 20.67617858,  0
        ]))
        assert np.allclose(pw, np.array([
            1.        , 2.95556551, 3.45277315, 3.5872617 , 3.64575071
        ]))
        assert np.allclose(ps, np.array([
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5
        ]))

    def test_include_mirror_True(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        tw, pw, ts, ps = locus.pressure_deflection_split_regions(
            N=5, include_mirror=True)
        t_max = max_theta_from_mach(M, gamma)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 20
        assert len([t for t in np.isclose(tw - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(ts - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(tw - (-t_max), 0) if t]) == 1
        assert len([t for t in np.isclose(ts - (-t_max), 0) if t]) == 1
        assert np.allclose(tw, np.array([
            -22.97353176, -22.95055823, -22.74379644, -20.67617858,
            -0.        ,   0.        ,  20.67617858,  22.74379644,
            22.95055823,  22.97353176
        ]))
        assert np.allclose(ts, np.array([
            22.97353176,  22.95055823,  22.74379644,  20.67617858,
            0.        ,  -0.        , -20.67617858, -22.74379644,
        -22.95055823, -22.97353176
        ]))
        assert np.allclose(pw, np.array([
            3.64575071, 3.5872617 , 3.45277315, 2.95556551, 1.        ,
            1.        , 2.95556551, 3.45277315, 3.5872617 , 3.64575071
        ]))
        assert np.allclose(ps, np.array([
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5       ,
            4.5       , 4.10913375, 3.81560199, 3.70192379, 3.64575192
        ]))

    def test_theta_origin_pr_to_freestream(self):
        gamma = 1.4
        M1 = 3
        theta_2 = 20
        theta_3 = -15

        locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
        locus2 = locus1.new_locus_from_shockwave(theta_2)
        locus3 = locus1.new_locus_from_shockwave(theta_3)

        tw, pw, ts, ps = locus2.pressure_deflection_split_regions(
            N=5, include_mirror=True)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 20
        assert np.allclose(tw, np.array([
            -2.87225267, -2.84938042, -2.64353015, -0.58502741, 20.        ,
            20.        , 40.58502741, 42.64353015, 42.84938042, 42.87225267
        ]))
        assert np.allclose(pw, np.array([
            13.66591115, 13.44702261, 12.94381725, 11.0844218 ,  3.77125752,
            3.77125752, 11.0844218 , 12.94381725, 13.44702261, 13.66591115
        ]))
        assert np.allclose(ts, np.array([
            42.87225267, 42.84938042, 42.64353015, 40.58502741, 20.        ,
            20.        , -0.58502741, -2.64353015, -2.84938042, -2.87225267
        ]))
        assert np.allclose(ps, np.array([
            13.66591578, 13.8761607 , 14.30173635, 15.4012961 , 16.8675321 ,
            16.8675321 , 15.4012961 , 14.30173635, 13.8761607 , 13.66591578
        ]))

        tw, pw, ts, ps = locus3.pressure_deflection_split_regions(
            N=5, include_mirror=True)
        assert (len(ts) + len(ts)) == (len(pw) + len(ps)) == 20
        assert np.allclose(tw, np.array([
            -41.86081018, -41.83394937, -41.59220207, -39.17472916,
            -15.        , -15.        ,   9.17472916,  11.59220207,
            11.83394937,  11.86081018
        ]))
        assert np.allclose(pw, np.array([
            13.19571904, 12.97283314, 12.45604464, 10.50413697,  2.8215624 ,
            2.8215624 , 10.50413697, 12.45604464, 12.97283314, 13.19571904
        ]))
        assert np.allclose(ts, np.array([
            11.86081018,  11.83394937,  11.59220207,   9.17472916,
            -15.        , -15.        , -39.17472916, -41.59220207,
            -41.83394937, -41.86081018
        ]))
        assert np.allclose(ps, np.array([
            13.19572119, 13.40862531, 13.83578023, 14.91214638, 16.26729013,
            16.26729013, 14.91214638, 13.83578023, 13.40862531, 13.19572119
        ]))


class Test_PressureDeflectionLocus_pressure_deflection_segment:
    def test_default_lengths(self):
        M = 2
        gamma = 1.4
        theta_final = 10
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        theta, pr = locus.pressure_deflection_segment(theta_final)
        assert len(theta) == len(pr) == 100

    def test_values(self):
        M = 2
        gamma = 1.4
        theta_final = 10
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        theta, pr = locus.pressure_deflection_segment(theta_final, N=5)
        assert len(theta) == len(pr) == 5
        assert np.allclose(theta, np.array([ 0. ,  2.5,  5. ,  7.5, 10. ]))
        assert np.allclose(pr, np.array(
            [1.        , 1.14913345, 1.31540694, 1.50052358, 1.7065786 ]))

        theta, pr = locus.pressure_deflection_segment(-theta_final, N=5)
        assert len(theta) == len(pr) == 5
        assert np.allclose(theta, -np.array([ 0. ,  2.5,  5. ,  7.5, 10. ]))
        assert np.allclose(pr, np.array(
            [1.        , 1.14913345, 1.31540694, 1.50052358, 1.7065786 ]))

    def test_theta_origin_pr_to_freestream(self):
        gamma = 1.4
        M1 = 3
        theta_2 = 20
        theta_3 = -15

        locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
        locus2 = locus1.new_locus_from_shockwave(theta_2)
        locus3 = locus1.new_locus_from_shockwave(theta_3)

        theta1, pr1 = locus1.pressure_deflection_segment(theta_2, N=5)
        assert np.allclose(theta1, np.array([ 0.,  5., 10., 15., 20. ]))
        assert np.allclose(pr1, np.array(
            [1.        , 1.45398306, 2.05447215, 2.82156232, 3.77125746 ]))

        theta2, pr2 = locus2.pressure_deflection_segment(5, N=5)
        assert np.allclose(theta2, np.array([ 20.  , 21.25, 22.5 , 23.75, 25.]))
        assert np.allclose(pr2, np.array(
            [3.77125746, 4.04420615, 4.3325282 , 4.636943  , 4.95821806 ]))

        theta2, pr2 = locus2.pressure_deflection_segment(-5, N=5)
        assert np.allclose(theta2, np.array([ 20.  , 18.75, 17.5 , 16.25, 15. ]))
        assert np.allclose(pr2, np.array(
            [3.77125746, 4.04420615, 4.3325282 , 4.636943  , 4.95821806 ]))

        theta3, pr3 = locus3.pressure_deflection_segment(5, N=5)
        assert np.allclose(theta3, np.array(
            [-15.  , -13.75, -12.5 , -11.25, -10.  ]))
        assert np.allclose(pr3, np.array(
            [2.82156232, 3.04502121, 3.28216195, 3.53357527, 3.79985865 ]))

        theta3, pr3 = locus3.pressure_deflection_segment(-5, N=5)
        assert np.allclose(theta3, np.array(
            [-15.  , -16.25, -17.5 , -18.75, -20.  ]))
        assert np.allclose(pr3, np.array(
            [2.82156232, 3.04502121, 3.28216195, 3.53357527, 3.79985865 ]))


@pytest.mark.parametrize("M", [
    int(2),
    float(2),
    np.atleast_1d([2]).astype(int),
    np.atleast_1d([2]).astype(np.float64)
])
def test_max_theta_from_mach(M):
    # the results is always correct regardless of the type of the mach number
    tm = max_theta_from_mach(M, 1.4)
    assert np.isclose(tm, 22.97353176093536)


def test_shock_polar_equation():
    M1 = 5
    gamma = 1.4
    M1s = characteristic_mach_number(M1, gamma)
    res = shock_polar_equation(1.5, M1s, gamma)
    assert isinstance(res, Number)
    assert np.isclose(res, 0.8388490177635208)

    res = shock_polar_equation([1.5], M1s, gamma)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, [0.8388490177635208])

    res = shock_polar_equation([1.25, 1.5], M1s, gamma)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, [0.85788745, 0.83884902])


def test_shock_polar():
    M1 = 2
    gamma = 1.4

    Vx_as, Vy_as = shock_polar(M1, gamma, include_mirror=True, N=5)
    assert np.allclose(Vx_as, [
        1.63299316, 1.48352672, 1.1226828 , 0.76183888, 0.61237244,
        0.61237244, 0.76183888, 1.1226828 , 1.48352672, 1.63299316])
    assert np.allclose(Vy_as, [
        0.        ,  0.19935999,  0.39528471,  0.30600611,  0.,
       -0.        , -0.30600611, -0.39528471, -0.19935999, -0.])

    Vx_as, Vy_as = shock_polar(M1, gamma, include_mirror=False, N=5)
    assert np.allclose(Vx_as, [
        1.63299316, 1.48352672, 1.1226828 , 0.76183888, 0.61237244])
    assert np.allclose(Vy_as, [
        0.        , 0.19935999, 0.39528471, 0.30600611, 0.])

    gamma = 1.2
    Vx_as, Vy_as = shock_polar(M1, gamma, include_mirror=False, N=5)
    assert np.allclose(Vx_as, [
        1.77281052, 1.59579546, 1.1684433 , 0.74109114, 0.56407607])
    assert np.allclose(Vy_as, [
        0, 2.36104640e-01, 4.68140838e-01, 3.62407031e-01, 0.])

    Vx_as, Vy_as = shock_polar(M1, gamma, include_mirror=False, N=50)
    assert len(Vx_as) == len(Vy_as) == 50

    Vx_as, Vy_as = shock_polar(M1, gamma, include_mirror=True, N=50)
    assert len(Vx_as) == len(Vy_as) == 100


class Test_beta_theta_max_for_unit_mach_downstream:
    @pytest.mark.parametrize("gamma", [1+1e-05, 1+1e-03, 1.1, 1.4, 2])
    def test_unit_upstream_mach(self, gamma):
        b, t = beta_theta_max_for_unit_mach_downstream(1, gamma)
        assert isinstance(b, Number)
        assert isinstance(t, Number)
        assert np.isclose(b, 90)
        assert np.isclose(t, 0)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1 + 1e-05,
            [73.40706182, 63.92652624, 65.56716295, 71.86217534, 78.71253601,
                84.28872169, 88.84710342, 89.41291498, 89.85965519, 89.87188325],
            [1.85846746, 16.00735509, 32.25617092, 51.32787522, 66.94414137,
                78.51941942, 87.69374538, 88.82577125, 89.71931026, 89.74376653]
        ),
        (
            1 + 1e-03,
            [73.40676156, 63.92193594, 65.55335387, 71.82959956, 78.64767961,
                84.15156648, 88.28207186, 88.59717696, 88.71808132, 88.7193609],
            [1.85742536, 15.99453105, 32.22318508, 51.25815935, 66.81159448,
                78.24370602, 86.56345599, 87.19421359, 87.43616137, 87.4387218]
        ),
        (
            1.1,
            [73.3783383 , 63.4981278 , 64.3116682 , 69.04591853, 73.74360285,
                76.56380691, 77.64260708, 77.67814023, 77.68989256, 77.69001134],
            [1.75881645, 14.81068128, 29.25439142, 45.29498822, 56.78496857,
                62.98902769, 65.28019062, 65.35502876, 65.37977261, 65.38002267]
        ),
        (
            1.4,
            [73.30801228, 62.53191731, 61.69290199, 63.82469504, 66.0919319 ,
               67.33559104, 67.7736509 , 67.78766864, 67.79229892, 67.7923457],
            [ 1.51516572, 12.11266889, 22.97353176, 34.07343978, 41.1176631 ,
                44.42901938, 45.53793219, 45.57299736, 45.58457445, 45.5846914]
        ),
        (
            2,
            [73.21296438, 61.38072672, 58.8753591 , 58.78950118, 59.42142646,
                59.83982763, 59.99339243, 59.99834654, 59.99998346, 60.],
            [ 1.18660439,  8.89920691, 16.18170643, 23.19264155, 27.42418552,
                29.34281595, 29.97354327, 29.99338449, 29.99993384, 30.]
        )
    ])
    def test_gamma(self, gamma, expected_beta, expected_theta):
        M1 = [1.1, 1.5, 2, 3, 5, 10, 50, 100, 1000, 1000000000.0]
        b, t = beta_theta_max_for_unit_mach_downstream(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1+1e-05,
            [90.        , 89.42709249, 88.18973789, 84.32049763],
            [1.97247265e-06, 6.23717873e-05, 1.97133384e-03, 6.20119214e-02]
        ),
        (
            1+1e-04,
            [90.        , 89.42709249, 88.18973787, 84.32049686],
            [1.97238389e-06, 6.23689803e-05, 1.97124502e-03, 6.20090937e-02]
        ),
        (
            1+1e-03,
            [90.        , 89.42709248, 88.18973763, 84.32048917],
            [1.97149675e-06, 6.23409246e-05, 1.97035722e-03, 6.19808317e-02]
        ),
        (
            1+1e-02,
            [90.        , 89.42709241, 88.18973524, 84.32041265],
            [1.96266904e-06, 6.20617491e-05, 1.96152302e-03, 6.16996221e-02]
        ),
    ])
    def test_low_mach_numbers_low_gammas(
        self, gamma, expected_beta, expected_theta
    ):
        M1 = [1+1e-05, 1+1e-04, 1+1e-03, 1.01]
        b, t = beta_theta_max_for_unit_mach_downstream(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)


class Test_beta_theta_c_for_unit_mach_downstream:
    @pytest.mark.parametrize("gamma", [1+1e-05, 1+1e-03, 1.1, 1.4, 2])
    def test_unit_upstream_mach(self, gamma):
        b, t = beta_theta_c_for_unit_mach_downstream(1, gamma)
        assert isinstance(b, Number)
        assert isinstance(t, Number)
        assert np.isclose(b, 90)
        assert np.isclose(t, 0)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1 + 1e-05,
            [73.33568019, 63.5703523 , 65.32302086, 71.81129717, 78.70809338,
                84.28857923, 88.84710337, 89.41291499, 89.85965521, 89.87188328],
            [14.91490426, 33.51142025, 46.8052855 , 61.14084513, 72.75046478,
                81.39537235, 88.27035619, 89.11933413, 89.78948262, 89.80782482]
        ),
        (
            1 + 1e-03,
            [73.33542133, 63.5660194 , 65.30929301, 71.77866314, 78.64321346,
                84.15142065, 88.28207179, 88.59717695, 88.71808133, 88.7193609],
            [14.91102672, 33.49803714, 46.77800499, 61.08724595, 72.65042011,
                81.18829549, 87.4225385 , 87.89557266, 88.0770278 , 88.07894808]
        ),
        (
            1.1,
            [73.31090841, 63.16626139, 64.07633017, 68.99076775, 73.73748484,
                76.56348366, 77.6426066 , 77.6781402 , 77.68989256, 77.69001134],
            [14.53796586, 32.24136001, 44.29017356, 56.46024026, 65.02644011,
                69.65835236, 71.37271021, 71.42874685, 71.44727489, 71.44746213]
        ),
        (
            1.4,
            [73.25018825, 62.25682634, 61.48537164, 63.76660294, 66.08399728,
                67.33509986, 67.77365012, 67.78766859, 67.79229892, 67.7923457],
            [13.56046129, 29.19088281, 38.76391346, 47.43563417, 52.75239581,
                55.2344517 , 56.06480654, 56.09106073, 56.09972877, 56.09981634]
        ),
        (
            2,
            [73.16796426, 61.17622428, 58.71234448, 58.73683092, 59.41325286,
                59.83928575, 59.99339155, 59.99834648, 59.99998346, 60.],
            [12.0859751 , 25.09107788, 32.16239666, 37.97708512, 41.27967915,
                42.74678107, 43.22595653, 43.24100795, 43.24597594, 43.24602613]
        )
    ])
    def test_gamma(self, gamma, expected_beta, expected_theta):
        M1 = [1.1, 1.5, 2, 3, 5, 10, 50, 100, 1000, 1000000000.0]
        b, t = beta_theta_c_for_unit_mach_downstream(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1+1e-05,
            [90.        , 89.42708995, 88.18965765, 84.31798341],
            [0.        , 0.48779034, 1.53794607, 4.81421665]
        ),
        (
            1+1e-04,
            [90.        , 89.42708995, 88.18965763, 84.31798275],
            [0.        , 0.48743173, 1.53786795, 4.81410516]
        ),
        (
            1+1e-03,
            [90.        , 89.42708994, 88.18965743, 84.31797621],
            [0.        , 0.48720442, 1.53751244, 4.81303127]
        ),
        (
            1+1e-02,
            [90.        , 89.42708988, 88.1896554 , 84.31791115],
            [0.        , 0.48606956, 1.534073  , 4.80237455]
        ),
    ])
    def test_low_mach_numbers_low_gammas(
        self, gamma, expected_beta, expected_theta
    ):
        M1 = [1+1e-05, 1+1e-04, 1+1e-03, 1.01]
        b, t = beta_theta_c_for_unit_mach_downstream(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)
