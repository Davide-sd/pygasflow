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
    max_theta_from_mach
)
from tempfile import TemporaryDirectory


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


def test_oblique_mach_downstream():
    expected_results = ss("m1", [1.5, 3], "beta", [60, 60], to_dict=True)
    expected_mach_downstream = expected_results["m2"]
    actual_mach_downstream_results = oblique_mach_downstream(
        expected_results["m1"],
        expected_results["beta"],
        expected_results["theta"]
    )
    assert np.allclose(expected_mach_downstream, actual_mach_downstream_results)


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
        assert callable(obj.func_of_theta)
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
        f1 = p.func_of_theta
        tm1 = p.theta_max

        p.M = 3
        f2 = p.func_of_theta
        tm2 = p.theta_max
        assert not np.isclose(tm1, tm2)
        assert id(f1) != id(f2)

        p.gamma = 1.2
        f3 = p.func_of_theta
        tm3 = p.theta_max
        assert not np.isclose(tm2, tm3)
        assert id(f2) != id(f3)

        p.theta_origin = 10
        f4 = p.func_of_theta
        tm4 = p.theta_max
        assert np.isclose(tm3, tm4)
        assert id(f3) != id(f4)

        p.pr_to_freestream = 3
        f5 = p.func_of_theta
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

