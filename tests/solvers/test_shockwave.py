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
#
# 3. When needed, further tests are added to core functions in order to make
# sure they produce correct results (on edge cases too). These are found in
# tests/test_shockwave.py
#

import numpy as np
import pytest
from pygasflow.solvers.shockwave import (
    oblique_shockwave_solver as ss,
    conical_shockwave_solver as css,
    normal_shockwave_solver as nss,
    print_conical_shockwave_results,
    print_normal_shockwave_results,
    print_oblique_shockwave_results,
)


class Test_normal_shockwave_solver:
    """normal_shockwave_solver is going to call shockwave_solver, so I
    check the results of both solvers.
    """
    @pytest.mark.parametrize("param, value", [
        ("mu", 2),
        ("mnd", 0.57735026),
        ("md", 0.57735026),
        ("pressure", 4.5),
        ("temperature", 1.6875),
        ("density", 2.66666666),
        ("total_pressure", 0.72087386),
    ])
    def test_1_shockwave_solver(self, param, value):
        gamma = 1.4
        expected_res1 = [
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

        # shockwave solver with beta=90deg
        res1 = ss(param if param != "md" else "mnd", value, gamma=gamma)
        assert np.allclose(res1, expected_res1)

        expected_res2 = [
            2,          # M1
            0.57735026, # M2
            4.5,        # Pressure Ratio, P2/P1
            2.66666666, # Density Ratio, RHO2/RHO1
            1.6875,     # Temperature Ratio, T2/T1
            0.72087386, # Total Pressure Ratio, P02/P01
        ]

        res2 = nss(param, value, gamma=gamma)
        assert np.allclose(res2, expected_res2)

    @pytest.mark.parametrize("param, value", [
        ("mu", 2),
        ("mnd", 0.52522573),
        ("pressure", 4.14285714),
        ("temperature", 1.18367346),
        ("density", 3.5),
        ("total_pressure", 0.64825952),
    ])
    def test_2_shockwave_solver(self, param, value):
        gamma = 1.1
        expected_res1 = [
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

        # shockwave solver with beta=90deg
        res1 = ss(param, value, gamma=gamma)
        assert np.allclose(res1, expected_res1)

        expected_res2 = [
            2,          # M1
            0.52522573, # M2
            4.14285714, # Pressure Ratio, P2/P1
            3.5,        # Density Ratio, RHO2/RHO1
            1.18367346, # Temperature Ratio, T2/T1
            0.64825952, # Total Pressure Ratio, P02/P01
        ]

        res2 = nss(param, value, gamma=gamma)
        assert np.allclose(res2, expected_res2)

    @pytest.mark.parametrize("param, value", [
        ("mu", [2, 5]),
        ("mnd", [0.57735026, 0.41522739]),
        ("pressure", [4.5, 29]),
        ("temperature", [1.6875, 5.79999999]),
        ("density", [2.66666666, 5]),
        ("total_pressure", [0.72087386, 0.06171631]),
    ])
    def test_multiple_machs_shockwave_solver(self, param, value):
        gamma = 1.4
        expected_res1 = [
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

        res1 = ss(param, value, gamma=gamma)
        assert np.allclose(res1, expected_res1)

        expected_res2 = [
            [2, 5],                     # M1
            [0.57735026, 0.41522739],   # M2
            [4.5, 29],                  # Pressure Ratio, P2/P1
            [2.66666666, 5],            # Density Ratio, RHO2/RHO1
            [1.6875, 5.79999999],       # Temperature Ratio, T2/T1
            [0.72087386, 0.06171631],   # Total Pressure Ratio, P02/P01
        ]

        res2 = nss(param, value, gamma=gamma)
        assert np.allclose(res2, expected_res2)

    def test_solver_to_dict(self):
        gamma = 1.4
        M = 2
        r1 = nss("mu", M, gamma=gamma, to_dict=False)
        assert len(r1) == 6

        r2 = nss("mu", M, gamma=gamma, to_dict=True)
        assert len(r2) == 6
        assert isinstance(r2, dict)

        assert np.isclose(r2["mu"], r1[0])
        assert np.isclose(r2["md"], r1[1])
        assert np.isclose(r2["pr"], r1[2])
        assert np.isclose(r2["dr"], r1[3])
        assert np.isclose(r2["tr"], r1[4])
        assert np.isclose(r2["tpr"], r1[5])


class Test_oblique_shockwave_solver:
    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", 5, "theta", 20),
        ("mu", 5, "beta", 29.8009155),
        ("mu", 5, "mnu", 2.48493913),
        ("beta", 29.8009155, "theta", 20),
        ("theta", 20, "beta", 29.8009155),
    ])
    def test_1_weak_single_mach(self, param1, value1, param2, value2):
        gamma = 1.4
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

        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="weak")
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", 5, "theta", 20),
        ("mu", 5, "beta", 84.5562548),
        ("mu", 5, "mnu", 4.97744911),
        ("beta", 84.5562548, "theta", 20),
        ("theta", 20, "beta", 84.5562548),

    ])
    def test_1_strong_single_mach(self, param1, value1, param2, value2):
        gamma = 1.4
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

        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="strong")
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", 5, "theta", 20),
        ("mu", 5, "beta", 27.749065268990808),
        ("mu", 5, "mnu", 2.328000412282412),
        ("beta", 27.749065268990808, "theta", 20),
        ("theta", 20, "beta", 27.749065268990808),
    ])
    def test_2_weak_single_mach(self, param1, value1, param2, value2):
        gamma = 1.2
        expected_res = [
            5.0,
            2.328000412282412,
            3.639365388144191,
            0.4907129773582181,
            27.749065268990808,
            20.0,
            5.821366457731361,
            3.866215696475352,
            1.5057014183244946,
            0.49956474525543926
        ]
        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="weak")
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", 5, "theta", 20),
        ("mu", 5, "beta", 86.88829093968721),
        ("mu", 5, "mnu", 4.992627989196466),
        ("beta", 86.88829093968721, "theta", 20),
        ("theta", 20, "beta", 86.88829093968721),
    ])
    def test_2_strong_single_mach(self, param1, value1, param2, value2):
        gamma = 1.2
        expected_res = [
            5.0,
            4.992627989196466,
            0.37215012062223435,
            0.34228183553244257,
            86.88829093968721,
            20.0,
            27.101455532917754,
            7.850514020486018,
            3.452188667161941,
            0.016011258932630855
        ]

        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="strong")
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", [2, 5], "theta", [20, 20]),
        ("mu", [2, 5], "beta", [53.4229405, 29.8009155]),
        ("mu", [2, 5], "mnu", [1.60611226, 2.48493913]),
    ])
    def test_3_weak_multiple_machs(self, param1, value1, param2, value2):
        gamma = 1.4
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

        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="weak")
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("param1, value1, param2, value2", [
        ("mu", [2, 5], "theta", [20, 20]),
        ("mu", [2, 5], "beta", [74.2701370, 84.5562548]),
        ("mu", [2, 5], "mnu", [1.92510115, 4.97744911]),
    ])
    def test_3_strong_multiple_machs(self, param1, value1, param2, value2):
        gamma = 1.4
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

        res = ss(
            param1, value1, param2, value2, gamma=gamma, flag="strong")
        assert np.allclose(res, expected_res)

    def test_solver_to_dict(self):
        gamma = 1.4
        M = 2
        r1 = ss("mu", M, gamma=gamma, to_dict=False)
        assert len(r1) == 10

        r2 = ss("mu", M, gamma=gamma, to_dict=True)
        assert len(r2) == 10
        assert isinstance(r2, dict)

        assert np.isclose(r2["mu"], r1[0])
        assert np.isclose(r2["mnu"], r1[1])
        assert np.isclose(r2["md"], r1[2])
        assert np.isclose(r2["mnd"], r1[3])
        assert np.isclose(r2["beta"], r1[4])
        assert np.isclose(r2["theta"], r1[5])
        assert np.isclose(r2["pr"], r1[6])
        assert np.isclose(r2["dr"], r1[7])
        assert np.isclose(r2["tr"], r1[8])
        assert np.isclose(r2["tpr"], r1[9])

    @pytest.mark.parametrize("err, param1, value1, param2, value2", [
        (ValueError, "mu", 2, "asd", 4), # p2_name not in ["beta", "theta", "mn1"]
        (ValueError, "mu", 2, "beta", None), # p2_value = None
        (ValueError, "mu", 2, "beta", -10), # beta < 0
        (ValueError, "mu", [2, 4], "beta", [20, -10]), # at least one beta < 0
        (ValueError, "mu", 2, "beta", 100), # beta > 90
        (ValueError, "mu", [2, 4], "beta", [20, 100]), # at least one beta > 90
        (ValueError, "mu", 2, "theta", -10), # beta < 0
        (ValueError, "mu", [2, 4], "theta", [20, -10]), # at least one beta < 0
        (ValueError, "mu", 2, "theta", 100), # beta > 90
        (ValueError, "mu", [2, 4], "theta", [20, 100]), # at least one beta > 90
        (ValueError, "asd", 2, "beta", 4), # p1_name not in available_p1names
        (ValueError, "mnu", 2, "mnu", 4), # p1_name = p2_name
        (ValueError, "theta", 2, None, None), # p1_name = theta and beta=None
        (ValueError, "theta", -10, "beta", 4), # p1_name = theta and theta < 0
        (ValueError, "theta", 100, "beta", 4), # p1_name = theta and theta > 90
        (ValueError, "beta", 2, None, None), # p1_name = beta and theta=None
        (ValueError, "beta", -10, "theta", 4), # p1_name = theta and theta < 0
        (ValueError, "beta", 100, "theta", 4), # p1_name = theta and theta > 90
        (ValueError, "beta", 60, "theta", 45), # detachment
        (ValueError, "beta", 89, "theta", 10), # detachment
        (ValueError, "beta", 5, "theta", 10), # detachment
    ])
    def test_raises_error(
        self, err, param1, value1, param2, value2
    ):
        with pytest.raises(err):
            if param2 is not None:
                ss(param1, value1, param2, value2)
            else:
                ss(param1, value1)

    @pytest.mark.parametrize("ratio_name, ratio_value", [
        ("pressure", 4.5),
        ("density", 2.66666667),
        ("temperature", 1.6875),
        ("total_pressure", 0.72087386),
    ])
    def test_ratio_theta_1(self, ratio_name, ratio_value):
        # one ratio + one theta -> 2 solutions
        expected_res = [
            [2.06488358, 3.53991435],
            [2., 2.],
            [0.69973294, 2.32136532],
            [0.57735027, 0.57735027],
            [75.59872102, 34.40127898],
            [20., 20.],
            [4.5, 4.5],
            [2.66666667, 2.66666667],
            [1.6875, 1.6875],
            [0.72087386, 0.72087386]
        ]
        res = ss(ratio_name, ratio_value, "theta", 20, gamma=1.4)
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("ratio_name, ratio_value", [
        ("pressure", [4.5, 3.00958333]),
        ("density", [2.66666667, 2.11524765]),
        ("temperature", [1.6875, 1.42280424]),
        ("total_pressure", [0.72087386, 0.87598838]),
    ])
    def test_ratio_theta_2(self, ratio_name, ratio_value):
        # 2 ratio + one theta -> 4 solutions
        expected_res = [
            [2.06488358, 3.53991435, 1.84225683, 2.27811752],
            [2.  , 2.  , 1.65, 1.65],
            [0.69973294, 2.32136532, 0.94844812, 1.47030132],
            [0.57735027, 0.57735027, 0.65395844, 0.65395844],
            [75.59872102, 34.40127898, 63.59083094, 46.40916906],
            [20., 20., 20., 20.],
            [4.5       , 4.5       , 3.00958333, 3.00958333],
            [2.66666667, 2.66666667, 2.11524765, 2.11524765],
            [1.6875    , 1.6875    , 1.42280424, 1.42280424],
            [0.72087386, 0.72087386, 0.87598838, 0.87598838]
        ]
        res = ss(ratio_name, ratio_value, "theta", 20, gamma=1.4)
        assert np.allclose(res, expected_res)

    @pytest.mark.parametrize("ratio_name, ratio_value", [
        ("pressure", 3.00958333),
        ("density", 2.11524765),
        ("temperature", 1.42280424),
        ("total_pressure", 0.87598838),
    ])
    def test_ratio_theta_3(self, ratio_name, ratio_value):
        # 1 ratio + 2 theta -> 4 solutions
        expected_res = [
            [1.67298279, 4.94101622, 1.84225683, 2.27811752],
            [1.65, 1.65, 1.65, 1.65],
            [0.69378551, 3.95891481, 0.94844812, 1.47030132],
            [0.65395844, 0.65395844, 0.65395844, 0.65395844],
            [80.4919436 , 19.5080564 , 63.59083094, 46.40916906],
            [10., 10., 20., 20.],
            [3.00958333, 3.00958333, 3.00958333, 3.00958333],
            [2.11524765, 2.11524765, 2.11524765, 2.11524765],
            [1.42280424, 1.42280424, 1.42280424, 1.42280424],
            [0.87598838, 0.87598838, 0.87598838, 0.87598838]
        ]
        res = ss(ratio_name, ratio_value, "theta", [10, 20], gamma=1.4)
        assert np.allclose(res, expected_res)


class Test_conical_shockwave:
    def setup_method(self, method):
        self.gamma_1 = 1.4
        self.expected_res_1 = [
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

    @pytest.mark.parametrize("err, m1, param_name, value", [
        (ValueError, 2, "mcc", 1.5), # wrong parameter name
        (ValueError, 2, "mc", 2), # Mc = M1
        (ValueError, 2, "mc", 3), # Mc > M1
        (ValueError, 2, "beta", -30), # beta < 0
        (ValueError, 2, "beta", 100), # beta > 90
        (ValueError, 2, "theta_c", -30), # theta_c < 0
        (ValueError, 2, "theta_c", 100), # theta_c > 90
        (ValueError, 2, "theta_c", 45), # detachment
        (ValueError, 2, "beta", 20), # detachment
    ])
    def test_conical_shockwave_raises_error(self, err, m1, param_name, value):
        with pytest.raises(err):
            css(m1, param_name, value)

    @pytest.mark.parametrize("m1, param_name, value", [
        (5, "theta_c", 20),
        (5, "beta", 24.9785489),
        (5, "mc", 3.37200575),
    ])
    def test_single_mach(self, m1, param_name, value):
        # very high tolerances since the online calculator is using a
        # fixed-step iterative procedure, whereas I'm using bisection
        tol = 1e-01
        res = css(m1, param_name, value, gamma=self.gamma_1)
        assert np.allclose(res, self.expected_res_1, atol=tol)

    def test_solver_to_dict(self):
        gamma = 1.4

        r1 = css(5, "theta_c", 20, 1.4, to_dict=False)
        assert len(r1) == 12

        r2 = css(5, "theta_c", 20, 1.4, to_dict=True)
        assert len(r2) == 12
        assert isinstance(r2, dict)

        assert np.isclose(r2["mu"], r1[0])
        assert np.isclose(r2["mc"], r1[1])
        assert np.isclose(r2["theta_c"], r1[2])
        assert np.isclose(r2["beta"], r1[3])
        assert np.isclose(r2["delta"], r1[4])
        assert np.isclose(r2["pr"], r1[5])
        assert np.isclose(r2["dr"], r1[6])
        assert np.isclose(r2["tr"], r1[7])
        assert np.isclose(r2["tpr"], r1[8])
        assert np.isclose(r2["pc_pu"], r1[9])
        assert np.isclose(r2["rhoc_rhou"], r1[10])
        assert np.isclose(r2["Tc_Tu"], r1[11])


def test_error_for_multiple_gamma():
    err_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=err_msg):
        ss("mu", [2, 3], "beta", 80, gamma=[1.1, 2])

    with pytest.raises(ValueError, match=err_msg):
        nss("mu", [2, 3], gamma=[1.1, 2])

    with pytest.raises(ValueError, match=err_msg):
        css([2.5, 5], "mc", 1.5, gamma=[1.1, 2])


@pytest.mark.parametrize("g", [0.9, 1])
def test_error_gamma_less_equal_than_one(g):
    err_msg = "The specific heats ratio must be > 1."
    with pytest.raises(ValueError, match=err_msg):
        ss("mu", [2, 3], "beta", 80, gamma=g)

    with pytest.raises(ValueError, match=err_msg):
        nss("mu", [2, 3], gamma=g)

    with pytest.raises(ValueError, match=err_msg):
        css([2.5, 5], "mc", 1.5, gamma=g)


@pytest.mark.parametrize("to_dict", [True, False])
def test_print_normal_shockwave_results(to_dict):
    res1 = nss("mu", 4, to_dict=to_dict)
    print_normal_shockwave_results(res1)
    print_normal_shockwave_results(res1, "{:.3f}")


@pytest.mark.parametrize("to_dict", [True, False])
def test_print_oblique_shockwave_results(to_dict):
    res1 = ss("mu", 4, "theta", 15, to_dict=to_dict)
    print_oblique_shockwave_results(res1)
    print_oblique_shockwave_results(res1, "{:.3f}")


@pytest.mark.parametrize("to_dict", [True, False])
def test_print_conical_shockwave_results(to_dict):
    res1 = css(4, "theta_c", 10, to_dict=to_dict)
    print_conical_shockwave_results(res1)
    print_conical_shockwave_results(res1, "{:.3f}")
