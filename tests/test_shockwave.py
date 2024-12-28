
import numpy as np
import os
import pytest
from pygasflow.generic import characteristic_mach_number
from pygasflow.shockwave import (
    oblique_mach_downstream,
    detachment_point_oblique_shock,
    load_data,
    create_mach_beta_theta_c_csv_file,
    PressureDeflectionLocus,
    max_theta_from_mach,
    shock_polar_equation,
    shock_polar,
    sonic_point_conical_shock,
    mach_from_theta_beta,
    sonic_point_oblique_shock,
    theta_from_mach_beta,
    mach_beta_from_theta_ratio,
    m1_from_rayleigh_pitot_pressure_ratio,
)
from pygasflow.solvers.shockwave import oblique_shockwave_solver as ss
from tempfile import TemporaryDirectory
from numbers import Number


class Test_oblique_mach_downstream:
    @pytest.mark.parametrize("beta, expected_mach_downstream", [
        (60, [1.04454822, 1.12256381]),
        ([60, 30], [1.04454822, 2.36734555])
    ])
    def test_input_beta(self, beta, expected_mach_downstream):
        M1 = [1.5, 3]
        solver_results = ss(
            "mu", M1, "beta", beta, to_dict=True)
        actual_mach_downstream = oblique_mach_downstream(
            M1,
            beta=beta,
        )
        assert np.allclose(solver_results["md"], expected_mach_downstream)
        assert np.allclose(actual_mach_downstream, expected_mach_downstream)

    @pytest.mark.parametrize("theta, flag, expected_mach_downstream", [
        (10, "weak", [2.50500068, 3.99916193]),
        (10, "strong", [0.48924158, 0.42546429]),
        ([10, 20], "weak", [2.50500068, 3.02215165]),
        ([10, 20], "strong", [0.48924158, 0.46018705])
    ])
    def test_input_theta(self, theta, flag, expected_mach_downstream):
        M1 = [3, 5]
        solver_results = ss(
            "mu", M1, "theta", theta, to_dict=True, flag=flag)
        actual_mach_downstream = oblique_mach_downstream(
            M1,
            theta=theta,
            flag=flag
        )
        assert np.allclose(solver_results["md"], expected_mach_downstream)
        assert np.allclose(actual_mach_downstream, expected_mach_downstream)

    def test_errors(self):
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



@pytest.mark.parametrize("M1, gamma, expected_beta, expected_theta_max", [
    (2.5, 1.4, 64.78216996529343, 29.79744066429195),
    ([2.5, 3.5], 1.4, [64.78216997, 65.68861403], [29.79744066, 36.86701013])
])
def test_detachment_point_oblique_shock(
    M1, gamma, expected_beta, expected_theta_max
):
    beta, theta_max = detachment_point_oblique_shock(M1, gamma)
    assert np.allclose(beta, expected_beta)
    assert np.allclose(theta_max, expected_theta_max)
    t = Number if isinstance(expected_beta, Number) else np.ndarray
    assert isinstance(beta, t)
    assert isinstance(theta_max, t)



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
        self, obj, M, gamma, theta_origin, pr_to_fs_at_origin, label
    ):
        assert obj.M == M
        assert obj.gamma == gamma
        assert obj.theta_origin == theta_origin
        assert obj.pr_to_fs_at_origin == pr_to_fs_at_origin
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
            M=2, gamma=1.4, theta_origin=10, pr_to_fs_at_origin=3, label="test")
        self._do_test_instantion(p, 2, 1.4, 10, 3, "test")

    def test_new_locus_from_shockwave(self):
        pdl1 = PressureDeflectionLocus(M=3, label="1")
        pdl2 = pdl1.new_locus_from_shockwave(20, label="2")
        self._do_test_instantion(
            pdl2, 1.9941316655645605, 1.4, 20, 3.771257463082658, "2")

        pdl1 = PressureDeflectionLocus(M=3, label="1", gamma=1.2)
        pdl2 = pdl1.new_locus_from_shockwave(20, label="2")
        assert np.isclose(pdl2.gamma, 1.2)

    def test_update_func(self):
        p = PressureDeflectionLocus(M=2)
        f1 = p.shockwave_at_theta
        tm1 = p.theta_max
        sp1 = p.sonic_point
        dp1 = p.detachment_point

        p.M = 3
        f2 = p.shockwave_at_theta
        tm2 = p.theta_max
        sp2 = p.sonic_point
        dp2 = p.detachment_point
        assert not np.isclose(tm1, tm2)
        assert id(f1) != id(f2)
        assert not np.allclose(sp1, sp2)
        assert not np.allclose(dp1, dp2)

        p.gamma = 1.2
        f3 = p.shockwave_at_theta
        tm3 = p.theta_max
        sp3 = p.sonic_point
        dp3 = p.detachment_point
        assert not np.isclose(tm2, tm3)
        assert id(f2) != id(f3)
        assert not np.allclose(sp2, sp3)
        assert not np.allclose(dp2, dp3)

        p.theta_origin = 10
        f4 = p.shockwave_at_theta
        tm4 = p.theta_max
        sp4 = p.sonic_point
        dp4 = p.detachment_point
        assert np.isclose(tm3, tm4)
        assert id(f3) != id(f4)
        assert not np.allclose(sp3, sp4)
        assert not np.allclose(dp3, dp4)

        p.pr_to_fs_at_origin = 3
        f5 = p.shockwave_at_theta
        tm5 = p.theta_max
        sp5 = p.sonic_point
        dp5 = p.detachment_point
        assert np.isclose(tm4, tm5)
        assert id(f4) != id(f5)
        assert not np.allclose(sp4, sp5)
        assert not np.allclose(dp4, dp5)

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

    def test_theta_origin_pr_to_fs_at_origin(self):
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


class Test_PressureDeflectionLocus_pressure_deflection_split:
    @pytest.mark.parametrize("mode", ["region", "sonic"])
    def test_default_lengths(self, mode):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t1, p1, t2, p2 = locus.pressure_deflection_split(
            include_mirror=False, mode=mode)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 200
        t1, p1, t2, p2 = locus.pressure_deflection_split(
            include_mirror=True, mode=mode)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 400

    def test_include_mirror_False_mode_region(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t1, p1, t2, p2 = locus.pressure_deflection_split(
            N=5, include_mirror=False, mode="region")
        t_max = max_theta_from_mach(M, gamma)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 10
        assert len([t for t in np.isclose(t1 - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(t2 - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(t1 - (-t_max), 0) if t]) == 0
        assert len([t for t in np.isclose(t2 - (-t_max), 0) if t]) == 0
        assert np.allclose(t1, np.array([
            0.        , 20.67617858, 22.74379644, 22.95055823, 22.97353176
        ]))
        assert np.allclose(t2, np.array([
            22.97353176, 22.95055823, 22.74379644, 20.67617858,  0
        ]))
        assert np.allclose(p1, np.array([
            1.        , 2.95556551, 3.45277315, 3.5872617 , 3.64575071
        ]))
        assert np.allclose(p2, np.array([
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5
        ]))

    def test_include_mirror_False_mode_sonic(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t1, p1, t2, p2 = locus.pressure_deflection_split(
            N=10, include_mirror=False, mode="sonic")
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 20
        assert np.allclose(t1, np.array([
            0.        , 14.71727253, 20.00638604, 21.90719488, 22.59031015
        ]))
        assert np.allclose(t2, np.array([
            22.83580906, 22.92403679, 22.95574419, 22.96713924, 22.97353176,
            22.97353176, 22.96713924, 22.95574419, 22.92403679, 22.83580906,
            22.59031015, 21.90719488, 20.00638604, 14.71727253,  0.
        ]))
        assert np.allclose(p1, np.array([
            1.        , 2.16381844, 2.8438803 , 3.20069968, 3.39207343
        ]))
        assert np.allclose(p2, np.array([
            3.49839081, 3.5590983 , 3.59440819, 3.61518838, 3.64575071,
            3.64575192, 3.67566939, 3.69530028, 3.72741343, 3.77923555,
            3.86090466, 3.98428551, 4.15660723, 4.36144866, 4.5
        ]))

    def test_include_mirror_True(self):
        M = 2
        gamma = 1.4
        locus = PressureDeflectionLocus(M=M, gamma=gamma)
        t1, p1, t2, p2 = locus.pressure_deflection_split(
            N=5, include_mirror=True)
        t_max = max_theta_from_mach(M, gamma)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 20
        assert len([t for t in np.isclose(t1 - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(t2 - t_max, 0) if t]) == 1
        assert len([t for t in np.isclose(t1 - (-t_max), 0) if t]) == 1
        assert len([t for t in np.isclose(t2 - (-t_max), 0) if t]) == 1
        assert np.allclose(t1, np.array([
            -22.97353176, -22.95055823, -22.74379644, -20.67617858,
            -0.        ,   0.        ,  20.67617858,  22.74379644,
            22.95055823,  22.97353176
        ]))
        assert np.allclose(t2, np.array([
            22.97353176,  22.95055823,  22.74379644,  20.67617858,
            0.        ,  -0.        , -20.67617858, -22.74379644,
        -22.95055823, -22.97353176
        ]))
        assert np.allclose(p1, np.array([
            3.64575071, 3.5872617 , 3.45277315, 2.95556551, 1.        ,
            1.        , 2.95556551, 3.45277315, 3.5872617 , 3.64575071
        ]))
        assert np.allclose(p2, np.array([
            3.64575192, 3.70192379, 3.81560199, 4.10913375, 4.5       ,
            4.5       , 4.10913375, 3.81560199, 3.70192379, 3.64575192
        ]))

    def test_theta_origin_pr_to_fs_at_origin(self):
        gamma = 1.4
        M1 = 3
        theta_2 = 20
        theta_3 = -15

        locus1 = PressureDeflectionLocus(M=M1, gamma=gamma)
        locus2 = locus1.new_locus_from_shockwave(theta_2)
        locus3 = locus1.new_locus_from_shockwave(theta_3)

        t1, p1, t2, p2 = locus2.pressure_deflection_split(
            N=5, include_mirror=True)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 20
        assert np.allclose(t1, np.array([
            -2.87225267, -2.84938042, -2.64353015, -0.58502741, 20.        ,
            20.        , 40.58502741, 42.64353015, 42.84938042, 42.87225267
        ]))
        assert np.allclose(p1, np.array([
            13.66591115, 13.44702261, 12.94381725, 11.0844218 ,  3.77125752,
            3.77125752, 11.0844218 , 12.94381725, 13.44702261, 13.66591115
        ]))
        assert np.allclose(t2, np.array([
            42.87225267, 42.84938042, 42.64353015, 40.58502741, 20.        ,
            20.        , -0.58502741, -2.64353015, -2.84938042, -2.87225267
        ]))
        assert np.allclose(p2, np.array([
            13.66591578, 13.8761607 , 14.30173635, 15.4012961 , 16.8675321 ,
            16.8675321 , 15.4012961 , 14.30173635, 13.8761607 , 13.66591578
        ]))

        t1, p1, t2, p2 = locus3.pressure_deflection_split(
            N=5, include_mirror=True)
        assert (len(t1) + len(t2)) == (len(p1) + len(p2)) == 20
        assert np.allclose(t1, np.array([
            -41.86081018, -41.83394937, -41.59220207, -39.17472916,
            -15.        , -15.        ,   9.17472916,  11.59220207,
            11.83394937,  11.86081018
        ]))
        assert np.allclose(p1, np.array([
            13.19571904, 12.97283314, 12.45604464, 10.50413697,  2.8215624 ,
            2.8215624 , 10.50413697, 12.45604464, 12.97283314, 13.19571904
        ]))
        assert np.allclose(t2, np.array([
            11.86081018,  11.83394937,  11.59220207,   9.17472916,
            -15.        , -15.        , -39.17472916, -41.59220207,
            -41.83394937, -41.86081018
        ]))
        assert np.allclose(p2, np.array([
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

    def test_theta_origin_pr_to_fs_at_origin(self):
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


def test_pressure_deflection_locus_example_4_10():
    # Example 4.10 from "Modern Compressible Flow, Anderson"

    gamma = 1.4
    M1 = 2.8
    theta1 = 16  # deg (positive because it creates a left running shock wave)
    T1 = 519     # °R
    p1 = 1       # atm

    l1 = PressureDeflectionLocus(M=M1, gamma=gamma, label="1")
    l2 = l1.new_locus_from_shockwave(theta1, label="2")

    res1 = l1.flow_quantities_at_locus_origin(p1, T1, None)
    assert np.isclose(res1["M"], 2.8)
    assert np.isclose(res1["T"], 519)
    assert np.isclose(res1["p"], 1)
    assert np.isnan(res1["rho"])
    assert np.isclose(res1["p0"], 27.13829555269978)
    assert np.isclose(res1["T0"], 1332.7919999999997)
    assert np.isnan(res1["rho0"])

    res2 = l2.flow_quantities_at_locus_origin(p1, T1, None)
    assert np.isclose(res2["M"], 2.0585267920301744)
    assert np.isclose(res2["T"], 721.4004347373847)
    assert np.isclose(res2["p"], 2.830893893824571)
    assert np.isnan(res2["rho"])
    assert np.isclose(res2["p0"], 24.264676915994347)
    assert np.isclose(res2["T0"], 1332.7919999999997)
    assert np.isnan(res2["rho0"])

    res3 = l2.flow_quantities_after_shockwave(0, p1, T1, None)
    assert np.isclose(res3["M"], 1.457847510395878)
    assert np.isclose(res3["T"], 935.2507108767023)
    assert np.isclose(res3["p"], 6.607497448853544)
    assert np.isnan(res3["rho"])
    assert np.isclose(res3["p0"], 22.82743737842192)
    assert np.isclose(res3["T0"], 1332.7919999999997)
    assert np.isnan(res3["rho0"])

    shock1 = l1.shockwave_at_theta(theta1, region="weak")
    assert np.isclose(shock1["theta"], theta1)
    assert np.isclose(shock1["beta"], 34.9226304011263)

    shock2 = l2.shockwave_at_theta(0, region="weak")
    assert np.isclose(shock2["theta"], -theta1)
    assert np.isclose(shock2["beta"], 45.33424941323747)


def test_pressure_deflection_locus_intersection_shocks_opposite_families():
    # This example reproduces Figure 4.24 from
    # "Modern Compressible Flow, Anderson", using data found in
    # http://mae-nas.eng.usu.edu/MAE_5420_Web/section9/section9.1.pdf

    M1 = 3
    theta_2 = 20
    theta_3 = -15

    l1 = PressureDeflectionLocus(M=M1, label="1")
    l2 = l1.new_locus_from_shockwave(theta_2, label="2")
    l3 = l1.new_locus_from_shockwave(theta_3, label="3")

    phi, p4_p1 = l2.intersection(l3)
    theta_4 = phi - l3.theta_origin
    theta_4p = phi - l2.theta_origin

    res1 = l1.flow_quantities_at_locus_origin(None, None, None)
    res2 = l2.flow_quantities_at_locus_origin(None, None, None)
    res3 = l3.flow_quantities_at_locus_origin(None, None, None)
    res4p = l2.flow_quantities_after_shockwave(phi, None, None, None)
    res4 = l3.flow_quantities_after_shockwave(phi, None, None, None)

    assert np.isclose(phi, 4.795958931693682)
    assert np.isclose(theta_4, 19.795958931693683)
    assert np.isclose(theta_4p, -15.204041068306317)
    assert np.isclose(res1["M"], 3)
    assert np.isclose(res2["M"], 1.9941316655645605)
    assert np.isclose(res3["M"], 2.25490231226494)
    assert np.isclose(res4["M"], 1.4605040215769651)
    assert np.isclose(res4p["M"], 1.431700680823202)

    shock_A = l1.shockwave_at_theta(theta_2)
    shock_B = l1.shockwave_at_theta(theta_3)
    shock_C = l2.shockwave_at_theta(phi)
    shock_D = l3.shockwave_at_theta(phi)
    assert np.isclose(shock_A["theta"], 20)
    assert np.isclose(shock_A["beta"], 37.76363414837576)
    assert np.isclose(shock_B["theta"], -15)
    assert np.isclose(shock_B["beta"], 32.24040018274467)
    assert np.isclose(shock_C["theta"], -15.204041068306317)
    assert np.isclose(shock_C["beta"], 45.763301527634496)
    assert np.isclose(shock_D["theta"], 19.795958931693683)
    assert np.isclose(shock_D["beta"], 46.5550175823021)


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


# NOTE: this is a very slow test
# NOTE: as gamma -> 1, shockwave.total_pressure_ratio goes to overflow.
# This is computed by shockwave_solver in order to double check that
# the provided M1,beta/theta_c gives sonic condition downstream of the
# shock wave. Let's ignore these warnings.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class Test_sonic_point_conical_shock:
    def cross_check_results(self, M1, beta, theta_c, gamma):
        # verify that M1, beta, theta_c give sonic condition downstream
        # of the shock wave
        M1 = np.atleast_1d(M1)
        theta_c = np.atleast_1d(theta_c)
        beta = np.atleast_1d(beta)

        for m1, b, t in zip(M1, beta, theta_c):
            res = ss("mu", m1, "beta", b, gamma=gamma, to_dict=True)
            assert np.isclose(res["mu"], m1)
            assert np.isclose(res["md"], 1)

    @pytest.mark.parametrize("gamma", [1+1e-05, 1+1e-03, 1.1, 1.4, 2])
    def test_unit_upstream_mach(self, gamma):
        b, t = sonic_point_conical_shock(1, gamma)
        assert isinstance(b, Number)
        assert isinstance(t, Number)
        assert np.isclose(b, 90)
        assert np.isclose(t, 0)
        self.cross_check_results(1, 90, 0, gamma)

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
        b, t = sonic_point_conical_shock(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)
        self.cross_check_results(M1, b, t, gamma)

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
        b, t = sonic_point_conical_shock(M1, gamma)
        assert np.allclose(b, expected_beta)
        assert np.allclose(t, expected_theta)
        self.cross_check_results(M1, b, t, gamma)


class Test_mach_from_theta_beta:
    @pytest.mark.parametrize("theta, beta, gamma, expected_mach", [
        (20.0, 29.80091552915533, 1.4, 5.0),
        (20.0, 84.55625485091414, 1.4, 5.0),
        (20.0, 27.749065268990808, 1.2, 5.0),
        (20.0, 86.88829093968721, 1.2, 5.0),
        (20.0, 53.42294052722866, 1.4, 2.0),
        (20.0, 74.27013704294903, 1.4, 2.0),
        (20.0, 49.55258242069359, 1.2, 2.0),
        (20.0, 78.79049726401232, 1.2, 2.0)
    ])
    def test_scalar_values(self, theta, beta, gamma, expected_mach):
        assert np.isclose(
            mach_from_theta_beta(theta, beta, gamma), expected_mach)

    @pytest.mark.parametrize("theta, beta", [
        ([10, 15], 40),
        (10, [40, 60]),
    ])
    def test_shape_error(self, theta, beta):
        with pytest.raises(
            ValueError,
            match="Flow deflection angle and Shock wave angle must have the same shape"
        ):
            mach_from_theta_beta(theta, beta, 1.4)

    @pytest.mark.parametrize("theta, beta, gamma", [
        (40, 20, 1.4),
        (40, 85, 1.4),
        (30, 20, 1.2),
        (50, 85, 1.2),
        ([10, 30], [20, 20], 1.4),
        ([40, 40], [40, 80], 1.4)
    ])
    def test_detachment_error(self, theta, beta, gamma):
        with pytest.raises(
            ValueError,
            match="There is no solution for the current choice of"
        ):
            mach_from_theta_beta(theta, beta, gamma)

    @pytest.mark.parametrize("theta, beta, gamma, expected_machs", [
        ([10, 15], [40, 60], 1.4, [1.96679662, 1.64484532]),
        ([10, 15], [40, 60], 1.2, [1.91952769, 1.57854825]),
        ([20, 35], [80, 80], 1.4, [2.48860574, 9.43109882]),
        ([20, 50], [80, 80], 1.2, [2.09045284, 9.27623392]),
        ([8, 30], [80, 80], 1.4, [1.51167029, 4.44585071])
    ])
    def test_arrays(self, theta, beta, gamma, expected_machs):
        assert np.allclose(
            mach_from_theta_beta(theta, beta, gamma),
            expected_machs
        )

    @pytest.mark.parametrize("theta, beta, gamma, expected_mach", [
        (0, 0, 1.4, np.inf),
        (0, 0, 1.2, np.inf),
    ])
    def test_particular_cases_1(self, theta, beta, gamma, expected_mach):
        with pytest.warns(
            UserWarning,
            match="WARNING: detachment detected in at least one element of the flow turning angle theta array. Be careful!"
        ):
            assert np.isclose(
                mach_from_theta_beta(theta, beta, gamma),
                expected_mach
            )

    @pytest.mark.parametrize("theta, beta, gamma, expected_mach", [
        (0, 90, 1.4, 1),
        (0, 90, 1.2, 1),
    ])
    def test_particular_cases_2(self, theta, beta, gamma, expected_mach):
        assert np.isclose(
            mach_from_theta_beta(theta, beta, gamma),
            expected_mach
        )


# # NOTE: as gamma -> 1, shockwave.total_pressure_ratio goes to overflow.
# # This is computed by shockwave_solver in order to double check that
# # the provided M1,beta/theta_c gives sonic condition downstream of the
# # shock wave. Let's ignore these warnings.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class Test_sonic_point_oblique_shock:
    def cross_check_results(self, M1, beta, theta, gamma):
        M1 = np.atleast_1d(M1)
        theta = np.atleast_1d(theta)
        beta = np.atleast_1d(beta)
        for m1, b, t in zip(M1, beta, theta):
            res = ss("beta", b, "theta", t, gamma=gamma, to_dict=True)
            assert np.isclose(res["mu"], m1)
            assert np.isclose(res["md"], 1)

    @pytest.mark.parametrize("gamma", [1+1e-05, 1+1e-03, 1.1, 1.4, 2])
    def test_unit_upstream_mach(self, gamma):
        b, t = sonic_point_oblique_shock(1, gamma)
        assert isinstance(b, Number)
        assert isinstance(t, Number)
        assert np.isclose(b, 90)
        assert np.isclose(t, 0)
        self.cross_check_results(1, 90, 0, gamma)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1 + 1e-05,
            [73.33568019, 63.5703523 , 65.32302086, 71.81129717, 78.70809338,
                84.28857923, 88.84710337, 89.41291499, 89.85965521, 89.87188199],
            [1.7230721 , 15.45597952, 31.93862571, 51.27074408, 66.93951399,
                78.51927552, 87.69374534, 88.82577126, 89.71931027, 89.74376399]
        ),
        (
            1 + 1e-03,
            [73.33542133, 63.5660194 , 65.30929301, 71.77866314, 78.64321346,
                84.15142065, 88.28207179, 88.59717695, 88.71808133, 88.71936077],
            [1.72211136, 15.44358114, 31.90575454, 51.20096317, 66.80694262,
                78.24355871, 86.56345592, 87.19421358, 87.43616137, 87.43872154]
        ),
        (
            1.1,
            [73.31090841, 63.16626139, 64.07633017, 68.99076775, 73.73748484,
                76.56348366, 77.6426066 , 77.6781402 , 77.68989256, 77.69001132],
            [1.63117351, 14.2993171 , 28.94911397, 45.23309775, 56.77859665,
                62.98870117, 65.28019014, 65.35502873, 65.37977261, 65.38002265]
        ),
        (
            1.4,
            [73.25018825, 62.25682634, 61.48537164, 63.76660294, 66.08399728,
                67.33509986, 67.77365012, 67.78766859, 67.79229892, 67.7923457],
            [ 1.40624387, 11.69333282, 22.70598675, 34.0083453 , 41.10940077,
                44.42852324, 45.53793141, 45.57299731, 45.58457445, 45.58469139]
        ),
        (
            2,
            [73.16796426, 61.17622428, 58.71234448, 58.73683092, 59.41325286,
                59.83928575, 59.99339155, 59.99834648, 59.99998346, 60.],
            [ 1.10240089,  8.59162051, 15.97315802, 23.1337368 , 27.4156766 ,
                29.34226861, 29.97354239, 29.99338444, 29.99993384, 29.99999999]
        )
    ])
    def test_gamma(self, gamma, expected_beta, expected_theta):
        M1 = [1.1, 1.5, 2, 3, 5, 10, 50, 100, 1000, 1e05]
        beta, theta = sonic_point_oblique_shock(M1, gamma)
        assert np.allclose(beta, expected_beta)
        assert np.allclose(theta, expected_theta)
        self.cross_check_results(M1, beta, theta, gamma)

    @pytest.mark.parametrize("gamma, expected_beta, expected_theta", [
        (
            1+1e-05,
            [90.        , 89.42708995, 88.18965765, 84.31798341],
            [0., 5.72926283e-05, 1.81093687e-03, 5.70097128e-02]
        ),
        (
            1+1e-04,
            [90.        , 89.42708995, 88.18965763, 84.31798275],
            [0., 5.72900499e-05, 1.81085528e-03, 5.70071154e-02]
        ),
        (
            1+1e-03,
            [90.        , 89.42708994, 88.18965743, 84.31797621],
            [0., 5.72642791e-05, 1.81003978e-03, 5.69811540e-02]
        ),
        (
            1+1e-02,
            [90.        , 89.42708988, 88.1896554 , 84.31791115],
            [0., 5.70078400e-05, 1.80192505e-03, 5.67228352e-02]
        ),
    ])
    def test_low_mach_numbers_low_gammas(
        self, gamma, expected_beta, expected_theta
    ):
        M1 = [1+1e-05, 1+1e-04, 1+1e-03, 1.01]
        beta, theta = sonic_point_oblique_shock(M1, gamma)
        assert np.allclose(beta, expected_beta)
        assert np.allclose(theta, expected_theta)
        self.cross_check_results(M1, beta, theta, gamma)

    def test_scalar_arguments(self):
        M1 = 1.5
        gamma = 1.4
        beta, theta = sonic_point_oblique_shock(M1, gamma)
        assert np.isclose(beta, 62.256826342333696)
        assert np.isclose(theta, 11.693332822369735)
        self.cross_check_results(M1, beta, theta, gamma)


class Test_theta_from_mach_beta:
    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        (2, 40, 1.4, 10.62290962494955),
        (2, 84, 1.4, 9.566153624055113),
        (5, 20, 1.4, 10.665388381092495),
        (5, 88, 1.4, 7.902247906959903),
        (2, 50, 1.2, 20.321237023814874),
        (2, 79, 1.2, 19.746525198757517)
    ])
    def test_scalar_values(self, M1, beta, gamma, expected_theta):
        theta = theta_from_mach_beta(M1, beta, gamma)
        assert np.isclose(theta, expected_theta)
        assert isinstance(theta, Number)

    @pytest.mark.parametrize("M1, beta, gamma", [
        (2, 20, 1.2),
        (5, 7, 1.4),
        (1.5, 40, 1.4),
    ])
    def test_scalar_values_detachment(self, M1, beta, gamma):
        with pytest.warns(
            UserWarning,
            match="WARNING: detachment detected in at least one element of the flow turning angle theta array. Be careful!"
        ):
            theta = theta_from_mach_beta(M1, beta, gamma)
            assert np.isnan(theta)

    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        ([2, 3, 5], 40, 1.4, [10.62290962, 21.84610158, 28.27500465]),
        ([2, 3, 5], 80, 1.4, [14.80738932, 23.92671996, 31.25367499]),
        ([2, 3, 5], 40, 1.2, [11.72688803, 24.2824173 , 31.46189435]),
        ([2, 3, 5], 80, 1.2, [18.46348799, 32.11244931, 43.9374662 ]),
    ])
    def test_m1_array_beta_scalar(self, M1, beta, gamma, expected_theta):
        theta = theta_from_mach_beta(M1, beta, gamma)
        assert np.allclose(theta, expected_theta)

    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        (2, [40, 80], 1.2, [11.72688803, 18.46348799]),
        (2, [40, 80], 1.4, [10.62290962, 14.80738932]),
    ])
    def test_m1_scalar_beta_array(self, M1, beta, gamma, expected_theta):
        theta = theta_from_mach_beta(M1, beta, gamma)
        assert np.allclose(theta, expected_theta)

    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        ([2, 3], [40, 80], 1.2, [11.72688803, 32.11244931]),
        ([2, 3], [40, 80], 1.4, [10.62290962, 23.92671996]),
    ])
    def test_m1_array_beta_array(self, M1, beta, gamma, expected_theta):
        theta = theta_from_mach_beta(M1, beta, gamma)
        assert np.allclose(theta, expected_theta)

    def test_m1_array_beta_array_wrong_shape(self):
        with pytest.raises(ValueError):
            theta_from_mach_beta([2, 3, 4], [40, 80], 1.4)

    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        ([2, 3], [20, 80], 1.4, [np.nan, 23.92671996]),
        ([2, 3], [20, 10], 1.4, [np.nan, np.nan]),
    ])
    def test_array_values_detachment(self, M1, beta, gamma, expected_theta):
        with pytest.warns(
            UserWarning,
            match="WARNING: detachment detected in at least one element of the flow turning angle theta array. Be careful!"
        ):
            theta = theta_from_mach_beta(M1, beta, gamma)
            assert np.allclose(theta, expected_theta, equal_nan=True)

    @pytest.mark.parametrize("M1, beta, gamma, expected_theta", [
        (2, 0, 1.4, np.nan),
        ([2, 3], 0, 1.4, [np.nan, np.nan]),
        (2, [0, 0], 1.4, [np.nan, np.nan]),
    ])
    def test_particular_cases(self, M1, beta, gamma, expected_theta):
        # the case beta=0 should only raise the detachment warning
        with pytest.warns(
            UserWarning,
            match="WARNING: detachment detected in at least one element of the flow turning angle theta array. Be careful!"
        ):
            theta = theta_from_mach_beta(M1, beta, gamma)
            assert np.allclose(theta, expected_theta, equal_nan=True)


class Test_mach_beta_from_theta_ratio:
    @pytest.mark.parametrize("ratio_name", ["pr", "dr", "tr", "tpr"])
    def test_ratio_name_error(self, ratio_name):
        with pytest.raises(
            ValueError,
            match="`ratio_name` must be one of the following"
        ):
            mach_beta_from_theta_ratio(20, ratio_name, 2, 1.4)

    @pytest.mark.parametrize("theta, ratio_name, ratio_value, gamma", [
        (20, "pressure", 1.65, 1.4),
    ])
    def test_no_solutions(self, theta, ratio_name, ratio_value, gamma):
        with pytest.raises(
            ValueError,
            match="There is no solution for the current choice of parameters."
        ):
            mach_beta_from_theta_ratio(theta, ratio_name, ratio_value, gamma)

    @pytest.mark.parametrize(
        "theta, ratio_name, ratio_val, gamma, expected_Mu, expected_beta, regions", [
        # case 1: one sol in the strong branch, another in the weak branch
        (20, "pressure", 5.211572502219574, 1.4,
            [2.1988953952848878, 4.000000000000945],
            [77.53610314972948, 32.46389685026701],
            ["strong", "weak"]),
        (20, "density", 2.8782256018884964, 1.4,
            [2.1988953952848878, 4.000000000000945],
            [77.53610314972948, 32.46389685026701],
            ["strong", "weak"]),
        (20, "temperature", 1.8106893701453053, 1.4,
            [2.1988953952848878, 4.000000000000945],
            [77.53610314972948, 32.46389685026701],
            ["strong", "weak"]),
        (20, "total_pressure", 0.6524015014542756, 1.4,
            [2.1988953952848878, 4.000000000000945],
            [77.53610314972948, 32.46389685026701],
            ["strong", "weak"]),
        # case 1: one sol in the strong branch, another in the weak branch
        (10, "pressure", 1.8758095561704022, 1.2,
            [1.4009961724797741, 3],
            [73.41247140444955, 26.587528595486877],
            ["strong", "weak"]),
        (10, "density", 1.6801976624069386, 1.2,
            [1.4009961724797741, 3],
            [73.41247140444955, 26.587528595486877],
            ["strong", "weak"]),
        (10, "temperature", 1.1164219532856885, 1.2,
            [1.4009961724797741, 3],
            [73.41247140444955, 26.587528595486877],
            ["strong", "weak"]),
        (10, "total_pressure", 0.9687652160781292, 1.2,
            [1.4009961724797741, 3],
            [73.41247140444955, 26.587528595486877],
            ["strong", "weak"]),
        # case 2: two weak solutions
        (20, "pressure", 3.009583333333333, 1.4,
            [1.8422568289764576, 2.278117524145113],
            [63.59083093945838, 46.409169060460385],
            ["weak", "weak"]),
        (20, "density", 2.1152476529621236, 1.4,
            [1.8422568289764576, 2.278117524145113],
            [63.59083093945838, 46.409169060460385],
            ["weak", "weak"]),
        (20, "temperature", 1.4228042419140903, 1.4,
            [1.8422568289764576, 2.278117524145113],
            [63.59083093945838, 46.409169060460385],
            ["weak", "weak"]),
        (20, "total_pressure", 0.8759883765270569, 1.4,
            [1.8422568289764576, 2.278117524145113],
            [63.59083093945838, 46.409169060460385],
            ["weak", "weak"]),
    ])
    def test_single_values(
        self, theta, ratio_name, ratio_val, gamma,
        expected_Mu, expected_beta, regions
    ):
        ratio_map = {
            "pressure": "pr",
            "density": "dr",
            "temperature": "tr",
            "total_pressure": "tpr",
        }
        Mu, beta = mach_beta_from_theta_ratio(
            theta, ratio_name, ratio_val, gamma)
        assert np.allclose(Mu, expected_Mu)
        assert np.allclose(beta, expected_beta)
        # cross-validate with the results computed by the solver with
        # standard input, parameters: upstream Mach number and theta
        res1 = ss("mu", Mu[0], "theta", theta, gamma=gamma,
            flag=regions[0], to_dict=True)
        res2 = ss("mu", Mu[1], "theta", theta, gamma=gamma,
            flag=regions[1], to_dict=True)
        assert np.isclose(res1["beta"], beta[0])
        assert np.isclose(res2["beta"], beta[1])
        assert np.isclose(res1[ratio_map[ratio_name]], ratio_val)
        assert np.isclose(res2[ratio_map[ratio_name]], ratio_val)


@pytest.mark.parametrize("pr, gamma, expected_mach", [
    (1.89293, 1.4, 1),
    (1.77157, 1.2, 1),
    (3.05, 1.4, 1.4002184324045146),
    (21.07, 1.4, 4.000186364255953),
    ([1.89293, 3.05, 21.07], 1.4, [1, 1.4002184324045146, 4.000186364255953])
])
def test_m1_from_rayleigh_pitot_pressure_ratio(pr, gamma, expected_mach):
    m1 = m1_from_rayleigh_pitot_pressure_ratio(pr, gamma)
    assert np.allclose(m1, expected_mach)
