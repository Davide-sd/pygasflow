import pygasflow.isentropic as ise
import pygasflow.fanno as fanno
import pygasflow.rayleigh as r
import pygasflow.shockwave as sw
import pygasflow.solvers as sol
from pygasflow.utils.common import ShockResults
import numpy as np
import pytest

# NOTE:
# In this module I mainly test two things:
# 1. The decorators do what they are supposed to do.
# 2. The functions raising conditional exceptions work as expected.
#
# Be careful with numpy array:
#       np.asarray([2, 3]) -> dtype=int
#       np.asarray([2., 3]) -> dtype=float
# which is going to produce different results in the functions when
# used with __no_check__. If used normally, the `check` decorator converts
# the provided values to float.

################################################################################
################################## CHECK DECORATOR #############################
################################################################################

def test_mach_zero():
    def test_0(r, is_iter=False):
        # Almost all functions require the mach number to be a Numpy array: the
        # decorator convert the provided input to a Numpy array. If I pass a single
        # mach number, the decorator converts it to a 1-D array. However, I would
        # like the function to return a scalar value, not a 1-D result array.
        # The decorator is supposed to convert 0-D or 1-D outputs to scalar values.
        # If a function returns a tuple of elements, the decorators will repeat the
        # same process to each element. In doing so:
        # 1. If I provide a scalar value, the functions will return either a scalar
        #    value or a tuple of scalars.
        # 2. If I provide a list or array of values (with more than one element),
        #    the function will return either an array or a tuple of arrays
        for e in r:
            assert isinstance(e, (list, tuple, np.ndarray)) == is_iter

    # I'm only going to test solvers, since they use many other functions implemented
    # in the different modules.
    test_0(sol.shockwave.oblique_shockwave_solver("mu", 2, gamma=1.4), False)
    test_0(sol.shockwave.oblique_shockwave_solver("mu", [2, 4], gamma=1.4), True)
    test_0(sol.shockwave.conical_shockwave_solver(5, "theta_c", 20, gamma=1.4), False)
    test_0(sol.shockwave.conical_shockwave_solver([2, 4], "theta_c", 20, gamma=1.4), True)
    test_0(sol.shockwave.conical_shockwave_solver([2, 4], "mc", 1.5, gamma=1.4), True)
    test_0(sol.isentropic.isentropic_solver("m", 2), False)
    test_0(sol.isentropic.isentropic_solver("m", [2, 4]), True)
    test_0(sol.fanno.fanno_solver("m", 2), False)
    test_0(sol.fanno.fanno_solver("m", [2, 4]), True)
    test_0(sol.rayleigh.rayleigh_solver("m", 2), False)
    test_0(sol.rayleigh.rayleigh_solver("m", [2, 4]), True)


def func(error, fn, *args, nocheck=False, **kwargs):
    """ Execute a statement and verify that the expected error is raised.
    """
    if not nocheck:
        with pytest.raises(error):
            fn(*args, **kwargs)
    else:
        with pytest.raises(error):
            fn.__no_check__(*args, **kwargs)


def test_raise_error_isentropic():
    # Raise errors when (note: the errors are raised inside the decorator):
    func(ValueError, ise.critical_velocity_ratio, -1)           # Mach is negative
    func(ValueError, ise.critical_velocity_ratio, [2, 3, -1])   # at least one Mach number is negative
    func(ValueError, ise.critical_velocity_ratio, 2, 0.9)       # gamma <= 1
    func(ValueError, ise.critical_velocity_ratio, 2, 1)         # gamma <= 1
    func(ValueError, ise.critical_velocity_ratio, 2, gamma=0.8) # gamma <= 1
    func(ValueError, ise.m_from_critical_area_ratio, 0.9)       # A/A* < 1
    func(ValueError, ise.m_from_critical_area_ratio, 2, flag="asd") # flag must be 'sub' or 'super'
    func(ValueError, ise.m_from_temperature_ratio, -1)          # ratio < 0
    func(ValueError, ise.m_from_temperature_ratio, [-1, 0.5])   # at least one ratio < 0
    func(ValueError, ise.m_from_temperature_ratio, 1.5)         # ratio > 1
    func(ValueError, ise.m_from_temperature_ratio, [1.5, 0.5])  # at least one ratio > 1
    func(ValueError, ise.m_from_pressure_ratio, -1)          # ratio < 0
    func(ValueError, ise.m_from_pressure_ratio, [-1, 0.5])   # at least one ratio < 0
    func(ValueError, ise.m_from_pressure_ratio, 1.5)         # ratio > 1
    func(ValueError, ise.m_from_pressure_ratio, [1.5, 0.5])  # at least one ratio > 1
    func(ValueError, ise.m_from_density_ratio, -1)          # ratio < 0
    func(ValueError, ise.m_from_density_ratio, [-1, 0.5])   # at least one ratio < 0
    func(ValueError, ise.m_from_density_ratio, 1.5)         # ratio > 1
    func(ValueError, ise.m_from_density_ratio, [1.5, 0.5])  # at least one ratio > 1
    func(ValueError, ise.m_from_critical_area_ratio, -1)          # ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio, [-1, 0.5, 1])   # at least one ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, -1, 0.5)          # a_ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, [-1, 0.5, 1], 0.5)   # at least one a_ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, 1.5, -1)         # p_ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, [1.5, 2], [-1, 0.5])  # at least one p_ratio < 0
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, 1.5, 1.5)         # p_ratio > 1
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, [1.5, 2], [1.5, 0.5])  # at least one p_ratio > 1
    func(ValueError, ise.m_from_critical_area_ratio_and_pressure_ratio, [1.5, 2, 3], [1.5, 0.5])  # a_ratio.shape != p_ratio.shape
    func(ValueError, ise.m_from_mach_angle, -1)     # angle < 0
    func(ValueError, ise.m_from_mach_angle, [-1, 0.5])     # at least one angle < 0
    func(ValueError, ise.m_from_mach_angle, 95)     # angle > 90
    func(ValueError, ise.m_from_mach_angle, [95, 0.5])     # at least one angle > 90
    func(ValueError, ise.m_from_prandtl_meyer_angle, -1)     # angle < 0
    func(ValueError, ise.m_from_prandtl_meyer_angle, [-1, 0.5])     # at least one angle < 0
    func(ValueError, ise.m_from_prandtl_meyer_angle, 140)     # angle > nu_max
    func(ValueError, ise.m_from_prandtl_meyer_angle, [140, 0.5])     # at least one angle > nu_max

    # Test for __no_check__. The aim is for the parameters to make into
    # the function and either produce errors or produce random junk results
    func(TypeError, ise.critical_velocity_ratio, -1, nocheck=True)  # int is not subscriptable
    func(TypeError, ise.critical_velocity_ratio, 2, gamma=0.9, nocheck=True)  # int is not subscriptable
    assert np.all(ise.critical_velocity_ratio.__no_check__([2, 3, -1]) - np.asarray([1, 1, 1]) == 0)


def test_raise_error_fanno():
    func(ValueError, fanno.m_from_critical_temperature_ratio, -1) # ratio <= 0
    func(ValueError, fanno.m_from_critical_temperature_ratio, 0)  # ratio <= 0
    func(ValueError, fanno.m_from_critical_temperature_ratio, 1.5) # ratio > upper_lim
    func(ValueError, fanno.m_from_critical_density_ratio, -1)     # ratio < lower_lim
    func(ValueError, fanno.m_from_critical_density_ratio, 0)      # ratio < lower_lim
    func(ValueError, fanno.m_from_critical_density_ratio, 0.25)   # ratio < lower_lim
    func(ValueError, fanno.m_from_critical_density_ratio, [-1, 0, 0.25, 0.5])   # at least one ratio < lower_lim
    func(ValueError, fanno.m_from_critical_total_pressure_ratio, -1)   # ratio < 1
    func(ValueError, fanno.m_from_critical_total_pressure_ratio, [0.25, 2])   # at least one ratio < 1
    func(ValueError, fanno.m_from_critical_velocity_ratio, -1)    # ratio < 0
    func(ValueError, fanno.m_from_critical_velocity_ratio, [-1, 0])    # at least one ratio < 0
    func(ValueError, fanno.m_from_critical_velocity_ratio, 10.5)    # ratio > upper_lim
    func(ValueError, fanno.m_from_critical_velocity_ratio, [0.5, 10.5])    # at least one ratio > upper_lim
    func(ValueError, fanno.m_from_critical_friction, -1, flag="sub") # fp < 0 when flag='sub'
    func(ValueError, fanno.m_from_critical_friction, [-1, 0.5], flag="sub") # at least one fp < 0 when flag='sub'
    func(ValueError, fanno.m_from_critical_friction, [-1, 0.5], flag="super") # at least one fp < 0 when flag='super'
    func(ValueError, fanno.m_from_critical_friction, [0.5, 10.5], flag="super") # at least one fp > upper_lim when flag='super'
    func(ValueError, fanno.m_from_critical_entropy, -1) # ep < 0
    func(ValueError, fanno.m_from_critical_entropy, [0.5, -1]) # at least one ep < 0


def test_raise_error_rayleigh():
    func(ValueError, r.m_from_critical_total_temperature_ratio, -1) # ratio < 0
    func(ValueError, r.m_from_critical_total_temperature_ratio, [-1, 0.5]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_total_temperature_ratio, 1.5) # ratio < 0
    func(ValueError, r.m_from_critical_total_temperature_ratio, [1.5, 0.5]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_total_temperature_ratio, [1.5, -0.5]) # ratio < 0, ratio > 1
    func(ValueError, r.m_from_critical_temperature_ratio, -1) # ratio < 0
    func(ValueError, r.m_from_critical_temperature_ratio, [-1, 0.5]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_temperature_ratio, 10.5) # ratio < 0
    func(ValueError, r.m_from_critical_temperature_ratio, [10.5, 0.15]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_temperature_ratio, [10.5, -0.5]) # ratio < 0, ratio > 1
    func(ValueError, r.m_from_critical_pressure_ratio, -1) # ratio < 0
    func(ValueError, r.m_from_critical_pressure_ratio, [-1, 0.5]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_pressure_ratio, 10.5) # ratio < 0
    func(ValueError, r.m_from_critical_pressure_ratio, [10.5, 0.15]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_pressure_ratio, [10.5, -0.5]) # ratio < 0, ratio > 1
    func(ValueError, r.m_from_critical_total_pressure_ratio, -1, "sub") # ratio < 0
    func(ValueError, r.m_from_critical_total_pressure_ratio, [-1, 1.5], "sub") # at least one ratio < 0
    func(ValueError, r.m_from_critical_total_pressure_ratio, 10.5, "sub") # ratio > upper_lim
    func(ValueError, r.m_from_critical_total_pressure_ratio, [1.15, 1.5], "sub") # at least one ratio > upper_lim
    func(ValueError, r.m_from_critical_total_pressure_ratio, 0.5, "super") # ratio < 0
    func(ValueError, r.m_from_critical_total_pressure_ratio, [1.5, 0.15], "super") # at least one ratio < 0
    func(ValueError, r.m_from_critical_density_ratio, -1) # ratio < lower_lim
    func(ValueError, r.m_from_critical_density_ratio, [-1, 2.5]) # at least one ratio < lower_lim
    func(ValueError, r.m_from_critical_velocity_ratio, -1) # ratio < 0
    func(ValueError, r.m_from_critical_velocity_ratio, [-1, 0.5]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_velocity_ratio, 10.5) # ratio < 0
    func(ValueError, r.m_from_critical_velocity_ratio, [10.5, 0.15]) # at least one ratio < 0
    func(ValueError, r.m_from_critical_velocity_ratio, [10.5, -0.5]) # ratio < 0, ratio > 1
    func(ValueError, r.m_from_critical_entropy, -1) # ratio < lower_lim
    func(ValueError, r.m_from_critical_entropy, [-1, 2.5]) # at least one ratio < lower_lim



def test_raise_error_shockwave():
    # CHECK_SHOCKWAVE DECORATOR

    # Raise errors when:
    func(ValueError, sw.pressure_ratio, -1)           # Mach < 1
    func(ValueError, sw.pressure_ratio, 0.5)           # Mach < 1
    func(ValueError, sw.pressure_ratio, [2, 0.5, -1])   # at least one Mach number is < 1
    func(ValueError, sw.pressure_ratio, 2, 0.9)       # gamma <= 1
    func(ValueError, sw.pressure_ratio, 2, 1)         # gamma <= 1
    func(ValueError, sw.pressure_ratio, 2, gamma=0.8) # gamma <= 1
    func(ValueError, sw.m1_from_pressure_ratio, 0.5, 0.8) # gamma <= 1
    func(ValueError, sw.m1_from_pressure_ratio, 0.5, gamma=0.8) # gamma <= 1
    func(ValueError, sw.m1_from_m2, 0.5, gamma=0.8) # gamma <= 1
    func(ValueError, sw.theta_from_mach_beta, -1, 20)           # M1 < 1
    func(ValueError, sw.theta_from_mach_beta, 0.5, 20)           # M1 < 1
    func(ValueError, sw.theta_from_mach_beta, [2, 0.5, -1], 20)           # at least one M1 < 1
    func(ValueError, sw.theta_from_mach_beta, 2, -5)           # beta < 0
    func(ValueError, sw.theta_from_mach_beta, 2, 95)           # beta > 90
    func(ValueError, sw.beta_from_mach_theta, 2, -5)           # theta < 0
    func(ValueError, sw.beta_from_mach_theta, 2, 95)           # theta > 90
    func(ValueError, sw.beta_from_upstream_mach, 0.5, 0.5)           # M1 < 1
    func(ValueError, sw.beta_from_upstream_mach, 2, 3)           # M1 < MN1
    func(ValueError, sw.beta_from_upstream_mach, [2, 0.5], [0.5, 0.5]) # at least one M1 < 1
    func(ValueError, sw.beta_from_upstream_mach, [2, 3], [1, 4]) # at least one M1 < MN1
    func(ValueError, sw.beta_from_upstream_mach, [2, 3], [1, 4, 3]) # at least one M1.shape != MN1.shape
    func(ValueError, sw.normal_mach_upstream, 0.5, 20) # M1 < 1
    func(ValueError, sw.normal_mach_upstream, [2, 0.5], 20) # at least one M1 < 1
    func(ValueError, sw.normal_mach_upstream, [2, 3]) # no beta nor theta
    func(ValueError, sw.normal_mach_upstream, [2, 3], -5) # beta < 0
    func(ValueError, sw.normal_mach_upstream, [2, 3], 95) # beta > 90
    func(ValueError, sw.normal_mach_upstream, [2, 3], None, -5) # theta < 0
    func(ValueError, sw.normal_mach_upstream, [2, 3], None, 95) # theta > 90
    func(ValueError, sw.normal_mach_upstream, [2, 3], None, 20, flag="asd") # flag is not 'weak', 'strong' or 'both'
    func(ValueError, sw.normal_mach_upstream, [2, 3], 20, flag="asd") # flag is not 'weak', 'strong' or 'both'
    func(ValueError, sw.normal_mach_upstream, [2, 3], theta=20, flag="asd") # flag is not 'weak', 'strong' or 'both'
    func(ValueError, sw.get_upstream_normal_mach_from_ratio, "asd", 2, nocheck=True) # invalid ratioName
    func(ValueError, sw.maximum_mach_from_deflection_angle, -5) # theta < 0
    func(ValueError, sw.maximum_mach_from_deflection_angle, 95) # theta > 90
    func(ValueError, sw.maximum_mach_from_deflection_angle, 50) # theta > upper_lim
    func(ValueError, sw.maximum_mach_from_deflection_angle, 20, 0.8) # gamma <= 1
    func(ValueError, sw.maximum_mach_from_deflection_angle, 20, 1) # gamma <= 1
    func(ValueError, sw.mach_from_theta_beta, -5, 20, 1.4) # beta < 0
    func(ValueError, sw.mach_from_theta_beta, 95, 20, 1.4) # beta > 90
    func(ValueError, sw.mach_from_theta_beta, 20, -5, 1.4) # theta < 0
    func(ValueError, sw.mach_from_theta_beta, 20, 95, 1.4) # theta > 90
    func(ValueError, sw.mach_from_theta_beta, 20, 20, 1) # gamma <= 1
    func(ValueError, sw.mach_from_theta_beta, [20, 25], [5, 10, 20]) # beta.shape != theta.shape
    func(ValueError, sw.shock_polar, 0.5)      # M < 1
    func(ValueError, sw.shock_polar, 2, 1)     # gamma <= 1
    func(ValueError, sw.shock_polar, 2, 0.5)   # gamma <= 1
    func(ValueError, sw.shock_polar, 2, N=1.2) # N is float
    func(ValueError, sw.shock_polar, 2, N=-1)  # N is integer < 1
    func(ValueError, sw.pressure_deflection, 2, N=1.2) # N is float
    func(ValueError, sw.pressure_deflection, 2, N=-1)  # N is integer < 1
    func(ValueError, sw.mach_from_nondimensional_velocity, -1) # V < 0
    func(ValueError, sw.mach_from_nondimensional_velocity, [2, -0.5]) # at least one V < 0
    func(ValueError, sw.mach_downstream, -1)    # M1 < 0
    func(ValueError, sw.mach_downstream, [-1, 1.5]) # at least one M1 < 0
    func(ValueError, sw.m1_from_pressure_ratio, 0.5) # ratio < 1
    func(ValueError, sw.m1_from_pressure_ratio, [0.5, 1.5]) # at least one ratio < 1
    func(ValueError, sw.m1_from_temperature_ratio, 0.5) # ratio < 1
    func(ValueError, sw.m1_from_temperature_ratio, [0.5, 1.5]) # at least one ratio < 1
    func(ValueError, sw.m1_from_density_ratio, 0.5) # ratio < 1
    func(ValueError, sw.m1_from_density_ratio, [0.5, 1.5]) # at least one ratio < 1
    func(ValueError, sw.m1_from_density_ratio, 10) # ratio > gr
    func(ValueError, sw.m1_from_density_ratio, [1.1, 10]) # at least one ratio > gr
    func(ValueError, sw.m1_from_total_pressure_ratio, -0.5) # ratio < 0
    func(ValueError, sw.m1_from_total_pressure_ratio, [-0.5, 0.5]) # at least one ratio < 0
    func(ValueError, sw.m1_from_total_pressure_ratio, 1.5) # ratio > 1
    func(ValueError, sw.m1_from_total_pressure_ratio, [0.5, 1.5]) # at least one ratio > 1
    func(ValueError, sw.m1_from_m2, 0.1) # M2 < lower_lim
    func(ValueError, sw.m1_from_m2, [0.1, 1.5]) # at least one M2 < lower_lim
    func(ValueError, sw.m1_from_m2, 1.5) # M2 > 1
    func(ValueError, sw.m1_from_m2, [0.5, 1.5]) # at least one M2 > 1


@pytest.mark.parametrize("k, should_warn", [
    ("m", True),
    ("m1", True),
    ("m2", True),
    ("mn1", True),
    ("mn2", True),
    ("pc_p1", True),
    ("rhoc_rho1", True),
    ("Tc_T1", True),
    ("pr", False),
    ("dr", False),
    ("tr", False),
    ("ttr", False),
    ("tpr", False),
])
def test_ShockResults_warning(k, should_warn):
    d = ShockResults(
        mu=1,
        mnu=2,
        md=3,
        mnd=4,
        pr=5,
        dr=6,
        tr=7,
        tpr=8,
        ttr=9,
        pc_pu=10,
        Tc_Tu=11,
        rhoc_rhou=12,
    )
    f = lambda k: d[k]
    if should_warn:
        with pytest.warns(
            UserWarning,
            match=f"Key '{k}' is deprecated"
        ):
            f(k)
    else:
        f(k)
