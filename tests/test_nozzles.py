import numpy as np
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle,
    CD_Min_Length_Nozzle,
    min_length_supersonic_nozzle_moc
)
from pygasflow.nozzles.rao_parabola_angles import Rao_Parabola_Angles
import pytest


def test_rao_loading_data():
    Rao_Parabola_Angles()


@pytest.mark.parametrize("NozzleClass, Lc, Ld", [
    (CD_Conical_Nozzle, 0.27474774194546225, 3.7452160573276165),
    (CD_TOP_Nozzle, 0.3475417887987028, 2.993687246707643)
])
def test_defaults(NozzleClass, Lc, Ld):
    n = NozzleClass()
    assert np.isclose(n.inlet_radius, 0.4)
    assert np.isclose(n.outlet_radius, 1.2)
    assert np.isclose(n.throat_radius, 0.2)
    assert np.isclose(n.junction_radius_0, 0)
    assert np.isclose(n.junction_radius_j, 0.1)
    assert np.isclose(n.theta_c, 40)
    if NozzleClass is CD_Conical_Nozzle:
        assert np.isclose(n.theta_N, 15)
        assert np.isclose(n.theta_e, 15)
    else:
        assert np.isclose(n.theta_N, 33.55072672)
        assert np.isclose(n.theta_e, 8.66843931)
    assert np.isclose(n.fractional_length, 0.8)
    assert n.geometry_type == "axisymmetric"
    assert n.N == 200

    assert np.isclose(n.length_convergent, Lc)
    assert np.isclose(n.length_divergent, Ld)
    assert np.isclose(n.length, n.length_convergent + n.length_divergent)


def test_CD_Conical_Nozzle_angles_length_relationship():
    n1 = CD_Conical_Nozzle()
    n2 = CD_Conical_Nozzle(theta_c=50)
    n3 = CD_Conical_Nozzle(theta_c=50, theta_N=30)
    assert n1.length_convergent > n2.length_convergent
    assert np.isclose(n2.length_convergent, n3.length_convergent)
    assert np.isclose(n1.length_divergent, n2.length_divergent)
    assert n2.length_divergent > n3.length_divergent

    Lc_before = n3.length_convergent
    Ld_before = n3.length_divergent
    n3.param.update({"theta_c": 45, "theta_N": 20})
    assert n3.length_convergent > Lc_before
    assert n3.length_divergent > Ld_before


def test_CD_TOP_Nozzle_angles_length_relationship():
    n1 = CD_TOP_Nozzle()
    n2 = CD_TOP_Nozzle(theta_c=50)
    n3 = CD_TOP_Nozzle(theta_c=50, K=0.7)
    assert n1.length_convergent > n2.length_convergent
    assert np.isclose(n2.length_convergent, n3.length_convergent)
    assert np.isclose(n1.length_divergent, n2.length_divergent)
    assert n2.length_divergent > n3.length_divergent

    Lc_before = n3.length_convergent
    Ld_before = n3.length_divergent
    n3.param.update({"theta_c": 45, "fractional_length": 0.9})
    assert n3.length_convergent > Lc_before
    assert n3.length_divergent > Ld_before


@pytest.mark.parametrize("NozzleClass, params", [
    [CD_Conical_Nozzle, {"theta_c": 45, "theta_N": 20}],
    (CD_TOP_Nozzle, {"theta_c": 45, "K": 0.9})
])
def test_build_geometry(NozzleClass, params):
    n1 = NozzleClass()
    n2 = NozzleClass(**params)
    x1, y1 = n1.build_geometry()
    x2, y2 = n2.build_geometry()
    assert not np.allclose(x1, x2)
    assert not np.allclose(y1, y2)

    assert np.allclose(x1, n1.length_array)
    assert np.allclose(y1, n1.wall_radius_array)
    assert n1.geometry_type == "axisymmetric"
    assert np.allclose(
        n1.area_ratio_array,
        (np.pi * n1.wall_radius_array**2) / n1.throat_area)

    assert np.allclose(x2, n2.length_array)
    assert np.allclose(y2, n2.wall_radius_array)
    assert np.allclose(
        n2.area_ratio_array,
        (np.pi * n2.wall_radius_array**2) / n2.throat_area)

    n2.N = 100
    x3, y3 = n2.build_geometry()
    assert (len(x3) < len(x2)) and (len(x3) == 100)


@pytest.mark.parametrize("NozzleClass, geometry_type", [
    (CD_Conical_Nozzle, "axisymmetric"),
    (CD_Conical_Nozzle, "planar"),
    (CD_TOP_Nozzle, "axisymmetric"),
    (CD_TOP_Nozzle, "planar"),
])
def test_areas(NozzleClass, geometry_type):
    if geometry_type not in ["axisymmetric", "planar"]:
        pytest.raises(ValueError, NozzleClass(geometry_type=geometry_type))
        return

    n = NozzleClass(geometry_type=geometry_type)
    if geometry_type == "planar":
        assert np.isclose(n.inlet_area, 2 * n.inlet_radius)
        assert np.isclose(n.outlet_area, 2 * n.outlet_radius)
        assert np.isclose(n.throat_area, 2 * n.throat_radius)
    else:
        assert np.isclose(n.inlet_area, np.pi * n.inlet_radius**2)
        assert np.isclose(n.outlet_area, np.pi * n.outlet_radius**2)
        assert np.isclose(n.throat_area, np.pi * n.throat_radius**2)


@pytest.mark.parametrize("NozzleClass", [
    CD_Conical_Nozzle, CD_TOP_Nozzle, CD_Min_Length_Nozzle
])
def test_str(NozzleClass):
    # no errors should be raised here
    n = NozzleClass()
    str(n)


def test_min_length_supersonic_nozzle_moc():
    ht = 1
    n = 25
    Me = 2.4
    gamma = 1.4
    wall, _, _, theta_w_max = min_length_supersonic_nozzle_moc(
        ht, n, Me, None, gamma)
    assert np.allclose(
        wall,
        np.array([
            [0.        , 1.        ],
            [0.84081769, 1.27926714],
            [1.50295615, 1.49398143],
            [1.79486755, 1.58422063],
            [2.03759639, 1.6557268 ],
            [2.26213361, 1.71863473],
            [2.47946671, 1.77641245],
            [2.69501676, 1.83065161],
            [2.91200486, 1.88218827],
            [3.13260732, 1.93148763],
            [3.35844588, 1.97880672],
            [3.59082686, 2.024273  ],
            [3.83087036, 2.06792589],
            [4.07958567, 2.10974011],
            [4.33791843, 2.14963933],
            [4.60678176, 2.18750422],
            [4.88707805, 2.22317713],
            [5.17971496, 2.25646451],
            [5.48561777, 2.28713799],
            [5.80573953, 2.31493419],
            [6.14106961, 2.33955389],
            [6.49264145, 2.3606604 ],
            [6.86153966, 2.37787745],
            [7.24890692, 2.39078645],
            [7.65595078, 2.39892336],
            [8.08395049, 2.40177496]
        ])
    )


@pytest.mark.parametrize("NozzleClass, show_area_ratio, geom", [
    (CD_Conical_Nozzle, True, "axisymmetric"),
    (CD_Conical_Nozzle, False, "axisymmetric"),
    (CD_TOP_Nozzle, True, "axisymmetric"),
    (CD_TOP_Nozzle, False, "axisymmetric"),
    (CD_Conical_Nozzle, True, "planar"),
    (CD_Conical_Nozzle, False, "planar"),
    (CD_TOP_Nozzle, True, "planar"),
    (CD_TOP_Nozzle, False, "planar"),
    (CD_Min_Length_Nozzle, True, "planar"),
    (CD_Min_Length_Nozzle, False, "planar"),
])
def test_get_points(NozzleClass, show_area_ratio, geom):
    # only test the first y-value (inlet) and the max y-values (outlet)
    # of the nozzle coordinates to save coding time
    n = NozzleClass(Ri=0.4, Rt=0.2, Re=1.2, geometry_type=geom)
    nozzle, container = n.get_points(show_area_ratio)
    if geom == "axisymmetric":
        expected_first_val = 4 if show_area_ratio else 0.4
        expected_last_val = 36 if show_area_ratio else 1.2
    else:
        expected_first_val = 2 if show_area_ratio else 0.4
        expected_last_val = 6 if show_area_ratio else 1.2
    assert np.isclose(nozzle[:, 1][0], expected_first_val)
    if NozzleClass is not CD_TOP_Nozzle:
        assert np.isclose(max(nozzle[:, 1]), expected_last_val)


