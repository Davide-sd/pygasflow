import numpy as np
from bokeh.plotting import figure
from pygasflow.interactive.diagrams import (
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram,
    GasDiagram,
    SonicDiagram
)
from pygasflow.interactive.diagrams.flow_base import BasePlot
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_hex


expected = {
    IsentropicDiagram: {
        "title": "Isentropic Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": (0, 5),
        "y_range": (0, 3),
        "size": (800, 300),
    },
    FannoDiagram: {
        "title": "Fanno Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": (0, 5),
        "y_range": (0, 3),
        "size": (800, 300),
    },
    RayleighDiagram: {
        "title": "Rayleigh Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": (0, 5),
        "y_range": (0, 3),
        "size": (800, 300),
    },
    NormalShockDiagram: {
        "title": "Normal Shock Properties",
        "x_label": "Upstream Mach, M1",
        "y_label": "Ratios",
        "x_range": (0, 5),
        "y_range": (0, 1.5),
        "size": (700, 400),
    },
    ObliqueShockDiagram: {
        "title": "Oblique Shock Properties: Mach - β - θ",
        "x_label": "Flow Deflection Angle, θ [deg]",
        "y_label": "Shock Wave Angle, β [deg]",
        "x_range": (0, 50),
        "y_range": (0, 90),
        "size": (700, 400),
    },
    ConicalShockDiagram: {
        "title": "Conical Shock Properties: Mach - β - θc",
        "x_label": "Half cone angle, θc [deg]",
        "y_label": "Shock Wave Angle, β [deg]",
        "x_range": (0, 60),
        "y_range": (0, 90),
        "size": (700, 400),
    },
    GasDiagram: {
        "title": "",
        "x_label": "Mass-specific gas constant, R, [J / (Kg K)]",
        "y_label": "Specific Heats",
        "x_range": (1.05, 2),
        "y_range": None,
        "size": (800, 300),
    },
    SonicDiagram: {
        "title": "Sonic condition",
        "x_label": "Ratio of specific heats, γ",
        "y_label": "Ratios",
        "x_range": (1.05, 2),
        "y_range": None,
        "size": (800, 300),
    },
}

diagrams = [
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram,
    GasDiagram,
    SonicDiagram,
]


@pytest.mark.parametrize("DiagramClass", diagrams)
def test_hierarchy(DiagramClass):
    assert issubclass(DiagramClass, BasePlot)
    assert hasattr(DiagramClass, "title")
    assert hasattr(DiagramClass, "x_range")
    assert hasattr(DiagramClass, "y_range")
    assert hasattr(DiagramClass, "x_label")
    assert hasattr(DiagramClass, "y_label")


@pytest.mark.parametrize("DiagramClass", diagrams)
def test_instantiation_no_params(DiagramClass):
    i1 = DiagramClass()
    assert isinstance(i1.figure, figure)
    assert i1.title == expected[DiagramClass]["title"]
    assert i1.x_label == expected[DiagramClass]["x_label"]
    assert i1.figure.xaxis.axis_label == expected[DiagramClass]["x_label"]
    assert i1.y_label == expected[DiagramClass]["y_label"]

    if DiagramClass is IsentropicDiagram:
        assert i1.figure.yaxis.axis_label == ["Ratios", "Angles [deg]"]
    else:
        assert i1.figure.yaxis.axis_label == expected[DiagramClass]["y_label"]
    assert i1.x_range == expected[DiagramClass]["x_range"]

    if (
        (DiagramClass is ObliqueShockDiagram)
        or (DiagramClass is ConicalShockDiagram)
    ):
        assert (i1.figure.x_range.start, i1.figure.x_range.end) == expected[DiagramClass]["x_range"]
    else:
        # not enforced on init
        assert np.allclose(
            (i1.figure.x_range.start, i1.figure.x_range.end),
            (np.nan, np.nan),
            equal_nan=True
        )

    if expected[DiagramClass]["y_range"] is not None:
        assert i1.y_range == expected[DiagramClass]["y_range"]
        assert (i1.figure.y_range.start, i1.figure.y_range.end) == expected[DiagramClass]["y_range"]

    assert i1.size == expected[DiagramClass]["size"]
    assert (i1.figure.width, i1.figure.height) == expected[DiagramClass]["size"]

    i1.x_range = (1, 2)
    assert (i1.figure.x_range.start, i1.figure.x_range.end) == (1, 2)
    i1.y_range = (3, 4)
    assert (i1.figure.y_range.start, i1.figure.y_range.end) == (3, 4)
    i1.y_range_right = (5, 6)


@pytest.mark.parametrize("DiagramClass", [
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram,
    GasDiagram,
    SonicDiagram
])
def test_instantiation_with_params(DiagramClass):
    params = dict(
        title="Title", x_label="x_label", y_label="y_label",
        x_range=(1, 2), y_range=(3, 4), size=(100, 200),
        colors=tuple(to_hex(c) for c in colormaps["tab20"].colors[:10])
    )
    if DiagramClass is IsentropicDiagram:
        params["y_label_right"] = "y_label right"
        params["y_range_right"] = (5, 6)

    i2 = DiagramClass(**params)
    assert isinstance(i2.figure, figure)
    assert i2.title == "Title"
    assert i2.x_label == "x_label"
    assert i2.figure.xaxis.axis_label == "x_label"
    assert i2.y_label == "y_label"

    if DiagramClass is IsentropicDiagram:
        assert i2.figure.yaxis.axis_label == ["y_label", "y_label right"]
    else:
        assert i2.figure.yaxis.axis_label == "y_label"

    assert i2.x_range == (1, 2)

    if (
        (DiagramClass is ObliqueShockDiagram)
        or (DiagramClass is ConicalShockDiagram)
    ):
        assert (i2.figure.x_range.start, i2.figure.x_range.end) == expected[DiagramClass]["x_range"]
    else:
        assert np.allclose(
            (i2.figure.x_range.start, i2.figure.x_range.end),
            (np.nan, np.nan),
            equal_nan=True
        )

    if expected[DiagramClass]["y_range"] is not None:
        assert i2.y_range == (3, 4)
        assert (i2.figure.y_range.start, i2.figure.y_range.end) == (3, 4)

    assert i2.size == (100, 200)
    assert (i2.figure.width, i2.figure.height) == (100, 200)


@pytest.mark.parametrize("DiagramClass", [
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
])
def test_update_flow_related_diagrams(DiagramClass):
    # verify new data is computed as numeric parameters are changed

    i = DiagramClass()
    i.gamma = 1.4
    old_sources = [r.data_source.data.copy() for r in i.figure.renderers]
    x_range_old = (old_sources[0]["xs"].min(), old_sources[0]["xs"].max())

    i.gamma = 1.2
    new_sources = [r.data_source.data.copy() for r in i.figure.renderers]
    x_range_new = (new_sources[0]["xs"].min(), new_sources[0]["xs"].max())
    assert np.allclose(x_range_old, x_range_new)
    for s1, s2 in zip(old_sources, new_sources):
        assert len(s1["xs"]) == 100
        assert np.allclose(s1["xs"], s2["xs"])
        assert not np.allclose(s1["ys"], s2["ys"])
        assert s1["v"] == s2["v"]

    i.N = 10
    assert len(i.figure.renderers) == len(new_sources)
    data = i.figure.renderers[0].data_source.data
    assert len(data["xs"]) == 10
    assert np.allclose(
        (data["xs"].min(), data["xs"].max()),
        x_range_old
    )

    i.mach_range = (2, 4)
    assert len(i.figure.renderers) == len(new_sources)
    data = i.figure.renderers[0].data_source.data
    assert len(data["xs"]) == 10
    assert np.allclose(
        (data["xs"].min(), data["xs"].max()),
        (2, 4)
    )


@pytest.mark.parametrize("DiagramClass", [
    ObliqueShockDiagram,
    ConicalShockDiagram,
])
def test_update_shock_related_diagrams(DiagramClass):
    # verify new data is computed as numeric parameters are changed

    i = DiagramClass()
    i.gamma = 1.4
    old_sources = [r.data_source.data.copy() for r in i.figure.renderers]
    x_range_old = (old_sources[0]["xs"].min(), old_sources[0]["xs"].max())

    i.gamma = 1.2
    new_sources = [r.data_source.data.copy() for r in i.figure.renderers]
    x_range_new = (new_sources[0]["xs"].min(), new_sources[0]["xs"].max())
    # by updating gamma, the curve moves to the left or to the right
    assert not np.allclose(x_range_old, x_range_new)

    # 7 mach lines + 1 sonic line + 1 region line
    assert len(new_sources) == 9
    # sonic line and region line are visible
    assert all(r.visible for r in i.figure.renderers[-2:])
    assert i.show_region_line and i.show_sonic_line

    for s1, s2 in zip(old_sources[:-2], new_sources[:-2]):
        # conical solver inserts a couple more values
        assert len(s1["xs"]) >= 100
        assert not np.allclose(s1["xs"], s2["xs"])
        assert np.allclose(s1["ys"], s2["ys"])
        assert s1["v"] == s2["v"]

    i.N = 10
    assert len(i.figure.renderers) == len(new_sources)
    data = i.figure.renderers[0].data_source.data.copy()
    assert len(data["xs"]) >= 10

    with pytest.raises(ValueError):
        # too few Mach numbers
        i.upstream_mach = [1, 2, 3]

    with pytest.raises(ValueError):
        # too many Mach numbers
        i.upstream_mach = [1, 2, 3, 4, 5, 6, 7, 8]

    i.upstream_mach = [5, 6, 7, 8, 9, 10, 11]
    new_data = i.figure.renderers[0].data_source.data.copy()
    # the first curve is using a new mach number, which changes both
    # the values on the x-axis (theta) and y-axis (beta)
    assert not np.allclose(data["xs"], new_data["xs"])
    assert not np.allclose(data["ys"], new_data["ys"])

    # hide sonic line
    i.show_sonic_line = False
    assert not i.figure.renderers[-2].visible

    # hide region line
    i.show_region_line = False
    assert not i.figure.renderers[-1].visible


def test_update_gas_diagram():
    d = GasDiagram()

    # we start with the R-Cp/Cv diagram
    assert d.select == 1
    assert d.figure.xaxis.axis_label == "Mass-specific gas constant, R, [J / (Kg K)]"
    assert len(d.figure.renderers) == 2
    old_data = d.figure.renderers[0].data_source.data.copy()

    # update gamma
    d.gamma = 1.6
    new_data = d.figure.renderers[0].data_source.data.copy()
    assert np.allclose(old_data["xs"], new_data["xs"])
    assert not np.allclose(old_data["ys"], new_data["ys"])
    assert len(old_data["xs"]) == len(new_data["xs"]) == 100
    x_range = (new_data["xs"].min(), new_data["xs"].max())

    d.R_range = (0, 4000)
    new_data2 = d.figure.renderers[0].data_source.data.copy()
    x_range2 = (new_data2["xs"].min(), new_data2["xs"].max())
    assert x_range != x_range2

    d.N = 10
    new_data3 = d.figure.renderers[0].data_source.data.copy()
    assert len(new_data3["xs"]) == 10

    # move to the gamma-Cp/Cv diagram
    d.select = 0
    assert d.figure.xaxis.axis_label == "Ratio of specific heats, γ"
    new_data4 = d.figure.renderers[0].data_source.data.copy()
    assert len(new_data4["xs"]) == 10
    assert not np.allclose(new_data4["xs"], new_data3["xs"])
    x_range3 = (new_data4["xs"].min(), new_data4["xs"].max())
    assert np.allclose(x_range3, (1.05, 2))

    d.gamma_range = (1.5, 1.75)
    new_data5 = d.figure.renderers[0].data_source.data.copy()
    x_range4 = (new_data5["xs"].min(), new_data5["xs"].max())
    assert x_range3 != x_range4


def test_update_sonic_diagram():
    d = SonicDiagram()
    assert len(d.figure.renderers) == 4
    old_data = d.figure.renderers[0].data_source.data.copy()
    assert len(old_data["xs"]) == 100
    x_range1 = (old_data["xs"].min(), old_data["xs"].max())

    d.gamma_range = (1.5, 1.75)
    new_data = d.figure.renderers[0].data_source.data.copy()
    x_range2 = (new_data["xs"].min(), new_data["xs"].max())
    assert x_range1 != x_range2

    d.N = 10
    new_data2 = d.figure.renderers[0].data_source.data.copy()
    assert len(new_data2["xs"]) == 10
