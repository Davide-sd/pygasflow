import numpy as np
import param
from bokeh.plotting import figure
from bokeh.models import GlyphRenderer, Line, Circle, Label, Arrow
from pygasflow.interactive.diagrams import (
    IsentropicDiagram,
    FannoDiagram,
    RayleighDiagram,
    NormalShockDiagram,
    ObliqueShockDiagram,
    ConicalShockDiagram,
    GasDiagram,
    SonicDiagram,
    NozzleDiagram,
    DeLavalDiagram,
    PressureDeflectionDiagram,
)
from pygasflow.interactive.diagrams.flow_base import BasePlot
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle,
    CD_Min_Length_Nozzle
)
from pygasflow.solvers import De_Laval_Solver
from pygasflow.shockwave import PressureDeflectionLocus
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_hex


expected = {
    IsentropicDiagram: {
        "title": "Isentropic Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": None,
        "y_range": (0, 3),
        "size": (800, 300),
    },
    FannoDiagram: {
        "title": "Fanno Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": None,
        "y_range": (0, 3),
        "size": (800, 300),
    },
    RayleighDiagram: {
        "title": "Rayleigh Flow",
        "x_label": "M",
        "y_label": "Ratios",
        "x_range": None,
        "y_range": (0, 3),
        "size": (800, 300),
    },
    NormalShockDiagram: {
        "title": "Normal Shock Properties",
        "x_label": "Upstream Mach, M1",
        "y_label": "Ratios",
        "x_range": None,
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
        "y_label": "Specific Heats [J / K]",
        "x_range": (0, 5000),
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
    PressureDeflectionDiagram: {
        "title": "",
        "x_label": "Deflection angle, θ [deg]",
        "y_label": "Pressure Ratio to Freestream",
        "x_range": None,
        "y_range": None,
        "size": (600, 400),
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
    PressureDeflectionDiagram
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

    if expected[DiagramClass]["x_range"] is not None:
        assert i1.x_range == expected[DiagramClass]["x_range"]

    if DiagramClass in [
        ObliqueShockDiagram, ConicalShockDiagram, GasDiagram, SonicDiagram
    ]:
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


@pytest.mark.parametrize("DiagramClass", diagrams)
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

    # ObliqueShockDiagram/ConicalShockDiagram recomputes appropriate x_range
    # on each update
    if DiagramClass is ObliqueShockDiagram:
        assert (i2.figure.x_range.start, i2.figure.x_range.end) == (0, 50)
    elif DiagramClass is ConicalShockDiagram:
        assert (i2.figure.x_range.start, i2.figure.x_range.end) == (0, 60)
    else:
        assert (i2.figure.x_range.start, i2.figure.x_range.end) == (1, 2)

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
    assert len(old_data["xs"]) == 10
    x_range1 = (old_data["xs"].min(), old_data["xs"].max())

    d.gamma_range = (1.5, 1.75)
    new_data = d.figure.renderers[0].data_source.data.copy()
    x_range2 = (new_data["xs"].min(), new_data["xs"].max())
    assert x_range1 != x_range2

    d.N = 20
    new_data2 = d.figure.renderers[0].data_source.data.copy()
    assert len(new_data2["xs"]) == 20


@pytest.mark.parametrize("NozzleClass", [
    CD_Conical_Nozzle, CD_TOP_Nozzle, CD_Min_Length_Nozzle])
def test_update_nozzle_diagram(NozzleClass):
    nozzle = NozzleClass()
    d = NozzleDiagram(nozzle=nozzle)
    y_range1 = (d.figure.y_range.start, d.figure.y_range.end)
    assert d.figure.xaxis.axis_label == "Length [m]"
    assert d.figure.yaxis.axis_label == "radius [m]"
    data1 = d.figure.renderers[0].data_source.data.copy()
    idx = -4 if NozzleClass is CD_Min_Length_Nozzle else -2
    data_sw1 = d.figure.renderers[idx].data_source.data.copy()
    assert np.allclose(data_sw1["x"], [np.nan, np.nan], equal_nan=True)
    assert np.allclose(data_sw1["y"], [np.nan, np.nan], equal_nan=True)

    d.show_area_ratio = True
    assert d.figure.yaxis.axis_label == "A/A*"
    data2 = d.figure.renderers[0].data_source.data.copy()
    assert np.allclose(data1["x"], data2["x"])
    assert not np.allclose(data1["y"], data2["y"])

    d.show_half_nozzle = False
    y_range2 = (d.figure.y_range.start, d.figure.y_range.end)
    assert not np.allclose(y_range1, y_range2)

    # simulate a change in flow condition, such that a shockwave is present
    # in the divergent
    with param.edit_constant(nozzle):
        nozzle.shockwave_location = (1, 0.25)
    idx = -4 if NozzleClass is CD_Min_Length_Nozzle else -2
    data_sw2 = d.figure.renderers[idx].data_source.data.copy()
    assert not np.allclose(data_sw2["x"], [np.nan, np.nan], equal_nan=True)
    assert not np.allclose(data_sw2["y"], [np.nan, np.nan], equal_nan=True)
    x_range_old = (d.figure.x_range.start, d.figure.x_range.end)

    # only show the divergent
    d.show_divergent_only = True
    x_range_new = (d.figure.x_range.start, d.figure.x_range.end)
    assert not np.allclose(x_range_new, x_range_old)

    if NozzleClass is CD_Min_Length_Nozzle:
        assert d.figure.renderers[-1].visible is False
        assert d.figure.renderers[-2].visible is False
        assert d._cb.visible is False
        d.show_characteristic_lines = True
        assert d.figure.renderers[-1].visible is True
        assert d.figure.renderers[-2].visible is True
        assert d._cb.visible is True


def test_update_nozzle_diagram_error_log():
    # verify that NozzleDiagram is able to catch errors and show them on the
    # error_log

    # for non-interactive applications
    n1 = CD_Conical_Nozzle(R0=0, Rj=0, Ri=0.4, Rt=0.2, Re=1.2)
    assert not n1.is_interactive_app
    assert n1.error_log == ""
    with pytest.raises(ValueError):
        n1.throat_radius = 1.5
    assert len(n1.error_log) > 0

    # for interactive applications
    n2 = CD_Conical_Nozzle(R0=0, Rj=0, Ri=0.4, Rt=0.2, Re=1.2)
    assert not n2.is_interactive_app
    d = NozzleDiagram(nozzle=n2)
    assert n2.is_interactive_app
    assert n2.error_log == ""
    # setting a value that forces an error
    n2.throat_radius = 1.4
    assert len(n2.error_log) > 0
    # setting a new value where no errors are raised
    n2.throat_radius = 0.4
    assert n2.error_log == ""


def test_update_de_laval_diagram():
    nozzle = CD_Conical_Nozzle()
    solver = De_Laval_Solver(
        R=287, gamma=1.4, P0=101325, T0=288, nozzle=nozzle, Pb_P0_ratio=0.1)
    assert not nozzle.is_interactive_app
    assert not solver.is_interactive_app
    d = DeLavalDiagram(solver=solver)
    assert nozzle.is_interactive_app
    assert solver.is_interactive_app

    x_range1 = (d.figure.x_range.start, d.figure.x_range.end)
    y_range1 = (d.figure.y_range.start, d.figure.y_range.end)
    assert d.figure.xaxis.axis_label == "Length [m]"
    assert d.figure.yaxis.axis_label == ["Ratios", "Mach number"]
    data1 = d.figure.renderers[0].data_source.data.copy()

    solver.gamma = 1.2
    data2 = d.figure.renderers[0].data_source.data.copy()
    # x-coordinates not equal because the position of the shockwave changes
    assert not np.allclose(data1["x"], data2["x"])
    assert not np.allclose(data1["y"], data2["y"])

    solver.Pb_P0_ratio = 0.2
    data3 = d.figure.renderers[0].data_source.data.copy()
    assert not np.allclose(data2["x"], data3["x"])
    assert not np.allclose(data2["y"], data3["y"])

    nozzle.outlet_radius = 2
    x_range2 = (d.figure.x_range.start, d.figure.x_range.end)
    data4 = d.figure.renderers[0].data_source.data.copy()
    # by changing the radius at the exit while keeping fixed theta_N,
    # the length of the divergent changes.
    assert not np.allclose(data3["x"], data4["x"])
    assert not np.allclose(data3["y"], data4["y"])
    assert not np.allclose(x_range1, x_range2)

    # changing solver will change the results shown on the diagram
    new_nozzle = CD_TOP_Nozzle()
    new_solver = De_Laval_Solver(
        R=287, gamma=1.4, P0=101325, T0=288,
        nozzle=new_nozzle, Pb_P0_ratio=0.1)
    d.solver = new_solver
    data5 = d.figure.renderers[0].data_source.data.copy()
    assert not np.allclose(data4["x"], data5["x"])
    assert not np.allclose(data4["y"], data5["y"])
    x_range_f1_old = (
        d.nozzle_diagram.figure.x_range.start,
        d.nozzle_diagram.figure.x_range.end
    )
    x_range_f2_old = (d.figure.x_range.start, d.figure.x_range.end)
    assert np.allclose(x_range_f1_old, x_range_f2_old)

    # verify that both figures share the same x-range
    d.nozzle_diagram.show_divergent_only = True
    x_range_f1_new = (
        d.nozzle_diagram.figure.x_range.start,
        d.nozzle_diagram.figure.x_range.end
    )
    x_range_f2_new = (d.figure.x_range.start, d.figure.x_range.end)
    assert np.allclose(x_range_f1_new, x_range_f2_new)
    assert not np.allclose(x_range_f1_old, x_range_f1_new)
    assert not np.allclose(x_range_f2_old, x_range_f2_new)

    # verify that it's possible to hide the nozzle geometry
    assert d._nozzle_panel.visible
    d.show_nozzle = False
    assert not d._nozzle_panel.visible

    # verify that all UI controls can be displayed
    d1 = DeLavalDiagram(solver=solver)
    d2 = DeLavalDiagram(solver=solver, _show_all_ui_controls=True)
    # NOTE: do not use methods/attributes starting with '_' in production
    # They are ment to stay private.
    ui_basic = d1._plot_widgets()
    ui_full = d2._plot_widgets()
    assert len(ui_full.objects) == len(ui_basic.objects) + 3


class Test_PressureDeflectionDiagram:
    def setup(self):
        M1 = 3
        theta_2 = 20
        theta_3 = -15

        pdl1 = PressureDeflectionLocus(M=M1, label="1")
        pdl2 = pdl1.new_locus_from_shockwave(theta_2, label="2")
        pdl3 = pdl1.new_locus_from_shockwave(theta_3, label="3")
        theta_intersection, pr_intersection = pdl2.intersection(pdl3)

        return pdl1, pdl2, pdl3, theta_intersection, pr_intersection

    def test_add_state(self):
        d = PressureDeflectionDiagram()
        label, circle = d.add_state(2, 1, "test")
        assert isinstance(label, Label)
        assert isinstance(circle, GlyphRenderer)
        assert isinstance(circle.glyph, Circle)
        assert circle.glyph.x == 2
        assert circle.glyph.y == 1
        assert label.x == 2
        assert label.y == 1
        assert label.text == "test"

    @pytest.mark.parametrize("show_state", [True, False])
    def test_add_locus(self, show_state):
        pdl1, pdl2, pdl3, th, pr = self.setup()
        d = PressureDeflectionDiagram()

        line, label, circle =  d.add_locus(pdl1, show_state=show_state)
        assert isinstance(line, GlyphRenderer)
        assert isinstance(line.glyph, Line)
        if show_state:
            assert isinstance(label, Label)
            assert isinstance(circle, GlyphRenderer)
            assert isinstance(circle.glyph, Circle)
        else:
            assert label is None
            assert circle is None

    @pytest.mark.parametrize("show_state", [True, False])
    def test_add_locus_split_regions(self, show_state):
        pdl1, pdl2, pdl3, th, pr = self.setup()
        d = PressureDeflectionDiagram()

        line_w, line_s, label, circle =  d.add_locus_split_regions(
            pdl1, show_state=show_state)
        for line in [line_w, line_s]:
            assert isinstance(line, GlyphRenderer)
            assert isinstance(line.glyph, Line)
        if show_state:
            assert isinstance(label, Label)
            assert isinstance(circle, GlyphRenderer)
            assert isinstance(circle.glyph, Circle)
        else:
            assert label is None
            assert circle is None

    @pytest.mark.parametrize("num_arrows", [2, 3])
    def test_add_path(self, num_arrows):
        pdl1, pdl2, pdl3, th, pr = self.setup()
        d = PressureDeflectionDiagram()

        line, arrows = d.add_path(
            (pdl1, pdl2.theta_origin), (pdl2, th), num_arrows=num_arrows)
        assert isinstance(line, GlyphRenderer)
        assert isinstance(line.glyph, Line)
        assert isinstance(arrows, Arrow)
        assert len(arrows.source.data["x_start"] == num_arrows)

    @pytest.mark.parametrize(
        "show_state, num_renderers, num_labels, num_arrows", [
        (True, 9, 4, 2),
        (False, 6, 1, 2),
    ])
    def test_complete_diagram(
        self, show_state, num_renderers, num_labels, num_arrows
    ):
        pdl1, pdl2, pdl3, th, pr = self.setup()
        d = PressureDeflectionDiagram()
        d.add_locus(pdl1, show_state=show_state)
        d.add_locus(pdl2, show_state=show_state)
        d.add_locus(pdl3, show_state=show_state)
        d.add_path((pdl1, pdl2.theta_origin), (pdl2, th))
        d.add_path((pdl1, pdl3.theta_origin), (pdl3, th))
        d.add_state(th, pr, "4")
        assert len(d.figure.renderers) == num_renderers
        labels = [l for l in d.figure.center if isinstance(l, Label)]
        arrows = [l for l in d.figure.center if isinstance(l, Arrow)]
        assert len(labels) == num_labels
        assert len(arrows) == num_arrows
