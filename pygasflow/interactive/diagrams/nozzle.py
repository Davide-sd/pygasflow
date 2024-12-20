from bokeh.models import (
    ColumnDataSource, Patch, Range1d, HoverTool,  ColorBar,
    LinearColorMapper
)
import colorcet
import numpy as np
import param
import panel as pn
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_Min_Length_Nozzle
)
from pygasflow.interactive.diagrams.flow_base import BasePlot


class NozzleDiagram(BasePlot, pn.viewable.Viewer):
    """Draw a representation of a nozzle.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 600,600

        from pygasflow.interactive.diagrams import NozzleDiagram
        from pygasflow.nozzles import CD_TOP_Nozzle
        nozzle = CD_TOP_Nozzle(Ri=0.2, Rt=0.1, Re=0.6, K=0.8)
        NozzleDiagram(nozzle=nozzle, show_full_nozzle=True)

    Set custom values to parameters and only show the figure:

    .. panel-screenshot::
        :large-size: 600,250

        from pygasflow.interactive.diagrams import NozzleDiagram
        from pygasflow.nozzles import CD_Min_Length_Nozzle
        nozzle = CD_Min_Length_Nozzle(Ri=0.3, Rt=0.2, Re=0.6)
        d = NozzleDiagram(nozzle=nozzle, show_characteristic_lines=True)
        d.show_figure()

    """
    # NOTE: it would be nice to have a class able to accept multiple
    # nozzles, then user would chose the nozzle to be drawn with a
    # dropdown. At the time of writing this, this is a PITA, because
    # while it's possible to dinamically edit a param.Selector, the
    # edits don't get updated in the respecting pn.widgets.Select.
    # So, design choice: this class only plots a single nozzle.

    nozzle = param.ClassSelector(
        class_=Nozzle_Geometry,
        label="Current nozzle to be shown on the diagram.")

    show_area_ratio = param.Boolean(
        default=False,
        label="Show A/A*",
        doc="If True show A/A*, otherwise shows the radius")

    show_full_nozzle = param.Boolean(
        default=False,
        label="Show full nozzle",
        doc="If False, only show the positive y-coordinate.")

    show_divergent_only = param.Boolean(
        default=False,
        label="Show divergent only")

    show_characteristic_lines = param.Boolean(
        default=False,
        label="Show characteristic lines",
        doc="Only used in MOC nozzles."
    )

    characteristic_scatter_kwargs = param.Dict({}, doc="""
        Rendering keywords for the scatter about characteristi lines.""")

    characteristic_cmap = param.ClassSelector(
        class_=(list, tuple, str), doc="""
        A sequence of colors to use as the target palette for mapping.

        This property can also be set as a String, to the name of any of the
        palettes shown in :ref:`bokeh.palettes`.

        If not provided, colorcet.bmy will be used.""")

    def __init__(self, **params):
        params.setdefault("x_label", "Length [m]")
        params.setdefault("y_label",
            "A/A*" if self.show_area_ratio else "radius [m]")
        params.setdefault("size", (600, 250))
        if "nozzle" in params:
            self._user_provided_nozzle = True
        params.setdefault(
            "nozzle", CD_Conical_Nozzle())
        super().__init__(**params)
        self.nozzle.is_interactive_app = True
        self._update_show_characteristic_lines()

    @param.depends(
        "show_full_nozzle", "show_area_ratio", watch=True, on_init=True
    )
    def _update_show_full_nozzle(self):
        # TODO: ideally, I would like to disable `show_full_nozzle` when
        # `show_area_ratio=True`. How to achieve that?
        if self.show_area_ratio:
            self.show_full_nozzle = False

    def _create_renderers(self):
        nozzle, container = self.nozzle.get_points(self.show_area_ratio)
        self.x_range = (container[:, 0].min(), container[:, 0].max())

        glyph_container = Patch(
            x="x", y="y",
            hatch_pattern="/",
            hatch_color="#1c1c1c",
            fill_color="lightgrey",
            hatch_scale=20
        )
        source_container = ColumnDataSource(
            {"x": container[:, 0], "y": container[:, 1]})
        self.figure.add_glyph(source_container, glyph_container)

        glyph_nozzle = Patch(x="x", y="y", fill_color="#a6cee3")
        source_nozzle = ColumnDataSource(
            {"x": nozzle[:, 0], "y": nozzle[:, 1]})
        self.figure.add_glyph(source_nozzle, glyph_nozzle)

        # shockwave
        sw_data, sw_ann_data = self._get_shockwave_coords()
        self.figure.line(
            "x", "y",
            source=ColumnDataSource(sw_data),
            line_color="red",
            line_width=2
        )
        self.figure.text(
            x="x", y="y", text="t",
            y_offset=-5,
            text_color="red",
            background_fill_color="white",
            background_fill_alpha=0.8,
            anchor="bottom_center",
            source=ColumnDataSource(sw_ann_data)
        )
        if isinstance(self.nozzle, CD_Min_Length_Nozzle):
            self._add_characteristic_lines()
        self._update_y_range()

    def _update_renderers(self):
        nozzle, container = self.nozzle.get_points(self.show_area_ratio)
        source_container = {"x": container[:, 0], "y": container[:, 1]}
        source_nozzle = {"x": nozzle[:, 0], "y": nozzle[:, 1]}
        sw_data, sw_ann_data = self._get_shockwave_coords()
        self.figure.renderers[0].data_source.data.update(source_container)
        self.figure.renderers[1].data_source.data.update(source_nozzle)
        self.figure.renderers[2].data_source.data.update(sw_data)
        self.figure.renderers[3].data_source.data.update(sw_ann_data)
        self.figure.yaxis.axis_label = "A/A*" if self.show_area_ratio else "radius [m]"
        if isinstance(self.nozzle, CD_Min_Length_Nozzle):
            self._update_characteristic_lines()
        self._update_y_range()
        self._update_x_range()

    def _create_data_for_characteristic_nodes(self):
        # there are many characteristic lines. By aggregating their data I can
        # use a single line-renderer -> no book-keeping
        xcl, ycl = np.array([]), np.array([])
        for data in self.nozzle.characteristics:
            xcl = np.append(xcl, data["x"])
            ycl = np.append(ycl, data["y"])
            xcl = np.append(xcl, [np.nan])
            ycl = np.append(ycl, [np.nan])

        x, y, p = np.array([]), np.array([]), np.array([])
        for char in self.nozzle.left_runn_chars:
            x = np.append(x, char["x"])
            y = np.append(y, char["y"])
            p = np.append(p, char["M"])

        # plot both left/right running characteristics
        xcl = np.append(xcl, xcl)
        ycl = np.append(ycl, -ycl)
        x = np.append(x, x)
        y = np.append(y, -y)
        p = np.append(p, p)

        if self.show_area_ratio:
            ycl = (2 * ycl) / self.nozzle.throat_area
            y = (2 * y) / self.nozzle.throat_area

        char_lines = {"x": xcl, "y": ycl}
        data = {"x": x, "y": y, "p": p}
        return char_lines, data

    def _add_characteristic_lines(self):
        char_lines, data = self._create_data_for_characteristic_nodes()
        self.figure.line(
            "x", "y",
            source=ColumnDataSource(char_lines),
            line_dash="dotted",
            line_width=0.75,
            line_color="black"
        )
        cmap = self.characteristic_cmap
        if not cmap:
            cmap = colorcet.bmy
        mapper = LinearColorMapper(
            palette=cmap,
            low=min(data["p"]),
            high=max(data["p"])
        )
        kwargs = dict(
            size=8,
            marker="circle"
        )
        kwargs.update(self.characteristic_scatter_kwargs)
        scat = self.figure.scatter(
            "x", "y",
            source=ColumnDataSource(data),
            color={'field': 'p', 'transform': mapper},
            **kwargs
        )
        tooltips = [("Mach number", "@p"), ("x", "@x"), ("y", "@y")]
        self.figure.add_tools(HoverTool(
            tooltips=tooltips,
            renderers=[scat]
        ))
        # TODO: how to place colorbar title to the right of the colorbar?
        self._cb = ColorBar(
            color_mapper=mapper,
            title="Mach number",
            width=8,
        )
        self.figure.add_layout(self._cb, "right")

    def _update_characteristic_lines(self):
        char_lines, data = self._create_data_for_characteristic_nodes()
        self.figure.renderers[-2].data_source.data.update(char_lines)
        self.figure.renderers[-1].data_source.data.update(data)
        # NOTE: in NozzlesPage there is a dropdown select, which allows
        # to change nozzle type. However, when a change happens, it doesn't
        # create a new NozzleDiagram, instead it updates the existing one.
        # Problem: the colorbar might not be present, because it is created
        # inside _create_renderers, which is only executed once, at NozzlesPage
        # loading.
        if hasattr(self, "_cb"):
            self._cb.color_mapper.update(low=min(data["p"]), high=max(data["p"]))

    def _get_shockwave_coords(self):
        x_val, y_val = self.nozzle.shockwave_location
        if any(t is None for t in [x_val, y_val]):
            x_val = np.nan
            y_val = np.nan
        if self.show_area_ratio:
            y_val = self.nozzle._compute_area_ratio(y_val)
        x = [x_val] * 2
        y = [y_val, -y_val]

        sw_data = {"x": x, "y": y}
        sw_ann_data = {k: [v[0]] for k, v in sw_data.items()}
        sw_ann_data["t"] = [" SW "]
        return sw_data, sw_ann_data

    @param.depends(
        "nozzle",
        "nozzle.inlet_radius",
        "nozzle.outlet_radius",
        "nozzle.throat_radius",
        "nozzle.junction_radius_0",
        "nozzle.junction_radius_j",
        "nozzle.theta_c",
        "nozzle.theta_N",
        "nozzle.theta_e",
        "nozzle.fractional_length",
        "nozzle.N",
        "nozzle.geometry_type",
        "nozzle.shockwave_location",
        "nozzle.gamma",
        "nozzle.n_lines",
        "show_area_ratio",
        watch=True
    )
    def update(self):
        self._update_func()

    @param.depends("show_characteristic_lines", watch=True)
    def _update_show_characteristic_lines(self):
        if isinstance(self.nozzle, CD_Min_Length_Nozzle):
            self.figure.renderers[-2].visible = self.show_characteristic_lines
            self.figure.renderers[-1].visible = self.show_characteristic_lines
            self._cb.visible = self.show_characteristic_lines

    @param.depends(
        "show_full_nozzle", "show_area_ratio", watch=True, on_init=True
    )
    def _update_y_range(self):
        if self.figure is not None:
            container = self.figure.renderers[0].data_source.data
            if not self.show_full_nozzle:
                y_range = (0, container["y"].max())
            else:
                y_range = (container["y"].min(), container["y"].max())
            self.figure.y_range = Range1d(*y_range)

    @param.depends("show_divergent_only", watch=True)
    def _update_x_range(self):
        if self.show_divergent_only:
            self.figure.x_range = Range1d(0, self.nozzle.length_divergent)
        else:
            self.figure.x_range = Range1d(
                -self.nozzle.length_convergent,
                self.nozzle.length_divergent
            )

    def _plot_widgets(self):
        controls = self.nozzle._get_params_for_ui()
        controls += [
            self.param.show_area_ratio,
            self.param.show_full_nozzle,
        ]
        if isinstance(self.nozzle, CD_Min_Length_Nozzle):
            controls.append(self.param.show_characteristic_lines)
        return pn.Column(*controls)

    @param.depends("nozzle")
    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )

    @param.depends("nozzle.error_log", watch=True)
    def _update_error_log(self):
        self.error_log = self.nozzle.error_log
