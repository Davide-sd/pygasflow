"""
Contains base classes for diagrams-components.
"""
from bokeh.core.property.vectorization import Value
from bokeh.models import (
    HoverTool,
    Range1d,
    VeeHead,
    Arrow,
    LabelSet,
    ColumnDataSource,
    Line
)
import itertools
from pygasflow.interactive.diagrams.flow_base import (
    CommonParameters,
    BasePlot,
    BasePlot
)
import panel as pn
import param


class ShockCommon(CommonParameters, BasePlot, pn.viewable.Viewer):
    """Common logic for the following plots:

    * Oblique Shock Properties: Mach - β - θ
    * Conical Shock Properties: Mach - β - θ_c
    """

    upstream_mach = param.List([1.1, 1.5, 2, 3, 5, 10, 1e9],
        bounds=(7, 7),  # to keep it simple, otherwise I would have to create
                        # and delete renderers. By setting a fixed length,
                        # I only need to deal with updating renderers.
        label="Upstream Mach numbers:",
        item_type=(float, int),
        doc="Comma separated list of 7 upstream Mach numbers to be shown.")

    show_sonic_line = param.Boolean(True, label="Show sonic line",
        doc="Show the line where the downstream Mach number is M2=1.")

    show_region_line = param.Boolean(True, label="Show region line", doc="""
        Show the line separating the strong solution from the weak solution""")

    tooltips = param.List(
        [("Variable", "@v"), ("θ", "@xs"), ("β", "@ys")],
        doc="Tooltips used on each line of the plot.",
    )

    _ann_arrow_length = param.Number(8, bounds=(1, None),
        doc="Length of the arrow for the sonic annotation.")

    _ann_arrow_start_offset = param.Tuple((0, 1), length=2,
        doc="Offset from the selected position of the start of the arrow.")

    _ann_arrows = param.List([],
        doc="Contains the arrows for the annotations.")

    _ann_labels = param.List([],
        doc="Contains the label set for the annotations.")

    _ann_label_offset = param.Number(11, bounds=(1, None),
        doc="Length of the arrow for the sonic annotation.")

    @param.depends(
        "show_sonic_line", "show_region_line", watch=True, on_init=True
    )
    def update_line_visibility(self):
        if self.figure is not None:
            self.figure.renderers[-2].visible = self.show_sonic_line
            self._ann_arrows[0].visible = self.show_sonic_line
            self._ann_arrows[1].visible = self.show_sonic_line
            self._ann_labels[0].visible = self.show_sonic_line
            self.figure.renderers[-1].visible = self.show_region_line
            self._ann_arrows[2].visible = self.show_region_line
            self._ann_arrows[3].visible = self.show_region_line
            self._ann_labels[1].visible = self.show_region_line

    @param.depends("upstream_mach", watch=True, on_init=True)
    def update_labels(self):
        lbls = [
            f"M1 = {self.upstream_mach[i]}"
            for i in range(len(self.upstream_mach))
        ]
        lbls[-1] = "M1 = ∞"
        self.labels = lbls

        # TODO: as of Bokeh 3.6.0, there is a bug somewhere.
        # Or maybe I coded things in a wrong way. Either way, the legend
        # won't update with new labels. Here is a fix:
        if self.legend is not None:
            for value, entry in zip(self.labels, self.legend.items):
                entry.update(label=Value(value))

    def _plot_widgets(self):
        return [
            self.param.upstream_mach,
            self.param.gamma,
            self.param.N,
            self.param.show_sonic_line,
            self.param.show_region_line
        ]

    @param.depends("upstream_mach", "gamma", "N", watch=True)
    def update(self):
        try:
            self.results = self._compute_results()
            self._update_func()
        except ValueError as err:
            self.error_log = "ValueError: %s" % err

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)
        ann_color = "#000000" if self._theme == "default" else "#ffffff"

        # plot the Mach curves
        max_theta = 0
        for label, source in zip(self.labels, self.results):
            max_theta = max(max_theta, source["xs"].max())
            line = self.figure.line(
                "xs", "ys",
                source=source,
                line_color=next(colors),
                line_width=2,
                legend_label=label
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))
        # adjust x_range
        max_theta = round(max_theta + 5 - (max_theta % 5))
        self.figure.x_range = Range1d(0, max_theta)

        # plot the line M2 = 1
        source = self.results[-4]
        self.figure.line(
            x="xs",
            y="ys",
            source=source,
            line_dash="dotted",
            line_color=ann_color,
            line_width=1,
            visible=self.show_sonic_line
        )

        vh = VeeHead(size=6, fill_color=ann_color, line_color=ann_color)
        idx = self.results[-3]
        a1 = Arrow(
            end=vh,
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] + self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] + self._ann_arrow_length,
            visible=self.show_sonic_line,
            line_color=ann_color
        )
        a2 = Arrow(
            end=vh,
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] - self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] - self._ann_arrow_length,
            visible=self.show_sonic_line,
            line_color=ann_color
        )
        labels_source = ColumnDataSource(data={
            "xs": [source["xs"][idx], source["xs"][idx]],
            "ys": [
                source["ys"][idx] + self._ann_label_offset,
                source["ys"][idx] - self._ann_label_offset,
            ],
            "labels": ["M2 < 1", "M2 > 1"]
        })
        l1 = LabelSet(
            x="xs", y="ys", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=labels_source,
            text_baseline="middle", text_align="center",
            text_color=ann_color,
            text_font_size="12px",
            visible=self.show_sonic_line
        )
        self.figure.add_layout(a1)
        self.figure.add_layout(a2)
        self.figure.add_layout(l1)

        # plot the line passing through (M,theta_max)
        source = self.results[-2]
        self.figure.line(
            x="xs",
            y="ys",
            source=source,
            line_dash="dashed",
            line_color=ann_color,
            line_width=1,
            visible=self.show_region_line
        )

        idx = self.results[-1]
        a3 = Arrow(
            end=vh,
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] + self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] + self._ann_arrow_length,
            visible=self.show_region_line,
            line_color=ann_color
        )
        a4 = Arrow(
            end=vh,
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] - self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] - self._ann_arrow_length,
            visible=self.show_region_line,
            line_color=ann_color
        )
        labels_source = ColumnDataSource(data={
            "xs": [source["xs"][idx], source["xs"][idx]],
            "ys": [
                source["ys"][idx] + self._ann_label_offset,
                source["ys"][idx] - self._ann_label_offset
            ],
            "labels": ["strong", "weak"]
        })
        l2 = LabelSet(
            x="xs", y="ys", text="labels",
            x_offset="x_offset", y_offset="y_offset", source=labels_source,
            text_baseline="middle", text_align="center",
            text_color=ann_color,
            text_font_size="12px",
            visible=self.show_region_line
        )
        self.figure.add_layout(a3)
        self.figure.add_layout(a4)
        self.figure.add_layout(l2)

        self._ann_arrows = [a1, a2, a3, a4]
        self._ann_labels = [l1, l2]

        if self.show_legend_outside:
            self.move_legend_outside()

    def _update_renderers(self):
        # update mach lines
        max_theta = 0
        for label, source, renderer in zip(
            self.labels, self.results, self.figure.renderers
        ):
            max_theta = max(max_theta, source["xs"].max())
            renderer.data_source.data.update(source)
        # adjust x_range
        max_theta = round(max_theta + 5 - (max_theta % 5))
        self.figure.update(x_range=Range1d(0, max_theta))

        # sonic line
        source = self.results[-4]
        self.figure.renderers[-2].data_source.data.update(source)
        idx = self.results[-3]
        self._ann_arrows[0].update(
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] + self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] + self._ann_arrow_length
        )
        self._ann_arrows[1].update(
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] - self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] - self._ann_arrow_length
        )
        self._ann_labels[0].source.data.update({
            "xs": [source["xs"][idx], source["xs"][idx]],
            "ys": [
                source["ys"][idx] + self._ann_label_offset,
                source["ys"][idx] - self._ann_label_offset,
            ]
        })

        # region line
        source = self.results[-2]
        self.figure.renderers[-1].data_source.data.update(source)
        idx = self.results[-1]
        self._ann_arrows[2].update(
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] + self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] + self._ann_arrow_length
        )
        self._ann_arrows[3].update(
            x_start=source["xs"][idx],
            y_start=source["ys"][idx] - self._ann_arrow_start_offset[1],
            x_end=source["xs"][idx],
            y_end=source["ys"][idx] - self._ann_arrow_length
        )
        self._ann_labels[1].source.data.update({
            "xs": [source["xs"][idx], source["xs"][idx]],
            "ys": [
                source["ys"][idx] + self._ann_label_offset,
                source["ys"][idx] - self._ann_label_offset,
            ]
        })

    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )
