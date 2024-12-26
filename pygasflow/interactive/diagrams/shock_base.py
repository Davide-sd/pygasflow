"""
Contains base classes for diagrams-components.
"""
from bokeh.core.property.vectorization import Value
from bokeh.models import (
    HoverTool,
    Range1d,
    GlyphRenderer,
)
import itertools
from pygasflow.interactive.diagrams.flow_base import (
    CommonParameters,
    BasePlot,
)
from numbers import Number
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

    additional_upstream_mach = param.List(item_type=Number, doc="""
        User can provides additional upstream Mach numbers to be plotted on
        the chart. This parameter must be set at instantiation and should
        be changed afterwards.""")

    add_upstream_mach = param.Boolean(True, doc="""
        If True, add the default upstream Mach numbers to the chart.
        Otherwise, only add the additional user-provided Mach numbers.
        This parameter must be set at instantiation and should
        be changed afterwards.""")

    add_sonic_line = param.Boolean(True, doc="""
        Add the line where the downstream Mach number is M2=1. This parameter
        must be set at instantiation and should be changed afterwards.""")

    add_region_line = param.Boolean(True, doc="""
        Add the line separating the strong solution from the weak solution.
        This parameter must be set at instantiation and should
        be changed afterwards.""")

    tooltips = param.List(
        [("Variable", "@v"), ("θ", "@x"), ("β", "@y"), ("Region", "@r")],
        doc="Tooltips used on each line of the plot.",
    )

    # pointers to renderers
    _sonic_line = param.ClassSelector(class_=GlyphRenderer)
    _region_line = param.ClassSelector(class_=GlyphRenderer)
    _upstream_mach_lines = param.List(item_type=GlyphRenderer)
    _additional_mach_lines = param.List(item_type=GlyphRenderer)

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

        if self.add_upstream_mach:
            # plot the Mach curves
            max_theta = 0
            for label, source in zip(self.labels, self.results):
                max_theta = max(max_theta, source["x"].max())
                line = self.figure.line(
                    "x", "y",
                    source=source,
                    line_color=next(colors),
                    line_width=2,
                    legend_label=label
                )
                self.figure.add_tools(HoverTool(
                    tooltips=self.tooltips,
                    renderers=[line]
                ))
                self._upstream_mach_lines.append(line)
            # adjust x_range
            self.figure.x_range = Range1d(
                0, round(max_theta + 5 - (max_theta % 5)))

        if self.add_sonic_line:
            # plot the line M2 = 1
            source = self.results[len(self.upstream_mach)]
            self._sonic_line = self.figure.line(
                x="x",
                y="y",
                source=source,
                line_dash="dotted",
                line_color=ann_color,
                line_width=1,
                legend_label="sonic line"
            )

        if self.add_region_line:
            # plot the line passing through (M,theta_max)
            source = self.results[len(self.upstream_mach) + 1]
            self._region_line = self.figure.line(
                x="x",
                y="y",
                source=source,
                line_dash="dashed",
                line_color=ann_color,
                line_width=1,
                legend_label="region line"
            )

        for source in self.results[len(self.upstream_mach) + 2:]:
            line = self.figure.line(
                "x", "y",
                source=source,
                line_color=next(colors),
                line_width=2,
                legend_label=source["v"][0]
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))
            self._additional_mach_lines.append(line)

        if self.show_legend_outside:
            self.move_legend_outside()

    def _update_renderers(self):
        if self.add_upstream_mach:
            # update mach lines
            max_theta = 0
            for label, source, renderer in zip(
                self.labels, self.results, self._upstream_mach_lines
            ):
                max_theta = max(max_theta, source["x"].max())
                renderer.data_source.data.update(source)
            # adjust x_range
            max_theta = round(max_theta + 5 - (max_theta % 5))
            self.figure.update(x_range=Range1d(0, max_theta))

        if self.add_sonic_line:
            source = self.results[len(self.upstream_mach)]
            self._sonic_line.data_source.data.update(source)

        if self.add_region_line:
            source = self.results[len(self.upstream_mach) + 1]
            self._region_line.data_source.data.update(source)

        for source, renderer in zip(
            self.results[len(self.upstream_mach) + 2:],
            self._additional_mach_lines
        ):
            renderer.data_source.data.update(source)

    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )
