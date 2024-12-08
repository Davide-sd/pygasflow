from bokeh.models import HoverTool, Range1d, LinearAxis
import itertools
import numpy as np
import param
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import isentropic_solver


class IsentropicDiagram(FlowCommon):
    """Interactive component to create a diagram for the isentropic flow.
    """

    y_label_right = param.String("Angles [deg]", label="Y Label right:")

    y_range_right = param.Range((0, 3), bounds=(0, 25), label="Y Range Right")

    def __init__(self, **params):
        params.setdefault("title", "Isentropic Flow")
        params.setdefault("y_range", (0, 3))
        params.setdefault("labels", [
            "P/P0", "rho/rho0", "T/T0",
            "p/p*", "rho/rho*", "T/T*", "U/U*", "A/A*",
            "Mach Angle", "Prandtl-Meyer Angle"
        ])
        params["_solver"] = isentropic_solver
        super().__init__(**params)

    @param.depends("y_range_right", watch=True)
    def update_y_range_right(self):
        # TODO: this is wrong. How do I set the range on the other
        # y-axis?
        self.figure.y_range = Range1d(*self.y_range)

    @param.depends("y_label_right", watch=True, on_init=True)
    def update_y_label_right(self):
        if self.figure is not None:
            self.figure.yaxis[1].axis_label = self.y_label_right

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)

        for l, r in zip(self.labels[:-2], self.results[1:-2]):
            source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
            line = self.figure.line(
                "xs", "ys",
                source=source,
                legend_label=l,
                line_width=2,
                line_color=next(colors)
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

        self.figure.extra_y_ranges['deg'] = Range1d(0, 90)
        for l, r in zip(self.labels[-2:], self.results[-2:]):
            source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
            line = self.figure.line(
                "xs", "ys",
                source=source,
                legend_label=l,
                line_width=2,
                line_color=next(colors),
                line_dash="dashed",
                y_range_name="deg",
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

        # create new y-axis
        y_deg = LinearAxis(
            axis_label=self.y_label_right,
            y_range_name="deg",
        )
        self.figure.add_layout(y_deg, 'right')

        self.move_legend_outside()
