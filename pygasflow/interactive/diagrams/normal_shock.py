from bokeh.models import HoverTool
import itertools
import numpy as np
import param
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import normal_shockwave_solver


class NormalShockDiagram(FlowCommon):
    """Interactive component to create a diagram for the properties of the
    flow as it crosses a normal shock wave.
    """

    mach_range = param.Range((1, 8), bounds=(1, 25),
        label="Mach range",
        doc="Minimum and Maximum Mach number for the visualization."
    )

    dividers = param.List([1, 100, 10, 10, 1], doc="""
        Some ratios are going to be much bigger than others at the same
        upstream Mach number. Each number of this list represents a
        quotient for a particular ratio in order to "normalize" the
        visualization.""")

    def __init__(self, **params):
        params.setdefault("title", "Normal Shock Properties")
        params.setdefault("x_label", "Upstream Mach, M1")
        params.setdefault("y_range", (0, 1.5))
        params.setdefault("size", (700, 400))
        params.setdefault("labels", [
            "M2", "p2/p1", "rho2/rho1", "T2/T1", "P02/P01*"
        ])
        params["_solver"] = normal_shockwave_solver
        params["_parameter_name"] = "m1"
        super().__init__(**params)

    def _create_figure(self):
        super()._create_figure(**{
            "x_axis_label": self.x_label,
            "y_axis_label": self.y_label,
            "title": self.title,
            "y_range": self.y_range
        })
        colors = itertools.cycle(self.colors)

        for i, (l, r) in enumerate(zip(self.labels, self.results[1:])):
            current_label = l
            if not np.isclose(self.dividers[i], 1):
                current_label = "(%s) / %s" % (current_label, self.dividers[i])
            source = {
                "xs": self.results[0],
                "ys": r / self.dividers[i],
                "v": [current_label] * len(r)
            }
            line = self.figure.line(
                "xs", "ys",
                source=source,
                legend_label=current_label,
                line_width=2,
                line_color=next(colors)
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

        self._place_legend_outside()

    def _update_figure(self):
        for i, (l, r, renderer) in enumerate(zip(
            self.labels, self.results[1:], self.figure.renderers
        )):
            current_label = l
            if not np.isclose(self.dividers[i], 1):
                current_label = "(%s) / %s" % (current_label, self.dividers[i])
            source = {
                "xs": self.results[0],
                "ys": r / self.dividers[i],
                "v": [current_label] * len(r)
            }
            renderer.data_source.data.update(source)
