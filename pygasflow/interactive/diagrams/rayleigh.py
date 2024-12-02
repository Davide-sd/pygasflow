from bokeh.models import HoverTool
import itertools
import numpy as np
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import rayleigh_solver


class RayleighDiagram(FlowCommon):
    """Interactive component to create a diagram for the Rayleigh flow,
    ie the 1D flow with friction.
    """

    def __init__(self, **params):
        params.setdefault("title", "Rayleigh Flow")
        params.setdefault("labels", [
            "p/p*", "rho/rho*", "T/T*",
            "P0/P0*", "T0/T0*", "U/U*", "(s*-s)/R"
        ])
        params["_solver"] = rayleigh_solver
        super().__init__(**params)

    def _create_figure(self):
        super()._create_figure(**{
            "x_axis_label": self.x_label,
            "y_axis_label": self.y_label,
            "title": self.title,
            "y_range": self.y_range
        })
        colors = itertools.cycle(self.colors)

        for l, r in zip(self.labels, self.results[1:]):
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

        self._place_legend_outside()