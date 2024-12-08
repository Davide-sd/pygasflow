from bokeh.models import HoverTool
import itertools
import numpy as np
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import fanno_solver


class FannoDiagram(FlowCommon):
    """Interactive component to create a diagram for the Fanno flow,
    ie the 1D flow with head addition.
    """

    def __init__(self, **params):
        params.setdefault("title", "Fanno Flow")
        params.setdefault("y_range", (0, 3))
        params.setdefault("labels", [
            "p/p*", "rho/rho*", "T/T*",
            "P0/P0*", "U/U*", "4fL*/D", "(s*-s)/R"
        ])
        params["_solver"] = fanno_solver
        super().__init__(**params)

    def _create_renderers(self):
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
