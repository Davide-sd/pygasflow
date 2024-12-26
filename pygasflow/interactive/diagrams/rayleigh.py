from bokeh.models import HoverTool
import itertools
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import rayleigh_solver


class RayleighDiagram(FlowCommon):
    """Interactive component to create a diagram for the Rayleigh flow,
    ie the 1D flow with friction.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 800,500

        from pygasflow.interactive.diagrams import RayleighDiagram
        RayleighDiagram()

    Set custom values to parameters and only show the figure:

    .. panel-screenshot::
        :large-size: 600,350

        from pygasflow.interactive.diagrams import RayleighDiagram
        d = RayleighDiagram(mach_range=(0, 3), gamma=1.2, size=(600, 350))
        d.show_figure()

    """

    def __init__(self, **params):
        params.setdefault("title", "Rayleigh Flow")
        params.setdefault("y_range", (0, 3))
        params.setdefault("labels", [
            "p/p*", "rho/rho*", "T/T*",
            "P0/P0*", "T0/T0*", "U/U*", "(s*-s)/R"
        ])
        params["_solver"] = rayleigh_solver
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

        if self.show_legend_outside:
            self.move_legend_outside()
