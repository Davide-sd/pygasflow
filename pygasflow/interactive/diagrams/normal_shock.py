from bokeh.models import HoverTool
import itertools
import numpy as np
import param
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import normal_shockwave_solver


class NormalShockDiagram(FlowCommon):
    """Interactive component to create a diagram for the properties of the
    flow as it crosses a normal shock wave.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 700,600

        from pygasflow.interactive.diagrams import NormalShockDiagram
        NormalShockDiagram()

    Set custom values to parameters and only show the figure:

    .. panel-screenshot::
        :large-size: 600,350

        from pygasflow.interactive.diagrams import NormalShockDiagram
        d = NormalShockDiagram(
            mach_range=(1, 3), gamma=1.2, size=(600, 350), y_range=(0, 1.05))
        d.show_figure()

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
        params["_parameter_name"] = "mu"
        super().__init__(**params)

    def _create_renderers(self):
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

        if self.show_legend_outside:
            self.move_legend_outside()

    def _update_renderers(self):
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
