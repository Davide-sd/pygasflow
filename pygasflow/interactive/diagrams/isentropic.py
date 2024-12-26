from bokeh.models import HoverTool, Range1d, LinearAxis
import itertools
import param
from pygasflow.interactive.diagrams.flow_base import FlowCommon
from pygasflow.solvers import isentropic_solver


class IsentropicDiagram(FlowCommon):
    """Interactive component to create a diagram for the isentropic flow.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 800,500

        from pygasflow.interactive.diagrams import IsentropicDiagram
        IsentropicDiagram()

    Set custom values to parameters and only show the figure about the
    ratios:

    .. panel-screenshot::
        :large-size: 600,350

        from pygasflow.interactive.diagrams import IsentropicDiagram
        d = IsentropicDiagram(
            select=1,
            mach_range=(0, 3),
            gamma=1.2,
            size=(600, 350)
        )
        d.show_figure()

    Set custom values to parameters and only show the figure about the
    angles:

    .. panel-screenshot::
        :large-size: 600,350

        from pygasflow.interactive.diagrams import IsentropicDiagram
        d = IsentropicDiagram(
            select=2,
            mach_range=(0, 3),
            gamma=1.2,
            size=(600, 350),
            angle_lines_kwargs={"line_dash": "solid"}
        )
        d.show_figure()

    """

    y_label_right = param.String("Angles [deg]", label="Y Label right:")

    y_range_right = param.Range((0, 3), bounds=(0, 25), label="Y Range Right")

    select = param.Selector(
        label="Diagram type:",
        objects={
            "Ratios + Angles": 0,
            "Ratios": 1,
            "Angles": 2,
        },
        doc="Chose which diagram to show.",
        default=1
    )

    angle_lines_kwargs = param.Dict({}, doc="""
        Keyword arguments to customize the appearance of the Mach Angle
        and Prandtl-Meyer Angle.""")

    ratio_lines_kwargs = param.Dict({}, doc="""
        Keyword arguments to customize the appearance of the ratios.""")

    def __init__(self, **params):
        params.setdefault("select", 0)
        params.setdefault("title", "Isentropic Flow")
        params.setdefault(
            "y_range",
            None if params["select"] == 2 else (0, 3)
        )
        params.setdefault(
            "y_label",
            "Angles [deg]" if params["select"] == 2 else "Ratios"
        )
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
        if (self.figure is not None) and self.show_angles:
            self.figure.yaxis[1].axis_label = self.y_label_right

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)

        if self.select == 0:
            self._add_ratios(colors)
            self._add_angles(colors, "deg")
        elif self.select == 1:
            self._add_ratios(colors)
        else:
            self._add_angles(colors)

        if self.show_legend_outside:
            self.move_legend_outside()

    def _add_ratios(self, colors):
        for l, r in zip(self.labels[:-2], self.results[1:-2]):
            source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
            kwargs = dict(
                legend_label=l,
                line_width=2,
                line_color=next(colors)
            )
            kwargs.update(self.angle_lines_kwargs)
            line = self.figure.line(
                "xs", "ys",
                source=source,
                **kwargs
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

    def _add_angles(self, colors, y_range_name="default"):
        if y_range_name != "default":
            self.figure.extra_y_ranges[y_range_name] = Range1d(0, 90)

        for l, r in zip(self.labels[-2:], self.results[-2:]):
            source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
            kwargs = dict(
                legend_label=l,
                line_width=2,
                line_color=next(colors),
                line_dash="dashed",
            )
            kwargs.update(self.angle_lines_kwargs)
            line = self.figure.line(
                "xs", "ys",
                source=source,
                y_range_name=y_range_name,
                **kwargs
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

        if y_range_name != "default":
            # create new y-axis
            y_deg = LinearAxis(
                axis_label=self.y_label_right,
                y_range_name=y_range_name,
            )
            self.figure.add_layout(y_deg, 'right')

    def _update_renderers(self):
        if self.select in [0, 1]:
            for l, r, renderer in zip(
                self.labels, self.results[1:], self.figure.renderers
            ):
                source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
                renderer.data_source.data.update(source)
        else:
            for l, r, renderer in zip(
                self.labels[-2:], self.results[-2:], self.figure.renderers
            ):
                source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
                renderer.data_source.data.update(source)

