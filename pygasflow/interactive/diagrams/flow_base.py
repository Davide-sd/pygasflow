"""
Contains base classes for diagrams-components.
"""

from bokeh.plotting import figure
from bokeh.models import (
    Legend,
    Range1d
)
from bokeh.palettes import Category10
import numpy as np
import panel as pn
import param


class BasePlot(param.Parameterized):
    figure = param.ClassSelector(class_=figure,
        doc="The Bokeh figure to populate")

    size = param.Tuple((800, 300), length=2, doc="""
        Size of the plot, (width, height) in pixel.""")

    error_log = param.String("", doc="""
        Visualize any error that raises from the computation.""")

    colors = param.Tuple(Category10[10], length=10,
        doc="List of categorical colors.")

    _theme = param.String("default", doc="""
        Theme used by the overall application. Useful to choose which
        color to apply to elements of the plot.""")

    _legend = param.ClassSelector(class_=Legend,
        doc="The legend, placed outside of the plotting area.")

    def _create_figure(self, **fig_kwargs):
        if self.figure is None:
            fig_kwargs.setdefault("height", self.size[1])
            fig_kwargs.setdefault("width", self.size[0])
            self.figure = figure(**fig_kwargs)

    def _place_legend_outside(self):
        # hide original legend and create a new one outside of the plot area
        self.figure.legend.visible = False
        legend_items = self.figure.legend.items.copy()
        # NOTE: if I don't clear the elements of the original legend,
        # the hidden legend will still occupy vertical space, which will
        # make hover tool not working over the region covered by the legend
        self.figure.legend.items.clear()
        legend = Legend(items=legend_items)
        # interactive legend
        legend.click_policy = "hide"
        self.figure.add_layout(legend, "right")
        self._legend = legend


class PlotSettings(BasePlot):
    title = param.String("", label="Title:")

    x_label = param.String("M", label="X Label:")

    y_label = param.String("Ratios", label="Y Label:")

    x_range = param.Range((0, 5), label="X Range")

    y_range = param.Range((0, 3), label="Y Range")

    @param.depends("x_range", watch=True)
    def update_x_range(self):
        self.figure.x_range = Range1d(*self.x_range)

    @param.depends("y_range", watch=True)
    def update_y_range(self):
        self.figure.y_range = Range1d(*self.y_range)

    @param.depends("title", watch=True, on_init=True)
    def update_title(self):
        if self.figure is not None:
            self.figure.title.text = self.title

    @param.depends("x_label", watch=True, on_init=True)
    def update_x_label(self):
        if self.figure is not None:
            self.figure.xaxis.axis_label = self.x_label

    @param.depends("y_label", watch=True, on_init=True)
    def update_y_label(self):
        if self.figure is not None:
            self.figure.yaxis[0].axis_label = self.y_label


class CommonParameters(param.Parameterized):
    gamma = param.Number(1.4, bounds=(1, 2),
        inclusive_bounds=(False, True),
        step=0.05,
        label="Ratio of specific heats, γ",
        doc="γ = Cp / Cv"
    )

    N = param.Integer(100, bounds=(10, None), softbounds=(10, 1000),
        label="Number of points:", doc="""
        Number of points for each curve. It affects the quality of the
        visualization as well as the speed of updates."""
    )

    labels = param.List([], doc="Labels to be used on the legend.")

    results = param.List([], doc="Results of the computation.")


class FlowCommon(CommonParameters, PlotSettings, pn.viewable.Viewer):
    mach_range = param.Range((0, 5), bounds=(0, 25),
        label="Mach range",
        doc="Minimum and Maximum Mach number for the visualization."
    )

    tooltips = param.List(
        [("Variable", "@v"), ("Mach", "@xs"), ("value", "@ys")],
        doc="Tooltips used on each line of the plot.",
    )

    _parameter_name = param.String("m", doc="""
        Name of the parameter to be passed to the solver function.""")

    _solver = param.Callable(doc="""
        Solver to be used to compute numerical results.""")

    @param.depends("mach_range", "gamma", "N", watch=True, on_init=True)
    def update(self):
        if self._solver is not None:
            M = np.linspace(self.mach_range[0], self.mach_range[1], self.N)
            try:
                self.results = self._solver(
                    self._parameter_name, M, gamma=self.gamma)
                if self.figure is not None:
                    self._update_figure()
                else:
                    self._create_figure()
            except ValueError as err:
                self.error_log = "ValueError: %s" % err

    def _update_figure(self):
        for l, r, renderer in zip(
            self.labels, self.results[1:], self.figure.renderers
        ):
            source = {"xs": self.results[0], "ys": r, "v": [l] * len(r)}
            renderer.data_source.data.update(source)

    def _plot_widgets(self):
        return [
            self.param.mach_range,
            self.param.gamma,
            self.param.N
        ]

    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )

