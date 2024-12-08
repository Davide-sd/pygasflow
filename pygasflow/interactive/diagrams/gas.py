from bokeh.models import HoverTool, Range1d
from bokeh.plotting import figure
import itertools
import numpy as np
import panel as pn
import param
from pygasflow.interactive.diagrams.flow_base import (
    PlotSettings
)
from pygasflow.solvers import gas_solver, sonic_condition


class GasDiagram(PlotSettings, pn.viewable.Viewer):
    """Plot the relationships Cp=f(gamma, R) and Cv=f(gamma, R).
    """

    select = param.Selector(
        label="Diagram type:",
        objects={
            "gamma - specific heats": 0,
            "R - specific heats": 1,
        },
        default=1
    )

    gamma = param.Number(1.4, bounds=(1, 2),
        inclusive_bounds=(False, True),
        step=0.05,
        label="Ratio of specific heats, γ",
        doc="γ = Cp / Cv"
    )

    gamma_range = param.Range((1.05, 2), bounds=(1, 2),
        step=0.01,
        label="Ratio of specific heats, γ",
        doc="γ = Cp / Cv"
    )

    R = param.Number(287.05, bounds=(0, 5000),
        step=0.05,
        label="Mass-specific gas constant, R",
        doc="[J / (Kg K)]"
    )

    R_range = param.Range((0, 5000), bounds=(0, 5000),
        label="Mass-specific gas constant, R",
        doc="[J / (Kg K)]"
    )

    N = param.Integer(100, bounds=(10, None), softbounds=(10, 1000),
        label="Number of points:", doc="""
        Number of points for each curve. It affects the quality of the
        visualization as well as the speed of updates."""
    )

    def __init__(self, **params):
        params.setdefault("x_label", "Mass-specific gas constant, R, [J / (Kg K)]")
        params.setdefault("y_label", "Specific Heats [J / K]")
        params.setdefault("title", "")
        params.setdefault("x_range", self.R_range)
        super().__init__(**params)

    @param.depends(
        "select",
        "gamma_range", "gamma",
        "R_range", "R", "N",
        watch=True
    )
    def update(self):
        try:
            self._update_func()
        except ValueError as err:
            self.error_log = "ValueError: %s" % err

    def _compute_results(self):
        if self.select == 0:
            gamma = np.linspace(*self.gamma_range, self.N)
            R = self.R
        else:
            gamma = self.gamma
            R = np.linspace(*self.R_range, self.N)
        return gas_solver("gamma", gamma, "R", R, to_dict=True)

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)
        results = self._compute_results()

        if self.select == 0:
            x_key = "gamma"
            tooltips = [("Variable", "@v"), ("gamma", "@xs"), ("value", "@ys")]
        else:
            x_key = "R"
            tooltips = [("Variable", "@v"), ("R", "@xs"), ("value", "@ys")]

        x_var = results[x_key]
        for k in ["Cp", "Cv"]:
            source = {
                "xs": x_var,
                "ys": results[k],
                "v": [k] * len(x_var)
            }
            line = self.figure.line(
                "xs", "ys",
                source=source,
                legend_label=k,
                line_width=2,
                line_color=next(colors)
            )
            self.figure.add_tools(HoverTool(
                tooltips=tooltips,
                renderers=[line]
            ))

        self.move_legend_outside()

    def _update_renderers(self):
        results = self._compute_results()
        x_key = "gamma" if self.select == 0 else "R"
        x_var = results[x_key]
        for k, renderer in zip(["Cp", "Cv"], self.figure.renderers):
            source = {
                "xs": x_var,
                "ys": results[k],
                "v": [k] * len(x_var)
            }
            renderer.data_source.data.update(source)

        self.figure.x_range = Range1d(results[x_key][0], results[x_key][-1])

    @param.depends("select", watch=True)
    def update_xlabel(self):
        label = "Ratio of specific heats, γ"
        if self.select == 1:
            label = "Mass-specific gas constant, R, [J / (Kg K)]"
        self.figure.xaxis.axis_label = label

    def _plot_widgets(self):
        if self.select == 0:
            return [
                self.param.select,
                self.param.gamma_range,
                self.param.R,
                self.param.N
            ]
        return [
            self.param.select,
            self.param.gamma,
            self.param.R_range,
            self.param.N
        ]

    @param.depends("select", on_init=True)
    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )


class SonicDiagram(PlotSettings, pn.viewable.Viewer):
    """Plot the sonic conditions T0/T*=f(gamma), a0/a*=f(gamma),
    p0/p*=f(gamma), rho0/rhoT*=f(gamma).
    """

    gamma_range = param.Range((1.05, 2), bounds=(1, 2),
        step=0.01,
        label="Ratio of specific heats, γ",
        doc="γ = Cp / Cv"
    )

    N = param.Integer(100, bounds=(10, None), softbounds=(10, 1000),
        label="Number of points:", doc="""
        Number of points for each curve. It affects the quality of the
        visualization as well as the speed of updates."""
    )

    tooltips = param.List(
        [("Variable", "@v"), ("gamma", "@xs"), ("value", "@ys")],
        doc="Tooltips used on each line of the plot.",
    )


    def __init__(self, **params):
        params.setdefault("x_label", "Ratio of specific heats, γ")
        params.setdefault("y_label", "Ratios")
        params.setdefault("title", "Sonic condition")
        params.setdefault("x_range", self.gamma_range)
        params.setdefault("N", 10)
        super().__init__(**params)

    @param.depends("gamma_range", "N", watch=True)
    def update(self):
        try:
            self._update_func()
        except ValueError as err:
            self.error_log = "ValueError: %s" % err

    def _compute_results(self):
        gammas = np.linspace(*self.gamma_range, self.N)
        results = sonic_condition(gammas, to_dict=True)
        results["gamma"] = gammas
        return results

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)
        results = self._compute_results()
        labels = ["T0/T*", "a0/a*", "p0/p*", "rho0/rho*"]
        for i, k in enumerate(["trs", "ars", "prs", "drs"]):
            source = {
                "xs": results["gamma"],
                "ys": results[k],
                "v": [labels[i]] * len(results["gamma"])
            }
            line = self.figure.line(
                "xs", "ys",
                source=source,
                legend_label=labels[i],
                line_width=2,
                line_color=next(colors)
            )
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))

        self.move_legend_outside()

    def _update_renderers(self):
        results = self._compute_results()
        labels = ["T0/T*", "a0/a*", "p0/p*", "rho0/rho*"]
        for i, (k, renderer) in enumerate(zip(
            ["trs", "ars", "prs", "drs"],
            self.figure.renderers
        )):
            source = {
                "xs": results["gamma"],
                "ys": results[k],
                "v": [labels[i]] * len(results["gamma"])
            }
            renderer.data_source.data.update(source)
        self.figure.x_range = Range1d(*self.gamma_range)

    def _plot_widgets(self):
        return [
            self.param.gamma_range,
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
