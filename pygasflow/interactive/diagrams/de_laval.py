from bokeh.models import (
    Range1d, HoverTool, LinearAxis,
)
import param
import panel as pn
from pygasflow.nozzles import CD_TOP_Nozzle
from pygasflow.solvers import De_Laval_Solver
from pygasflow.interactive.diagrams.flow_base import BasePlot
from pygasflow.interactive.diagrams.nozzle import NozzleDiagram
import itertools


class DeLavalDiagram(BasePlot, pn.viewable.Viewer):
    """Plot M, P/P0, T/T0, rho/rho0 along the length of a convergent-divergent
    nozzle.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 700,850

        from pygasflow.interactive.diagrams import DeLavalDiagram
        from pygasflow.nozzles import CD_TOP_Nozzle
        from pygasflow.solvers import De_Laval_Solver
        nozzle = CD_TOP_Nozzle(Ri=0.2, Rt=0.1, Re=0.6, K=0.8)
        solver = De_Laval_Solver(
            gamma=1.4, R=287.05, T0=500, P0=101325,
            Pb_P0_ratio=0.2, nozzle=nozzle
        )
        DeLavalDiagram(solver=solver)

    Only show the figure the solutions along the length of the nozzle:

    .. panel-screenshot::
        :large-size: 700,250

        from pygasflow.interactive.diagrams import DeLavalDiagram
        from pygasflow.nozzles import CD_TOP_Nozzle
        from pygasflow.solvers import De_Laval_Solver
        nozzle = CD_TOP_Nozzle(Ri=0.2, Rt=0.1, Re=0.6, K=0.8)
        solver = De_Laval_Solver(
            gamma=1.4, R=287.05, T0=500, P0=101325,
            Pb_P0_ratio=0.2, nozzle=nozzle
        )
        d = DeLavalDiagram(
            solver=solver,
            show_nozzle=False,
            title="Flow in a Thrust Optimized Parabolic nozzle."
        )
        d.show_figure()

    """

    solver = param.ClassSelector(
        class_=De_Laval_Solver,
        doc="The solver instance.")

    tooltips = param.List(
        [("Variable", "@v"), ("Length", "@x"), ("value", "@y")],
        doc="Tooltips used on each line of the plot.")

    labels = param.List([], doc="Labels to be used on the legend.")

    show_nozzle = param.Boolean(
        default=True,
        label="Show nozzle geometry")

    nozzle_diagram = param.ClassSelector(
        class_=NozzleDiagram,
        doc="The nozzle diagram instance to be shown.")

    _show_all_ui_controls = param.Boolean(
        default=False,
        doc="""
            The diagram shows M, p/p0, T/T0, rho/rho0. The latter 3 are
            functions of M, gamma, Pb/P0. If only the diagrams are being shown,
            there is no need to show sliders for P0, T0, R. However, on the
            DeLavalPage, they are fundamental because a table of results is
            also being shown, which depends on them.""")

    def __init__(self, **params):
        params.setdefault("x_label", "Length [m]")
        params.setdefault("y_label", "Ratios")
        params.setdefault("y_range", (0, 1.2))
        params.setdefault("size", (700, 250))
        params.setdefault("labels", [
            "M", "p/p0", "rho/rho0", "T/T0"
        ])
        params.setdefault(
            "solver", De_Laval_Solver(is_interactive_app=True))
        params.setdefault("nozzle_diagram", NozzleDiagram(
            nozzle=params["solver"].nozzle,
            size=params["size"])
        )
        x_range = params["nozzle_diagram"].figure.x_range
        x_range = (x_range.start, x_range.end)
        params.setdefault("x_range", x_range),
        super().__init__(**params)
        # catch exceptions and show them on the error_log
        self.solver.is_interactive_app = True
        self.solver.nozzle.is_interactive_app = True
        # NOTE: I need this in order to change its visibility
        self._nozzle_panel = pn.pane.Bokeh(self.nozzle_diagram.figure)
        self._update_nozzle_diagram_visibility()

    @param.depends("show_nozzle", watch=True)
    def _update_nozzle_diagram_visibility(self):
        self._nozzle_panel.visible = self.show_nozzle

    @param.depends("nozzle_diagram.show_divergent_only", watch=True)
    def _update_show_divergent_only(self):
        self.figure.x_range = self.nozzle_diagram.figure.x_range

    @param.depends(
        "solver",
        "solver.flow_results",
        watch=True
    )
    def update(self):
        self._update_func()

    @param.depends("solver", "solver.nozzle", watch=True)
    def _update_nozzle_diagram(self):
        self.nozzle_diagram.nozzle = self.solver.nozzle

    @param.depends("solver.error_log", "nozzle_diagram.error_log", watch=True)
    def _update_error_log(self):
        msg = ""
        if len(self.solver.error_log) > 0:
            msg += self.solver.error_log
        if len(self.nozzle_diagram.error_log) > 0:
            msg += self.nozzle_diagram.error_log
        self.error_log = msg

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)

        i = 0
        length = self.solver.flow_results[0]
        for l, d in zip(self.labels, self.solver.flow_results[1:]):
            source = {
                "x": length,
                "y": d,
                "v": [l] * len(length)
            }
            kwargs = dict(
                source=source,
                legend_label=l,
                line_width=2,
                line_color=next(colors)
            )
            if i == 0:
                kwargs["y_range_name"] = "mach"
            line = self.figure.line("x", "y", **kwargs)
            self.figure.add_tools(HoverTool(
                tooltips=self.tooltips,
                renderers=[line]
            ))
            i += 1

        # create new y-axis
        y_mach = LinearAxis(
            axis_label="Mach number",
            y_range_name="mach",
        )
        self.figure.add_layout(y_mach, 'right')
        self.figure.extra_y_ranges['mach'] = Range1d(
            0, self.solver.flow_results[1].max() * 1.05)

    def _update_renderers(self):
        length = self.solver.flow_results[0]
        for d, r in zip(self.solver.flow_results[1:], self.figure.renderers):
            old_data = r.data_source.data
            labels = [old_data["v"][0]] * len(d)
            source = {"x": length, "y": d, "v": labels}
            r.data_source.data.update(source)
        self.figure.extra_y_ranges['mach'] = Range1d(
            0, self.solver.flow_results[1].max() * 1.05)
        self.figure.x_range = self.nozzle_diagram.figure.x_range


    def _plot_widgets(self):
        pw = self.nozzle_diagram._plot_widgets()
        if not self._show_all_ui_controls:
            sw = [
                self.solver.param.gamma,
                self.solver.param.Pb_P0_ratio,
            ]
        else:
            sw = [
                self.solver.param.gamma,
                self.solver.param.R,
                self.solver.param.T0,
                self.solver.param.P0,
                self.solver.param.Pb_P0_ratio,
            ]
        idx = 2
        if isinstance(self.nozzle_diagram.nozzle, CD_TOP_Nozzle):
            idx = 1
        return pn.Column(
            *sw, *pw[:idx],
            pn.Card(
                *pw[idx:],
                sizing_mode='stretch_width',
                title="Geometry:",
                collapsed=True
            ),
            self.param.show_nozzle,
            self.nozzle_diagram.param.show_divergent_only,
            self.solver.nozzle.param.N
        )

    @param.depends("solver.current_flow_condition", watch=True, on_init=True)
    def _update_diagram_title(self):
        new_title = "Flow Condition: " + self.solver.current_flow_condition
        self.nozzle_diagram.title = new_title

    @param.depends("solver", "solver.nozzle")
    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.Column(
                    self._nozzle_panel,
                    pn.pane.Bokeh(self.figure),
                )
            )
        )

