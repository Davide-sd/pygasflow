from bokeh.core.property.vectorization import Value
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, GlyphRenderer
import itertools
import numpy as np
import panel as pn
import param
from pygasflow.generic import characteristic_mach_number
from pygasflow.interactive.diagrams.flow_base import BasePlot, CommonParameters
from pygasflow.shockwave import (
    shock_polar,
    shock_polar_equation,
    max_theta_from_mach,
    beta_from_mach_theta
)
from pygasflow.solvers import shockwave_solver
from scipy.optimize import minimize_scalar, bisect


def _func_to_minimize(Vx_as, M1s, theta_max, gamma):
    """This function is used in a minimization problem to find the value
    of Vx/a* at which `theta_max` occurs.

    Parameters
    ----------
    Vx_as : float
        The value of Vx/a*
    M1s : float
        Characteristic upstream Mach number
    theta_max : float
    """
    Vy_as = shock_polar_equation(Vx_as, M1s, gamma)
    theta = np.degrees(np.atan(Vy_as / Vx_as))
    return 1 / (theta - theta_max)


def _func_to_bisect(Vx_as, M1s, theta_target, gamma):
    """This function is used in a root finding problem, to find the value
    of Vx/a* at which `theta_target` occurs.

    Parameters
    ----------
    Vx_as : float
        The value of Vx/a*
    M1s : float
        Characteristic upstream Mach number
    theta_target : float
    """
    Vy_as = shock_polar_equation(Vx_as, M1s, gamma)
    theta = np.degrees(np.atan(Vy_as / Vx_as))
    return theta - theta_target


class ShockPolarDiagram(CommonParameters, BasePlot, pn.viewable.Viewer):
    """Create a shock polar diagram.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 600,525

        from pygasflow.interactive.diagrams import ShockPolarDiagram
        ShockPolarDiagram(mach_number=3, gamma=1.2, theta=10)

    Set custom values to parameters and only show the figure:

    .. panel-screenshot::
        :large-size: 600,350

        from pygasflow.interactive.diagrams import ShockPolarDiagram
        d = ShockPolarDiagram(mach_number=4, gamma=1.4, theta=30)
        d.show_figure()

    """
    show_mach_at_infinity = param.Boolean(False,
        doc="Toggle the visibility of the curve at Mach infinity.",
        label="Show M = ∞")

    show_theta_line = param.Boolean(True,
        doc="Show the line representing the deflection angle, θ.",
        label="Show deflection angle line")

    show_beta_line = param.Boolean(True,
        doc="Show the line representing the shock wave angle, β.",
        label="Show shock wave angle line")

    show_sonic_circle = param.Boolean(True, doc="""
        Show the line representing the sonic circle.""")

    mach_number = param.Number(5, bounds=(1, None), softbounds=(1, 10),
        step=0.05,
        doc="Mach number to represent on the diagram.",
        label="Upstream Mach number")

    theta = param.Number(15, bounds=(0, 90), step=0.05,
        doc="Flow deflection angle, θ [deg]",
        label="Flow deflection angle, θ [deg]")

    include_mirror = param.Boolean(False, doc="""
        If True, shows results both on +Y and -Y axis.""")

    characteristic_mach_number = param.Number(constant=True,
        doc="Get the characteristic upstream Mach number.")

    theta_max = param.Number(constant=True, doc="""
        Get the maximum deflection angle associated with the current
        upstream Mach number.""")

    helper_line_kwargs = param.Dict({}, doc="""
        Rendering keywords for helper lines.""")

    helper_scatter_kwargs = param.Dict({}, doc="""
        Rendering keywords for helper lines.""")

    # pointers to renderers
    _mach_line = param.ClassSelector(class_=GlyphRenderer)
    _mach_inf = param.ClassSelector(class_=GlyphRenderer)
    _sonic_circle = param.ClassSelector(class_=GlyphRenderer)
    _theta_line = param.ClassSelector(class_=GlyphRenderer)
    _beta_weak_line = param.ClassSelector(class_=GlyphRenderer)
    _beta_strong_line = param.ClassSelector(class_=GlyphRenderer)
    _helper_line_1 = param.ClassSelector(class_=GlyphRenderer)
    _helper_scatter_1 = param.ClassSelector(class_=GlyphRenderer)
    _helper_line_2 = param.ClassSelector(class_=GlyphRenderer)
    _helper_scatter_22 = param.ClassSelector(class_=GlyphRenderer)
    _helper_line_3 = param.ClassSelector(class_=GlyphRenderer)
    _helper_scatter_3 = param.ClassSelector(class_=GlyphRenderer)
    _helper_line_4 = param.ClassSelector(class_=GlyphRenderer)
    _helper_scatter_4 = param.ClassSelector(class_=GlyphRenderer)

    def __init__(self, **params):
        params.setdefault("size", (600, 350))
        params.setdefault("x_label", "Vx / a*")
        params.setdefault("y_label", "Vy / a*")
        params.setdefault("title", "Shock Polar Diagram")
        super().__init__(**params)
        self.figure.match_aspect = True
        if self.show_legend_outside:
            self.move_legend_outside()

    @param.depends("mach_number", "gamma", watch=True, on_init=True)
    def _update_upstream_char_mach_number(self):
        with param.edit_constant(self):
            self.theta_max = max_theta_from_mach(self.mach_number, self.gamma)
            self.characteristic_mach_number = characteristic_mach_number(
                self.mach_number, self.gamma)

    @param.depends(
        "mach_number", "gamma", "theta", "N", "include_mirror", watch=True
    )
    def update(self):
        try:
            if self.theta > self.theta_max:
                raise ValueError(
                    f"For M={self.mach_number}, gamma={self.gamma}, it must be"
                    f" theta <= {self.theta_max}."
                )
            Vx_as, Vy_as = shock_polar(
                self.mach_number, self.gamma, self.N,
                include_mirror=self.include_mirror
            )
            thetas = np.degrees(np.atan2(Vy_as, Vx_as))

            # Compute data for tooltips.
            # find where theta_max occurs
            Vx_as_at_theta_max = minimize_scalar(
                _func_to_minimize,
                bounds=(Vx_as.min(), Vx_as.max()),
                method="bounded",
                args=(
                    self.characteristic_mach_number,
                    self.theta_max,
                    self.gamma
                )
            )
            Vx_as_at_theta_max = Vx_as_at_theta_max.x

            pr = np.zeros_like(Vx_as)
            tpr = np.zeros_like(Vx_as)
            tr = np.zeros_like(Vx_as)
            dr = np.zeros_like(Vx_as)
            m2 = np.zeros_like(Vx_as)
            betas = np.zeros_like(Vx_as)
            for i, (v, t) in enumerate(zip(Vx_as, abs(thetas))):
                res = shockwave_solver(
                    "mu", self.mach_number,
                    "theta", t,
                    gamma=self.gamma,
                    flag="weak" if v >= Vx_as_at_theta_max else "strong"
                )
                _, _, m2[i], _, betas[i], _, pr[i], dr[i], tr[i], tpr[i] = res

            mach_line_data = {
                "xs": Vx_as,
                "ys": Vy_as,
                "theta": thetas,
                "beta": betas,
                "pr": pr,
                "dr": dr,
                "tr": tr,
                "tpr": tpr,
                "m2": m2,
            }

            Vx_as_strong = bisect(
                _func_to_bisect,
                a=Vx_as.min(),
                b=Vx_as_at_theta_max,
                args=(self.characteristic_mach_number, self.theta, self.gamma)
            )
            Vy_as_strong = shock_polar_equation(
                Vx_as_strong, self.characteristic_mach_number, self.gamma)
            Vx_as_weak = bisect(
                _func_to_bisect,
                a=Vx_as_at_theta_max,
                b=Vx_as.max(),
                args=(self.characteristic_mach_number, self.theta, self.gamma)
            )
            Vy_as_weak = shock_polar_equation(
                Vx_as_weak, self.characteristic_mach_number, self.gamma)
            line_theta_data = {
                "xs": [0, Vx_as_strong, Vx_as_weak],
                "ys": [0, Vy_as_strong, Vy_as_weak]
            }

            beta = beta_from_mach_theta(
                self.mach_number, self.theta, self.gamma)
            beta = {k: np.deg2rad(v) for k, v in beta.items()}
            m1 = -Vy_as_strong / (Vx_as.max() - Vx_as_strong)
            q1 = -m1 * Vx_as.max()
            m2 = np.sin(beta["strong"]) / np.cos(beta["strong"])
            x_strong = q1 / (m2 - m1)
            y_strong = m2 * x_strong
            m3 = -Vy_as_weak / (Vx_as.max() - Vx_as_weak)
            q3 = -m3 * Vx_as.max()
            m4 = np.sin(beta["weak"]) / np.cos(beta["weak"])
            x_weak = q3 / (m4 - m3)
            y_weak = m4 * x_weak
            line_beta_strong = {
                "xs": [0, x_strong],
                "ys": [0, y_strong],
            }
            line_beta_weak = {
                "xs": [0, x_weak],
                "ys": [0, y_weak],
            }
            helper_line_1 = {
                "xs": [Vx_as_weak, Vx_as.max()],
                "ys": [Vy_as_weak, 0],
            }
            helper_line_2 = {
                "xs": [Vx_as_strong, Vx_as.max()],
                "ys": [Vy_as_strong, 0],
            }
            helper_line_3 = {
                "xs": [x_weak, Vx_as_weak if self.show_theta_line else Vx_as.max()],
                "ys": [y_weak, Vy_as_weak if self.show_theta_line else 0],
            }
            helper_line_4 = {
                "xs": [x_strong, Vx_as_strong if self.show_theta_line else Vx_as.max()],
                "ys": [y_strong, Vy_as_strong if self.show_theta_line else 0],
            }

            self.results = [
                mach_line_data,
                line_theta_data,
                line_beta_strong,
                line_beta_weak,
                helper_line_1,
                helper_line_2,
                helper_line_3,
                helper_line_4,
            ]
            self._update_func()
            self.error_log = ""
        except ValueError as err:
            self.error_log = "ValueError: %s" % err

    def _create_renderers(self):
        colors = itertools.cycle(self.colors)
        helper_line_kwargs = dict(
            line_color="#000000",
            line_dash="dotted",
        )
        helper_line_kwargs.update(self.helper_line_kwargs)
        helper_scatter_kwargs = dict(
            line_color="#000000",
            fill_color="#000000",
            size=5,
        )
        helper_scatter_kwargs.update(self.helper_scatter_kwargs)

        if self.show_sonic_circle:
            t = np.linspace(
                -np.pi/2 if self.include_mirror else 0, np.pi/2, self.N)
            self._sonic_circle = self.figure.line(
                "xs", "ys",
                source=ColumnDataSource(data={
                    "xs": np.cos(t),
                    "ys": np.sin(t)
                }),
                line_color=next(colors),
                line_width=2,
                legend_label="Sonic circle",
            )

        self._mach_line = self.figure.line(
            "xs", "ys",
            source=ColumnDataSource(data=self.results[0]),
            line_color=next(colors),
            line_width=2,
            legend_label=f"M = {round(self.mach_number, 3)}"
        )
        self.figure.add_tools(HoverTool(
            tooltips=[
                ("M1", f"{self.mach_number}"),
                ("M2", "@m2"),
                ("Vx / a*", "@xs"),
                ("Vy / a*", "@ys"),
                ("θ [deg]", "@theta"),
                ("β [deg]", "@beta"),
                ("P2 / P1", "@pr"),
                ("T2 / T1", "@tr"),
                ("rho2 / rho1", "@dr"),
                ("P02 / P01", "@tpr"),
            ],
            renderers=[self._mach_line]
        ))

        if self.show_theta_line:
            self._theta_line = self.figure.line(
                "xs", "ys",
                source=ColumnDataSource(data=self.results[1]),
                line_color=next(colors),
                line_width=2,
                legend_label="θ"
            )
            self._helper_line_1 = self._add_line(
                self.results[4], **helper_line_kwargs)
            self._helper_scatter_1 = self._add_scatter(
                self.results[4], **helper_scatter_kwargs)
            self._helper_line_2 = self._add_line(
                self.results[5], **helper_line_kwargs)
            self._helper_scatter_2 = self._add_scatter(
                self.results[5], **helper_scatter_kwargs)
            self._link_visibility(
                self._theta_line, self._helper_line_1, self._helper_scatter_1)
            self._link_visibility(
                self._theta_line, self._helper_line_2, self._helper_scatter_2)

        if self.show_beta_line:
            self._beta_strong_line = self.figure.line(
                "xs", "ys",
                source=ColumnDataSource(data=self.results[2]),
                line_color=next(colors),
                line_width=2,
                legend_label="β (strong)"
            )
            self._beta_weak_line = self.figure.line(
                "xs", "ys",
                source=ColumnDataSource(data=self.results[3]),
                line_color=next(colors),
                line_width=2,
                legend_label="β (weak)"
            )
            self._helper_line_3 = self._add_line(
                self.results[6], **helper_line_kwargs)
            self._helper_scatter_3 = self._add_scatter(
                self.results[6], **helper_scatter_kwargs)
            self._helper_line_4 = self._add_line(
                self.results[7], **helper_line_kwargs)
            self._helper_scatter_4 = self._add_scatter(
                self.results[7], **helper_scatter_kwargs)
            self._link_visibility(
                self._beta_weak_line, self._helper_line_3, self._helper_scatter_3)
            self._link_visibility(
                self._beta_strong_line, self._helper_line_4, self._helper_scatter_4)

        if self.show_mach_at_infinity:
            Vx_inf_as, Vy_inf_as = shock_polar(
                1e06, self.gamma, self.N,
                include_mirror=self.include_mirror
            )
            self._mach_inf = self.figure.line(
                "xs", "ys",
                source=ColumnDataSource(data={
                    "xs": Vx_inf_as,
                    "ys": Vy_inf_as,
                }),
                line_color=next(colors),
                line_width=2,
                legend_label="M = ∞"
            )

    def _add_scatter(self, data, **kwargs):
        return self.figure.scatter(
            "xs", "ys",
            source=ColumnDataSource(data),
            **kwargs
        )

    def _add_line(self, data, **kwargs):
        return self.figure.line(
            "xs", "ys",
            source=ColumnDataSource(data),
            **kwargs
        )

    def _link_visibility(self, primary_line, secondary_line, scatter):
        callback = CustomJS(
            args=dict(
                primary_line=primary_line,
                secondary_line=secondary_line,
                scatter=scatter
            ),
            code="""
            secondary_line.visible = primary_line.visible;
            scatter.visible = primary_line.visible;
            """
        )
        primary_line.js_on_change("visible", callback)

    def _update_renderers(self):
        self._mach_line.data_source.data.update(self.results[0])
        if self._theta_line:
            self._theta_line.data_source.data.update(self.results[1])
            self._helper_line_1.data_source.data.update(self.results[4])
            self._helper_scatter_1.data_source.data.update(self.results[4])
            self._helper_line_2.data_source.data.update(self.results[5])
            self._helper_scatter_2.data_source.data.update(self.results[5])
        if self._beta_strong_line:
            self._beta_strong_line.data_source.data.update(self.results[2])
            self._beta_weak_line.data_source.data.update(self.results[3])
            self._helper_line_3.data_source.data.update(self.results[6])
            self._helper_scatter_3.data_source.data.update(self.results[6])
            self._helper_line_4.data_source.data.update(self.results[7])
            self._helper_scatter_4.data_source.data.update(self.results[7])
        legend = self.legend if self.legend is not None else self.figure.legend
        idx = 1 if self.show_sonic_circle else 0
        legend.items[idx].update(label=Value(value=f"M = {round(self.mach_number, 2)}"))


    def _plot_widgets(self):
        return [
            self.param.mach_number,
            self.param.theta,
            self.param.gamma,
        ]

    def __panel__(self):
        return pn.Column(
            pn.Row(pn.pane.Str(self.param.error_log)),
            pn.FlexBox(
                pn.GridBox(*self._plot_widgets(), ncols=1),
                pn.pane.Bokeh(self.figure)
            )
        )
