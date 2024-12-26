import numpy as np
from pygasflow.interactive.diagrams.shock_base import ShockCommon
from pygasflow.shockwave import (
    load_data,
    mach_cone_angle_from_shock_angle,
    sonic_point_conical_shock,
    detachment_point_conical_shock
)

class ConicalShockDiagram(ShockCommon):
    """Interactive component to create a diagram for the properties of the
    axisymmetric supersonic flow over a sharp cone at zero angle of attack
    to the free stream.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 700,650

        from pygasflow.interactive.diagrams import ConicalShockDiagram
        ConicalShockDiagram()

    Set custom values to parameters, hide sonic and region lines, and only
    show the figure:

    .. panel-screenshot::
        :large-size: 700,450

        from pygasflow.interactive.diagrams import ConicalShockDiagram
        d = ConicalShockDiagram(
            upstream_mach=[1.1, 1.35, 1.75, 2.25, 3.5, 6, 1e06],
            gamma=1.2,
            add_region_line=False,
            add_sonic_line=False,
            title="Conical Shock Properties for γ=1.2"
        )
        d.show_figure()

    Only shows user-specified upstream Mach numbers:

    .. panel-screenshot::
        :large-size: 700,450

        from pygasflow.interactive.diagrams import ConicalShockDiagram
        d = ConicalShockDiagram(
            add_upstream_mach=False,
            add_region_line=False,
            add_sonic_line=False,
            additional_upstream_mach=[2, 4],
        )
        d.show_figure()

    """

    def __init__(self, **params):
        params.setdefault(
            "title", "Conical Shock Properties: Mach - β - θc")
        params.setdefault("x_label", "Half cone angle, θc [deg]")
        params.setdefault("y_label", "Shock Wave Angle, β [deg]")
        params.setdefault("x_range", (0, 60))
        params.setdefault("y_range", (0, 90))
        params.setdefault("size", (700, 400))
        params.setdefault("show_minor_grid", True)
        params.setdefault("tooltips",
            [("Variable", "@v"), ("θc", "@x"), ("β", "@y"), ("Region", "@r")])
        params.setdefault("upstream_mach",
            [1.05, 1.15, 1.4, 2, 3, 5, 1000000000.0])
        super().__init__(**params)

    def _compute_mach_line_data(self, M1, label):
        theta_c = np.zeros(self.N)
        # NOTE: to avoid errors in the integration process of Taylor-Maccoll
        # equation, beta should be different than Mach angle and 90deg,
        # hence an offset is applied.
        offset = 1e-08
        beta = np.linspace(
            np.rad2deg(np.arcsin(1 / M1)) + offset, 90 - offset, self.N)
        for i, b in enumerate(beta):
            Mc, tc = mach_cone_angle_from_shock_angle(M1, b, self.gamma)
            theta_c[i] = tc
        theta_c = np.insert(theta_c, 0, 0)
        theta_c = np.append(theta_c, 0)
        beta = np.insert(beta, 0, np.rad2deg(np.arcsin(1 / M1)))
        beta = np.append(beta, 90)
        beta_d, _ = detachment_point_conical_shock(M1, self.gamma)
        region = np.empty(len(beta), dtype=object)
        idx = beta <= beta_d
        region[idx] = "weak"
        region[~idx] = "strong"
        source = {
            "x": theta_c,
            "y": beta,
            "v": [label] * len(beta),
            "r": region
        }
        return source

    def _compute_results(self):
        results = []

        ############################### PART 1 ###############################

        for j, M1 in enumerate(self.upstream_mach):
            source = self._compute_mach_line_data(M1, self.labels[j])
            results.append(source)

        ############################### PART 2 ###############################

        try:
            M, beta, theta_c = load_data(self.gamma)
            i2 = 54
        except FileNotFoundError:
            # If there is no data for M2=1, we just need to generate it.
            # IT IS SLOW!!!
            M = np.asarray([
                1, 1.005, 1.05, 1.2, 1.3, 1.4, 1.5,
                1.65, 1.8, 2, 3, 4, 5, 10, 10000])
            beta = np.zeros_like(M)
            theta_c = np.zeros_like(M)
            for i, m in enumerate(M):
                try:
                    beta[i], theta_c[i] = sonic_point_conical_shock(
                        m, self.gamma)
                except ValueError:
                    beta[i], theta_c[i] = np.nan, np.nan
            i2 = int(len(beta) / 4)

        theta_c = np.asarray(theta_c)
        results.append({
            "x": theta_c,
            "y": np.asarray(beta)
        })

        ############################### PART 3 ###############################

        # Compute the line passing through theta_c_max
        M = np.asarray([1.0005, 1.0025, 1.005, 1.025, 1.05, 1.07, 1.09,
                        1.12, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
                        1.6, 1.75, 2, 2.25, 3, 4, 5, 10, 100, 10000])
        b = np.zeros_like(M)
        tc = np.zeros_like(M)
        for i, m in enumerate(M):
            b[i], tc[i] = detachment_point_conical_shock(m, self.gamma)
        tc = np.insert(tc, 0, 0)
        b = np.insert(b, 0, 90)
        results.append({
            "x": tc,
            "y": b,
            "v": [""] * len(b)
        })

        ############################### PART 4 ###############################

        for m in self.additional_upstream_mach:
            source = self._compute_mach_line_data(m, f"M1 = {m}")
            results.append(source)

        return results
