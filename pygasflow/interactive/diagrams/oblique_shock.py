from bokeh.models import HoverTool
import numpy as np
from pygasflow.interactive.diagrams.shock_base import ShockCommon
from pygasflow.shockwave import (
    theta_from_mach_beta,
    sonic_point_oblique_shock,
    detachment_point_oblique_shock
)

class ObliqueShockDiagram(ShockCommon):
    """Interactive component to create a diagram for the properties of the
    flow as it crosses an oblique shock wave.

    Examples
    --------

    Show an interactive application:

    .. panel-screenshot::
        :large-size: 700,650

        from pygasflow.interactive.diagrams import ObliqueShockDiagram
        ObliqueShockDiagram()

    Set custom values to parameters and only show the figure:

    .. panel-screenshot::
        :large-size: 700,450

        from pygasflow.interactive.diagrams import ObliqueShockDiagram
        d = ObliqueShockDiagram(
            upstream_mach=[1.1, 1.35, 1.75, 2.25, 3.5, 6, 1e06],
            gamma=1.2,
            show_region_line=False,
            show_sonic_line=False,
            show_minor_grid=True,
            title="Oblique Shock Properties for γ=1.2",
            N=1000
        )
        d.show_figure()

    """

    def __init__(self, **params):
        params.setdefault("title", "Oblique Shock Properties: Mach - β - θ")
        params.setdefault("x_label", "Flow Deflection Angle, θ [deg]")
        params.setdefault("y_label", "Shock Wave Angle, β [deg]")
        params.setdefault("x_range", (0, 50))
        params.setdefault("y_range", (0, 90))
        params.setdefault("size", (700, 400))
        super().__init__(**params)

    def _compute_results(self):
        results = []

        ############################### PART 1 ###############################

        # compute the Mach curves
        for i, m in enumerate(self.upstream_mach):
            beta_min = np.rad2deg(np.arcsin(1 / m))
            betas = np.linspace(beta_min, 90, self.N)
            thetas = theta_from_mach_beta(m, betas, self.gamma)
            source = {
                "xs": thetas,
                "ys": betas,
                "v": [self.labels[i]] * len(betas)
            }
            results.append(source)

        # ############################### PART 2 ###############################

        # compute the line M2 = 1
        M1 = np.logspace(0, 3, 5 * self.N)
        beta_sonic, theta_sonic = sonic_point_oblique_shock(
            M1, self.gamma)
        source = {
            "xs": theta_sonic,
            "ys": beta_sonic,
            "v": [""] * len(M1)
        }
        results.append(source)

        # annotations
        # index of the sonic line where to place the annotation
        desired_x = theta_sonic.max() * self.sonic_ann_location
        idx = np.where(source["xs"] <= desired_x)[0][-1]
        results.append(idx)

        ############################### PART 3 ###############################

        # compute the line passing through (M,theta_max)
        beta, theta_max = detachment_point_oblique_shock(M1, self.gamma)
        source = {
            "xs": theta_max,
            "ys": beta,
            "v": [""] * len(M1)
        }
        results.append(source)

        # index of the region line where to place the annotation
        desired_x = theta_max.max() * self.region_ann_location
        idx = np.where(source["xs"] <= desired_x)[0][-1]
        results.append(idx)

        return results
