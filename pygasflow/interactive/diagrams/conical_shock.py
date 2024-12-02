from bokeh.models import HoverTool
import numpy as np
from pygasflow.interactive.diagrams.shock_base import ShockCommon
from pygasflow.shockwave import (
    load_data,
    mach_cone_angle_from_shock_angle,
    beta_theta_c_for_unit_mach_downstream,
    max_theta_c_from_mach
)

class ConicalShockDiagram(ShockCommon):
    """Interactive component to create a diagram for the properties of the
    axisymmetric supersonic flow over a sharp cone at zero angle of attack
    to the free stream.
    """

    def __init__(self, **params):
        params.setdefault(
            "title", "Conical Shock Properties: Mach - β - θc")
        params.setdefault("x_label", "Half cone angle, θc [deg]")
        params.setdefault("y_label", "Shock Wave Angle, β [deg]")
        params.setdefault("x_range", (0, 60))
        params.setdefault("y_range", (0, 90))
        params.setdefault("size", (700, 400))
        super().__init__(**params)

    def _compute_results(self):
        results = []

        ############################### PART 1 ###############################

        for j, M1 in enumerate(self.upstream_mach):
            theta_c = np.zeros(self.N)
            # NOTE: to avoid errors in the integration process of Taylor-Maccoll
            # equation, beta should be different than Mach angle and 90deg,
            # hence an offset is applied.
            offset = 1e-08
            theta_s = np.linspace(
                np.rad2deg(np.arcsin(1 / M1)) + offset, 90 - offset, self.N)
            for i, ts in enumerate(theta_s):
                Mc, tc = mach_cone_angle_from_shock_angle(M1, ts, self.gamma)
                theta_c[i] = tc
            theta_c = np.insert(theta_c, 0, 0)
            theta_c = np.append(theta_c, 0)
            theta_s = np.insert(theta_s, 0, np.rad2deg(np.arcsin(1 / M1)))
            theta_s = np.append(theta_s, 90)
            results.append({
                "xs": theta_c,
                "ys": theta_s,
                "v": [self.labels[j]] * len(theta_s)
            })

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
                    beta[i], theta_c[i] = beta_theta_c_for_unit_mach_downstream(
                        m, self.gamma)
                except ValueError:
                    beta[i], theta_c[i] = np.nan, np.nan
            i2 = int(len(beta) / 4)

        results.append({
            "xs": np.asarray(theta_c),
            "ys": np.asarray(beta)
        })
        results.append(i2)

        ############################### PART 3 ###############################

        # Compute the line passing through theta_c_max
        M = np.asarray([1.0005, 1.0025, 1.005, 1.025, 1.05, 1.07, 1.09,
                        1.12, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
                        1.6, 1.75, 2, 2.25, 3, 4, 5, 10, 100, 10000])
        b = np.zeros_like(M)
        tc = np.zeros_like(M)
        for i, m in enumerate(M):
            _, tc[i], b[i] = max_theta_c_from_mach(m, self.gamma)
        tc = np.insert(tc, 0, 0)
        b = np.insert(b, 0, 90)
        results.append({
            "xs": tc,
            "ys": b,
            "v": [""] * len(b)
        })
        results.append(16)
        return results
