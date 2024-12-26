import numpy as np
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles.utils import (
    convergent,
    nozzle_length
)


class CD_Conical_Nozzle(Nozzle_Geometry):
    """
    Convergent-Divergent nozzle with conical divergent.

    Examples
    --------

    Compute the length of a conical nozzle:

    >>> from pygasflow.nozzles import CD_Conical_Nozzle
    >>> Ri, Re, Rt = 0.4, 1.2, 0.2
    >>> nozzle = CD_Conical_Nozzle(Ri, Re, Rt, theta_c=30, theta_N=25)
    >>> nozzle.length
    np.float64(2.53988146753074)

    Change the angle of the divergent section and retrieve the new length
    of the nozzle:

    >>> nozzle.theta_N = 60
    >>> nozzle.length
    np.float64(1.0082903768654763)

    Visualize the nozzle:

    .. bokeh-plot::
        :source-position: above

        from pygasflow.nozzles import CD_Conical_Nozzle
        Ri, Re, Rt = 0.4, 1.2, 0.2
        nozzle = CD_Conical_Nozzle(Ri, Re, Rt, theta_c=30, theta_N=25)
        nozzle.plot(interactive=False)

    """

    _params_to_document = [
        "inlet_radius", "outlet_radius", "throat_radius",
        "theta_c", "theta_N", "N", "geometry_type",
        "inlet_area", "outlet_area", "throat_area",
        "length_convergent", "length_divergent", "length",
        "shockwave_location"
    ]

    def __init__(
        self, Ri=0.4, Re=1.2, Rt=0.2, Rj=0.1, R0=0,
        theta_c=40, theta_N=15, **params
    ):
        params.setdefault("title", "Conical Nozzle")
        super().__init__(
            Ri, Re, Rt, R0, Rj, theta_N=theta_N, theta_c=theta_c, **params
        )

    def __str__(self):
        s = "C-D Conical Nozzle\n"
        s += super().__str__()
        s += "Angles:\n"
        s += f"\ttheta_c\t{self.theta_c}\n"
        s += f"\ttheta_N\t{self.theta_N}\n"
        return s

    def _compute_intersection_points(self):
        Ri, Rt, Re = self.inlet_radius, self.throat_radius, self.outlet_radius
        R0 = self.junction_radius_0
        Rj = self.junction_radius_j

        # divergent length
        self.length_divergent = nozzle_length(Rt, Re, Rj, 1, self.theta_N)

        # find interesting points for the convergent
        x0, y0, x1, y1, xc, yc = convergent(self.theta_c, Ri, R0, Rt, Rj / Rt)
        # convergent length
        self.length_convergent = xc
        self.length = self.length_convergent + self.length_divergent
        # offset to the left, I want x=0 to be throat section
        x0 -= xc
        x1 -= xc

        # point of tangency between the smaller throat junction and the straight line
        xN = Rj * np.sin(np.deg2rad(self.theta_N))
        RN = -np.sqrt(Rj**2 - xN**2) + (Rt + Rj)

        self._intersection_points = {
            "S": [-self.length_convergent, Ri],   # start point
            "0": [x0, y0],  # combustion chamber circle - convergent straight line
            "1": [x1, y1],  # convergent straight line - throat circle
            "N": [xN, RN],  # throat circle - divergent straight line
            "E": [self.length_divergent, Re] # end point
        }

    def build_geometry(self):
        """Discretize the length of the nozzle and compute the nozzle profile.

        Parameters
        ----------
        N : int
            Number of discretization elements along the length of the nozzle. Default to 100.

        Returns
        -------
        x : array_like
            x-coordinate along the nozzle length.
        y : array_like
            y_coordinate of the nozzle wall.
        """
        Ri, Rt, Re = self.inlet_radius, self.throat_radius, self.outlet_radius
        R0 = self.junction_radius_0
        Rj = self.junction_radius_j

        Lc = self.length_convergent
        Ld = self.length_divergent

        x0, y0 = self._intersection_points["0"]
        x1, y1 = self._intersection_points["1"]
        xN, RN = self._intersection_points["N"]

        theta_c = np.deg2rad(self.theta_c)
        theta_N = np.deg2rad(self.theta_N)
        # intercept of the straight line to the throat section
        q = y1 + x1 * np.tan(theta_c)
        # intercept for the straight line of the divergent
        qN = RN - np.tan(theta_N) * xN

        # Compute the points
        x = np.linspace(-Lc, Ld, self.N)
        y = np.zeros_like(x)
        # junction between combustion chamber and convergent
        y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
        # straight line of the convergent
        idx = np.bitwise_and(x > x0, x < x1)
        y[idx] = -np.tan(theta_c) * x[idx] + q
        # junction between convergent and divergent
        idx = np.bitwise_and(x >= x1, x <= xN)
        y[idx] = -np.sqrt(Rj**2 - x[idx]**2) + (Rt + Rj)
        # straight line of the divergent
        y[x > xN] = np.tan(theta_N) * x[x > xN] + qN

        return x, y

    def _get_params_for_ui(self):
        return [
            self.param.throat_radius,
            self.param.outlet_radius,
            self.param.inlet_radius,
            self.param.junction_radius_j,
            self.param.junction_radius_0,
            self.param.theta_c,
            self.param.theta_N,
        ]
