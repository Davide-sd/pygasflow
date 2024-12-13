"""

Notes
-----
Online you can find thesis and papers (the latter literally "copied" the
procedure written in the thesis) that build the TOP profile based on the
parabolic equation: x = a * y^2 + b * y + c

Then they set the slope constraint on the start and end points
of the parabolic section, ending up with a system of 3 equations in the
unkowns a,b,c. This is wrong, because the aformentioned parabolic equation
does not consider a rotated parabola!!!!

For example:

    (1) xN = a * yN^2 + b * yN + c
    (2) xE = a * yE^2 + b * yE + c
    (3) dxN / dyN = 2 * a * yN + b = 1 / tan(theta_N)

Here, you are not giving the constraint on the slope of the end point
(xE, yE). Therefore, the computed theta_e will be wrong.
Another example:

    (1) xN = a * yN^2 + b * yN + c
    (2) dxE / dyE = 2 * a * yE + b = 1 / tan(theta_E)
    (3) dxN / dyN = 2 * a * yN + b = 1 / tan(theta_N)

Here, you end up with the correct computed slopes, but xE - xN will
be wrong.

All this because the actual parabola is rotated!!! To solve for a
rotated parabola, you can use the Quadratic Bézier parameterization
or the general parabola equation:
(A * x + C * y)^2 + D * x + E * y + F = 0

"""


import numpy as np
import param
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles.utils import (
    rotated_parabola,
    convergent,
    nozzle_length
)
from pygasflow.nozzles.rao_parabola_angles import Rao_Parabola_Angles


class CD_TOP_Nozzle(Nozzle_Geometry):
    """
    Convergent-Divergent nozzle based on Rao's Thrust Optimized Parabolic
    contours. This is an approximation to the more complex Rao's method of
    characteristic.

    Examples
    --------

    Compute the length of a TOP nozzle:

    >>> from pygasflow.nozzles import CD_TOP_Nozzle
    >>> Ri, Rt = 0.4, 0.2
    >>> nozzle = CD_TOP_Nozzle(Ri, Rt, theta_c=30, K=0.9)
    >>> nozzle.length
    np.float64(3.7946930717892124)

    Change the fractional length of the nozzle and retrieve the new length:

    >>> nozzle.fractional_length = 0.7
    >>> nozzle.length
    np.float64(3.0462712601123014)

    Visualize the nozzle:

    .. bokeh-plot::
        :source-position: above

        from pygasflow.nozzles import CD_TOP_Nozzle
        Ri, Rt = 0.4, 0.2
        nozzle = CD_TOP_Nozzle(Ri, Rt, theta_c=30, K=0.9)
        nozzle.plot(interactive=False)

    """

    _params_to_document = [
        "inlet_radius", "throat_radius", "fractional_length",
        "theta_c", "N", "geometry_type", "theta_N", "theta_e",
        "inlet_area", "outlet_area", "throat_area",
        "length_convergent", "length_divergent", "length",
        "shockwave_location"
    ]

    def __init__(self, Ri=0.4, Rt=0.2, R0=0, theta_c=40, K=0.8, **params):
        params.setdefault("title", "TOP Nozzle")
        Re = params.pop("Re", 6*Rt)
        super().__init__(
            Ri, Re, Rt, R0, theta_c=theta_c, fractional_length=K, **params
        )

    @param.depends(
        "inlet_radius", "outlet_radius", "throat_radius",
        "geometry_type", "theta_c",
        "N", "fractional_length", "junction_radius_0", "junction_radius_j",
        watch=True, on_init=True
    )
    def _update_geometry(self):
        # this method needs to be in this class and not in the parent
        # class in order to break a recursion call. If this was removed, then
        # `Nozzle_geometry._update_geometry` calls `_compute_intersection_points()`,
        # which is going to update `theta_N, theta_e`, which would tring
        # another `Nozzle_geometry._update_geometry`.
        super()._update_geometry()

    def __str__(self):
        s = "C-D TOP Nozzle\n"
        s += super().__str__()
        s += "Angles:\n"
        s += "\ttheta_c\t{}\n".format(self.theta_c)
        s += "\ttheta_N\t{}\n".format(self.theta_N)
        s += "\ttheta_e\t{}\n".format(self.theta_e)
        return s

    def _compute_intersection_points(self):
        Ri, Rt, Re = self.inlet_radius, self.throat_radius, self.outlet_radius
        A_ratio_exit = self.outlet_area / self.throat_area
        K = self.fractional_length
        R0 = self.junction_radius_0
        theta_c = self.theta_c

        # find Rao's approximation parabola angles
        pangles = Rao_Parabola_Angles()
        theta_N, theta_e = pangles.angles_from_Lf_Ar(K * 100, A_ratio_exit)
        with param.edit_constant(self):
            self.param.update(dict(
                theta_N=theta_N,
                theta_e=theta_e
            ))
        theta_N = np.deg2rad(theta_N)
        theta_e = np.deg2rad(theta_e)

        # divergent length
        self.length_divergent = nozzle_length(Rt, Re, 0.382 * Rt, K)

        # Rao used a radius of 1.5 * Rt for curve 1 (at the left of the throat)
        factor = 1.5
        # find interesting points for the convergent
        x0, y0, x1, y1, xc, yc = convergent(theta_c, Ri, R0, Rt, factor)
        # convergent length
        self.length_convergent = xc
        self.length = self.length_convergent + self.length_divergent
        # offset to the left, I want x=0 to be throat section
        x0 -= xc
        x1 -= xc

        # point of tangency between the smaller throat junction and the parabola
        xN = 0.382 * Rt * np.sin(theta_N)
        RN = -np.sqrt((0.382 * Rt)**2 - xN**2) + 1.382 * Rt

        self._intersection_points = {
            "S": [-self.length_convergent, Ri],   # start point
            "0": [x0, y0],  # combustion chamber circle - convergent straight line
            "1": [x1, y1],  # convergent straight line - throat circle
            "orig": [0, Rt],  # throat circle left - throat circle right
            "N": [xN, RN],  # throat circle - parabola
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

        Lc = self.length_convergent
        Ld = self.length_divergent

        x0, y0 = self._intersection_points["0"]
        x1, y1 = self._intersection_points["1"]
        xN, RN = self._intersection_points["N"]

        theta_c = np.deg2rad(self.theta_c)
        theta_N = np.deg2rad(self.theta_N)
        theta_e = np.deg2rad(self.theta_e)
        # intercept of the straight line to the throat section
        q = y1 + x1 * np.tan(theta_c)

        # Compute the points
        x = np.linspace(-Lc, Ld, self.N)
        y = np.zeros_like(x)

        # junction between combustion chamber and convergent
        y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
        # straight line in the convergent
        idx = np.bitwise_and(x > x0, x < x1)
        y[idx] = -np.tan(theta_c) * x[idx] + q
        # curve 1: junction between convergent and divergent, left of throat
        idx = np.bitwise_and(x >= x1, x <= 0)
        y[idx] = -np.sqrt((1.5 * Rt)**2 - x[idx]**2) + 2.5 * Rt
        # curve 2: junction between convergent and divergent, right of throat
        idx = np.bitwise_and(x > 0, x < xN)
        y[idx] = -np.sqrt((0.382 * Rt)**2 - x[idx]**2) + 1.382 * Rt
        # parabola: here I use rotated_parabola for conveniance
        y[x >= xN] = rotated_parabola(
            (xN, RN), (Ld, Re),
            theta_N, theta_e,
            x[x >= xN]
        )

        # P0 = np.asarray([xN, RN])
        # P2 = np.asarray([Ln, Re])
        # t = (x[x >= xN] - xN) / (Ln - xN)
        # xy = Quadratic_Bezier_Parabola(P0, P2, theta_N, theta_e, t)
        # # NOTE: I must update also the x-coordinates, otherwise the plot
        # # would be incorrect.
        # x[x >= xN] = xy[:, 0]
        # y[x >= xN] = xy[:, 1]

        return x, y

    def _get_params_for_ui(self):
        return [
            self.param.throat_radius,
            self.param.inlet_radius,
            self.param.junction_radius_0,
            self.param.theta_c,
            self.param.fractional_length,
        ]
