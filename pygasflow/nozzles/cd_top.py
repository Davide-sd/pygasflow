import numpy as np
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles.utils import (
    quadratic_bezier_parabola,
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

    Examples
    --------

    .. plot::
       :context: reset
       :format: python
       :include-source: True

       from pygasflow import CD_TOP_Nozzle
       import matplotlib.pyplot as plt
       Ri = 0.4
       Re = 1.2
       Rt = 0.2
       geom = CD_TOP_Nozzle(Ri, Re, Rt, 0.40, 30, 0.7)
       x, y = geom.build_geometry(1000)
       plt.figure()
       plt.plot(x, y)
       for k in geom.Intersection_Points.keys():
           plt.plot(*geom.Intersection_Points[k], 'v')
       plt.xlabel("Length")
       plt.ylabel("Radius")
       plt.axis('equal')
       plt.grid()
       plt.show()

    """

    _title = "TOP Nozzle"

    def __init__(self, Ri, Re, Rt, R0, theta_c, K=0.8,
                    geometry_type="axisymmetric", N=100):
        """
        Parameters
        ----------
        Ri : float
            Inlet radius.
        Re : float
            Exit (outlet) radius.
        Rt : float
            Throat radius.
        R0 : float
            Radius of the junction between combustion chamber and convergent.
        theta_c : float
            Half angle in degrees of the convergent.
        K : float
            Fractional Length of the nozzle with respect to a same exit
            area ratio conical nozzle with 15 deg half-cone angle.
            Default to 0.8.
        geometry_type : string
            Specify the geometry type of the nozzle. Can be either
            ``'axisymmetric'`` or ``'planar'``.
            If ``'planar'`` is specified, Ri, Re, Rt will be considered as
            half of the height of the respective sections (therefore, R is the
            distance from the line of symmetry and the nozzle wall).
            To compute the cross section area, "axisymmetric" uses the formula
            A = pi * r**2, whereas "planar" uses the formula A = 2 * r. Note
            the lack of width in the planar formula, this is because in the
            area ratios it simplifies, hence it is not considered here.
        N : int
            Number of discretization elements along the length of the nozzle. Default to 100.
        """
        if (Ri <= 0) or (Re <= 0) or (Rt <= 0):
            raise ValueError("All input areas must be > 0.")
        if (Ri <= Rt) or (Re <= Rt):
            raise ValueError("Must be Ai > At and Ae > At.")
        if (not isinstance(N, (int))) and (N <= 1):
            raise ValueError("The number of elements for discretization must be N > 1.")
        if K <= 0:
            raise ValueError("The fractional Length must be K > 0.")
        if R0 <= 0:
            raise ValueError("Junction radius between Combustion Chamber and Convergent must be > 0.")
        if (theta_c <= 0) or (theta_c >= 90):
            raise ValueError("It must be 0 < theta_c < 90.")

        super().__init__(Ri, Re, Rt, None, None, geometry_type)
        self._K = K
        self._R0 = R0
        self._theta_c = theta_c

        self._compute_intersection_points()

        x, y = self.build_geometry(N)
        self._length_array = x
        self._wall_radius_array = y
        self._area_ratio_array = 2 * y / self._At
        if self._geometry_type == "axisymmetric":
            self._area_ratio_array = np.pi * y**2 / self._At

    def __str__(self):
        s = "C-D TOP Nozzle\n"
        s += super().__str__()
        s += "Angles:\n"
        s += "\ttheta_c\t{}\n".format(self._theta_c)
        s += "\ttheta_N\t{}\n".format(self._theta_N)
        s += "\ttheta_e\t{}\n".format(self._theta_e)
        return s

    @property
    def Fractional_Length(self):
        return self._K

    @property
    def Intersection_Points(self):
        return self._intersection_points

    def _compute_intersection_points(self):
        Ri, Rt, Re = self._Ri, self._Rt, self._Re
        A_ratio_exit = self._Ae / self._At
        K = self._K
        R0 = self._R0
        theta_c = self._theta_c

        # find Rao's approximation parabola angles
        pangles = Rao_Parabola_Angles()
        theta_N, theta_e = pangles.angles_from_Lf_Ar(K * 100, A_ratio_exit)
        self._theta_N = theta_N
        self._theta_e = theta_e
        theta_N = np.deg2rad(theta_N)
        theta_e = np.deg2rad(theta_e)

        # divergent length
        self._Ld = nozzle_length(Rt, Re, 0.382 * Rt, K)

        # Rao used a radius of 1.5 * Rt for curve 1 (at the left of the throat)
        factor = 1.5
        # find interesting points for the convergent
        x0, y0, x1, y1, xc, yc = convergent(theta_c, Ri, R0, Rt, factor)
        # convergent length
        self._Lc = xc
        # offset to the left, I want x=0 to be throat section
        x0 -= xc
        x1 -= xc

        # point of tangency between the smaller throat junction and the parabola
        xN = 0.382 * Rt * np.sin(theta_N)
        RN = -np.sqrt((0.382 * Rt)**2 - xN**2) + 1.382 * Rt

        self._intersection_points = {
            "S": [-self._Lc, Ri],   # start point
            "0": [x0, y0],  # combustion chamber circle - convergent straight line
            "1": [x1, y1],  # convergent straight line - throat circle
            "orig": [0, Rt],  # throat circle left - throat circle right
            "N": [xN, RN],  # throat circle - parabola
            "E": [self._Ld, Re] # end point
        }

    def build_geometry(self, N):
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
        Ri, Rt, Re = self._Ri, self._Rt, self._Re
        R0 = self._R0

        Lc = self.length_convergent
        Ld = self.length_divergent

        x0, y0 = self._intersection_points["0"]
        x1, y1 = self._intersection_points["1"]
        xN, RN = self._intersection_points["N"]

        theta_c = np.deg2rad(self._theta_c)
        theta_N = np.deg2rad(self._theta_N)
        theta_e = np.deg2rad(self._theta_e)
        # intercept of the straight line to the throat section
        q = y1 + x1 * np.tan(theta_c)

        # Compute the points
        x = np.linspace(-Lc, Ld, N)
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
