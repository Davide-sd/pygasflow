import numpy as np
import param
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles.utils import convergent
from pygasflow.isentropic import (
    prandtl_meyer_angle,
    m_from_prandtl_meyer_angle,
    m_from_critical_area_ratio,
    mach_angle
)
from scipy import interpolate


def min_length_supersonic_nozzle_moc(ht, n, Me=None, A_ratio=None, gamma=1.4):
    """
    Compute the contour of the minimum length supersonic nozzle in a planar
    case with the Method of characteristics.

    The method of characteristics provides a technique for properly designing
    the contour of a supersonic nozzle for shockfree, isentropic flow, taking
    into account the multidimensional flow inside the duct.

    Assumptions:

    * Planar case
    * Sharp corner at the throat,
    * M = 1 and theta = 0 at the throat.

    Parameters
    ----------
    ht : float
        Throat height. Must be > 0.
    n : int
        Number of characteristic lines.
    Me : float
        Mach number at the exit section. Default to None. Must be > 1.
        If set to None, A_ratio must be provided. If both are set,
        Me will be used.
    A_ratio : float
        Ratio between the exit area and the throat area. Since this
        nozzle is planar, it is equivalent to Re/Rt. It must be > 1.
        Default to None. If set to None, Me must be provided.
        If both are set, Me will be used.
    gamma : float
        Specific heats ratio. Default to 1.4. Must be > 1.

    Returns
    -------
    wall : np.ndarray  [2 x n+1]
        Coordinates of points on the nozzle's wall
    characteristics : list
        List of dictionaries. Each dictionary contains the keys "x", "y"
        for the coordinates of the points of each characteristic. Here, with
        characteristic, I mean the points of the right and left running
        characteristic.
    left_runn_chars : list
        List of dictionaries. Each dictionary contains the keys "Km", "Kp",
        "theta", "nu", "M", "mu", "x", "y". Each dictionary represents the
        points lying on the same left-running characteristic.
    theta_w_max : float
        Maximum wall inclination at the sharp corner.

    Examples
    --------

    .. plot::
       :context: reset
       :format: python
       :include-source: True

       from pygasflow.nozzles.moc import min_length_supersonic_nozzle_moc
       import numpy as np
       import matplotlib.pyplot as plt
       import matplotlib.cm as cmx
       ht = 1.5
       n = 20
       Me = 5
       gamma = 1.4
       wall, characteristics, left_runn_chars, theta_w_max = min_length_supersonic_nozzle_moc(ht, n, Me, None, gamma)
       x, y, z = np.array([]), np.array([]), np.array([])
       for char in left_runn_chars:
           x = np.append(x, char["x"])
           y = np.append(y, char["y"])
           z = np.append(z, char["M"])
       plt.figure()
       # draw characteristics lines
       for c in characteristics:
           plt.plot(c["x"], c["y"], "k:", linewidth=0.65)
           # draw nozzle wall
           plt.plot(wall[:, 0], wall[:, 1], "k")
       # over impose grid points colored by Mach number
       sc = plt.scatter(x, y, c=z, s=15, vmin=min(z), vmax=max(z), cmap=cmx.cool)
       cbar = plt.colorbar(sc, orientation='vertical', aspect=40)
       cbar.ax.get_yaxis().labelpad = 15
       cbar.ax.set_ylabel("Mach number", rotation=270)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.title(r"$M_e$ = {}, n = {}, ht = {} ".format(Me, n, ht))
       plt.grid()
       plt.axis('equal')
       plt.tight_layout()
       plt.show()

    """
    if ht <= 0:
        raise ValueError("The throat height must be a number > 0.")
    # TODO: is n > 2 correct?
    if n <= 2:
        raise ValueError("The number of characteristic lines must be an integer > 2.")
    if gamma <= 1:
        raise ValueError("Specific heats ratio must be > 1.")

    if Me:
        if Me <= 1:
            raise ValueError("Exit Mach number must be > 1.")
    elif A_ratio:
        if A_ratio <= 1:
            raise ValueError("Area ratio must be > 1.")
        Me = m_from_critical_area_ratio(A_ratio, "super", gamma)
    else:
        raise ValueError("Either Me or A_ratio must be provided.")

    # Prandtl-Meyer function for the designed exit Mach number
    vme = prandtl_meyer_angle(Me, gamma)

    # max angle of the wall downstream of the throat (equation 11.33)
    theta_w_max = vme / 2

    # read carefoully this:
    # http://www.ltas-aea.ulg.ac.be/cms/uploads/Aerothermodynamics05.pdf
    # especially slide 32
    # TODO: is theta_1 small enough for very high n?
    theta_1 = 5e-02
    # theta_1 = theta_w_max - np.floor(theta_w_max)

    delta_theta = (theta_w_max - theta_1) / (n - 1)

    ###
    ### Generate the grid
    ###

    # collection of left running characteristics.
    # each left running characteristic is composed by a certain number of
    # points, resulting from the intersection of this left running characteristic
    # with right running characteristics...
    left_runn_chars = []

    for i in range(n):
        # number of points on the current left-running characteristic
        npt = n + 1 - i
        # init
        left_runn_chars.append({
            "Km": np.zeros(npt),        # right running characteristic K-
            "Kp": np.zeros(npt),        # left running characteristic K+
            "theta": np.zeros(npt),     # deflection angle
            "nu": np.zeros(npt),        # Prandtl-Meyer angle
            "M": np.zeros(npt),         # Mach number
            "mu": np.zeros(npt),        # Mach angle
            "x": np.zeros(npt),         # x coordinate
            "y": np.zeros(npt),         # y coordinate
        })

        for j in range(npt - 1):
            # note that after the first line, theta_1 = 0
            theta = theta_1 + delta_theta * j

            if i == 0:
                nu = theta
                Km = theta + nu
            else:
                Km = left_runn_chars[i - 1]["Km"][j + 1]
                nu = Km - theta

            M = m_from_prandtl_meyer_angle(nu, gamma)
            left_runn_chars[i]["theta"][j] = theta
            left_runn_chars[i]["nu"][j] = nu
            left_runn_chars[i]["Km"][j] = Km
            left_runn_chars[i]["Kp"][j] = theta - nu
            left_runn_chars[i]["M"][j] = M
            left_runn_chars[i]["mu"][j] = mach_angle(M)

        left_runn_chars[i]["theta"][j + 1] = left_runn_chars[i]["theta"][j]
        left_runn_chars[i]["nu"][j + 1] = left_runn_chars[i]["nu"][j]
        left_runn_chars[i]["Km"][j + 1] = left_runn_chars[i]["Km"][j]
        left_runn_chars[i]["Kp"][j + 1] = left_runn_chars[i]["Kp"][j]
        left_runn_chars[i]["M"][j + 1] = left_runn_chars[i]["M"][j]
        left_runn_chars[i]["mu"][j + 1] = left_runn_chars[i]["mu"][j]

        # after first line, we do not need this value anymore
        theta_1 = 0

    ###
    ### Compute nodes coordinates
    ###

    # For readibility purposes, define tangent for angles in degrees
    def tand(angle):
        return np.tan(np.deg2rad(angle))

    for i, l in enumerate(left_runn_chars):
        # number of points in the left running characteristic
        theta, mu = l["theta"], l["mu"]
        _n = len(theta)
        x = np.zeros(_n)
        y = np.zeros(_n)

        for j in range(_n):
            # the first characteristic is a special case, because at its left
            # there is only the point (0, 1)
            if i == 0:
                # point in the simmetry line
                if j == 0:
                    x[j] = -1 / tand(theta[j] - mu[j])
                    y[j] = 0

                # point at the wall
                elif j == _n - 1:
                    num = y[j-1] - 1 - x[j-1] * tand(0.5 * (theta[j-1] + mu[j-1] + theta[j] + mu[j]))
                    den = tand(0.5 * (theta_w_max + theta[j])) - tand(0.5 * (theta[j-1] + mu[j-1] + theta[j] + mu[j]))
                    x[j] = num / den
                    y[j] = 1 + x[j] * tand(0.5 * (theta_w_max + theta[j]))

                # points in the flow region
                else:
                    num = (1 - y[j-1] + x[j-1] * tand(0.5 * (theta[j-1] + mu[j-1] + theta[j] + mu[j])))
                    den = tand(0.5 * (theta[j-1] + mu[j-1] + theta[j] + mu[j])) - tand(theta[j] - mu[j])
                    x[j] = num / den
                    y[j] = tand(theta[j] - mu[j]) * x[j] + 1

            # all other left characteristics
            else:
                # previous left running characteristic line
                lprev = left_runn_chars[i - 1]
                # values of the point in the previous left running characteristic line
                x_prev = lprev["x"][j+1]
                y_prev = lprev["y"][j+1]
                theta_prev = lprev["theta"][j+1]
                mu_prev = lprev["mu"][j+1]

                # points in the simmetry line
                if j == 0:
                    x[j] = x_prev - y_prev / (tand(0.5 * (theta[j] + theta_prev - mu[j] - mu_prev)))
                    y[j] = 0

                # point at the wall
                elif j == _n - 1:
                    num = x_prev * tand(0.5 * (theta_prev + theta[j])) \
                        - y_prev + l["y"][j-1] - l["x"][j-1] * tand(0.5 * (theta[j] + theta[j-1] + mu[j] + mu[j-1]))

                    den = tand(0.5 * (theta[j] + theta_prev)) \
                        - tand(0.5 * (theta[j-1] + theta[j] + mu[j-1] + mu[j]))

                    x[j] = num / den
                    y[j] = y_prev + (l["x"][j] - x_prev) * tand(0.5 * (theta_prev + theta[j]))

                # points in the flow region
                else:
                    s1 = tand(0.5 * (theta[j] + theta[j-1] + mu[j] + mu[j-1]))
                    s2 = tand(0.5 * (theta[j] + theta_prev - mu[j] - mu_prev))
                    x[j] = (y_prev - l["y"][j-1] + s1 * l["x"][j-1] - s2 * x_prev) / (s1 - s2)
                    y[j] = l["y"][j-1] + (l["x"][j] - l["x"][j-1]) * s1

            # add the computed coordinates points to the left running characteristic
            left_runn_chars[i]["x"] = x
            left_runn_chars[i]["y"] = y

    for l in left_runn_chars:
        l["x"] *= ht
        l["y"] *= ht

    # each symmetry line point has a left and right-running characteristic.
    # I pack them togheter for visualization purposes.
    characteristics = []
    # extract the wall coordinates
    wall = np.zeros((n + 1, 2))
    # first coordinate is the sharp corner
    wall[0, :] = [0, 1 * ht]

    for i, l in enumerate(left_runn_chars):
        # each characteristic starts from the sharp corner
        x = np.zeros(len(l["x"]) + i + 1)
        y = np.zeros(len(l["x"]) + i + 1)
        x[0] = 0
        y[0] = 1 * ht

        # add the points of the current right-running characteristic. These points
        # are included in the previous left-running characteristics.
        for j in range(i):
            x[j + 1] = left_runn_chars[j]["x"][i - j]
            y[j + 1] = left_runn_chars[j]["y"][i - j]

        # add the point of the current left-running characteristic
        if i == 0:
            j = -1
        x[j+2:] = left_runn_chars[i]["x"]
        y[j+2:] = left_runn_chars[i]["y"]

        characteristics.append({
            "x": x,
            "y": y
        })

        wall[i + 1, :] = [l["x"][-1], l["y"][-1]]

    return wall, characteristics, left_runn_chars, theta_w_max


class CD_Min_Length_Nozzle(Nozzle_Geometry):
    """
    Planar Convergent-Divergent nozzle based on Minimum Length computed
    with the Method of Characteristics.

    Examples
    --------

    Compute the length of a minimum length nozzle:

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

        from pygasflow.nozzles import CD_Min_Length_Nozzle
        Ri, Re, Rt = 0.4, 0.5, 0.2
        nozzle = CD_Min_Length_Nozzle(Ri, Re, Rt, theta_c=30, n_lines=20)
        nozzle.plot(interactive=False, show_characteristic_lines=True)

    """

    _params_to_document = [
        "inlet_radius", "outlet_radius", "throat_radius",
        "theta_c", "gamma", "n_lines", "N",
        "inlet_area", "outlet_area", "throat_area",
        "length_convergent", "length_divergent", "length",
        "shockwave_location", "theta_w_max", "wall",
        "characteristics", "left_runn_chars"
    ]

    theta_w_max = param.Number(constant=True, doc="Maximum angle of the wall.")

    wall = param.Array(constant=True, doc="""
        Mx2 array of wall coordinates computed by MOC. First column is
        the x-coordinates, second column is the y-coordinates.""")

    characteristics = param.List(item_type=dict, constant=True, doc="""
        List of dictionaries. Each dictionary contains the keys "x", "y"
        for the coordinates of the points of each characteristic. Here, with
        characteristic, I mean the points of the right and left running
        characteristic.""")

    left_runn_chars = param.List(item_type=dict, constant=True, doc="""
        List of dictionaries. Each dictionary contains the keys "Km", "Kp",
        "theta", "nu", "M", "mu", "x", "y". Each dictionary represents the
        points lying on the same left-running characteristic.""")

    def __init__(
        self, Ri=0.4, Re=1.2, Rt=0.2, Rj=0.1, R0=0, theta_c=40,
        n_lines=10, gamma=1.4, **params
    ):
        params.setdefault("title", "MOC Nozzle")
        params["geometry_type"] = "planar"
        super().__init__(
            Ri, Re, Rt, theta_c=theta_c, n_lines=n_lines,
            gamma=gamma, R0=R0, **params
        )

    @param.depends(
        "inlet_radius", "outlet_radius", "throat_radius",
        "geometry_type", "theta_c", "n_lines", "gamma",
        "N", "junction_radius_0",
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
        s = "C-D Minimum Length Nozzle\n"
        s += super().__str__()
        s += f"Number of characteristic lines: {self.n_lines}\n"
        s += "Angles:\n"
        s += "\ttheta_c\t{}\n".format(self.theta_c)
        s += "\ttheta__w_max\t{}\n".format(self.theta_w_max)
        return s

    def _compute_intersection_points(self):
        Ri, Rt, Re = self.inlet_radius, self.throat_radius, self.outlet_radius
        R0 = self.junction_radius_0
        Rj = self.junction_radius_j
        n = self.n_lines
        gamma = self.gamma

        # find interesting points for the convergent
        x0, y0, x1, y1, xc, yc = convergent(self.theta_c, Ri, R0, Rt, Rj / Rt)
        self.length_convergent = xc
        # offset to the left, I want x=0 to be throat section
        x0 -= xc
        x1 -= xc

        # compute the wall points
        wall, characteristics, left_runn_chars, theta_w_max = min_length_supersonic_nozzle_moc(
            Rt, n, None, self.outlet_area / self.throat_area, gamma)
        self.length_divergent = wall[-1, 0]
        self.length = self.length_convergent + self.length_divergent

        with param.edit_constant(self):
            self.param.update(dict(
                theta_w_max=theta_w_max,
                wall=wall,
                characteristics=characteristics,
                left_runn_chars=left_runn_chars
            ))

        self._intersection_points = {
            "S": [-self.length_convergent, Ri],   # start point
            "0": [x0, y0],  # combustion chamber circle - convergent straight line
            "1": [x1, y1],  # convergent straight line - throat circle
            "orig": [0, Rt],  # throat origin
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
        N = self.N

        Lc = self.length_convergent
        Ld = self.length_divergent

        x0, y0 = self._intersection_points["0"]
        x1, y1 = self._intersection_points["1"]

        theta_c = np.deg2rad(self.theta_c)
        # intercept of the straight line to the throat section
        q = y1 + x1 * np.tan(theta_c)

        # Compute the points
        x = np.linspace(-Lc, Ld, N)
        y = np.zeros_like(x)
        # junction between combustion chamber and convergent
        y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
        # straight line of the convergent
        idx = np.bitwise_and(x > x0, x < x1)
        y[idx] = -np.tan(theta_c) * x[idx] + q
        # junction between convergent and divergent
        idx = np.bitwise_and(x >= x1, x <= 0)
        y[idx] = -np.sqrt(Rj**2 - x[idx]**2) + (Rt + Rj)
        # straight line of the divergent
        # interpolation of the divergent's wall points
        # TODO: try to use UnivariateSpline and look for a decent smoothing
        s = interpolate.InterpolatedUnivariateSpline(
            self.wall[:, 0], self.wall[:, 1])
        y[x > 0] = s(x[x > 0])
        # I absolutely need the origin to be a sharp corner!
        if 0 in x:
            y[0] = Rt
        else:
            idx = np.where(x < 0)[0]
            y[idx[-1]] = Rt
            x[idx[-1]] = 0

        return x, y

    def _get_params_for_ui(self):
        return [
            self.param.throat_radius,
            self.param.outlet_radius,
            self.param.inlet_radius,
            self.param.junction_radius_0,
            self.param.theta_c,
            self.param.n_lines,
            self.param.gamma,
        ]
