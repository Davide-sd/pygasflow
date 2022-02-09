import numpy as np
from scipy import interpolate

class Nozzle_Geometry:
    """
    Represents a nozzle geometry.
    This is meant as a base class.

    TODO: can I make it abstract?
    """

    def __init__(self, Ri, Re, Rt, Lc, Ld, geometry_type="planar"):
        """
        Parameters
        ----------
        Ri : float
            Inlet radius.
        Re : float
            Exit (outlet) radius.
        Rt : float
            Throat radius.
        Rj : float
            Radius of the junction between convergent and divergent.
        R0 : float
            Radius of the junction between combustion chamber and convergent.
        Lc : float
            Length of the convergent.
        Ld : float
            Length of the divergent. Default to None. If None, theta_N will
            be used to compute the divergent's length.
        geometry_type : string
            Specify the geometry type of the nozzle. Can be either
            ``'axisymmetric'`` or ``'planar'``
        """
        geometry_type = geometry_type.lower()
        if geometry_type not in ['axisymmetric', 'planar']:
            raise ValueError("Geometry type can be 'axisymmetric' or 'planar'.")
        self._geometry_type = geometry_type

        self._Ri = Ri
        self._Re = Re
        self._Rt = Rt

        # compute areas
        # for planar case, the radius corresponds to half the height of
        # the nozzle. Width remain constant along nozzle's length, therefore
        # it simplifies, hence it is not considered in the planar case.
        self._Ai = 2 * Ri
        self._Ae = 2 * Re
        self._At = 2 * Rt
        if geometry_type == "axisymmetric":
            self._Ai = np.pi * Ri**2
            self._Ae = np.pi * Re**2
            self._At = np.pi * Rt**2

        self._Lc = Lc
        if Lc == None: self._Lc = 0
        self._Ld = Ld
        if Ld == None: self._Ld = 0

        self._length_array = None
        self._area_ratio_array = None
        self._wall_radius_array = None

    @property
    def inlet_radius(self):
        return self._Ri

    @property
    def outlet_radius(self):
        return self._Re

    @property
    def critical_radius(self):
        return self._Rt

    @property
    def inlet_area(self):
        return self._Ai

    @property
    def outlet_area(self):
        return self._Ae

    @property
    def critical_area(self):
        return self._At

    @property
    def length_convergent(self):
        return self._Lc

    @property
    def length_divergent(self):
        return self._Ld

    @property
    def length(self):
        return self._Lc + self._Ld

    @property
    def length_array(self):
        return self._length_array

    @property
    def wall_radius_array(self):
        return self._wall_radius_array

    @property
    def area_ratio_array(self):
        return self._area_ratio_array

    def __str__(self):
        s = ""
        s += "Radius:\n"
        s += "\tRi\t{}\n".format(self._Ri)
        s += "\tRe\t{}\n".format(self._Re)
        s += "\tRt\t{}\n".format(self._Rt)
        s += "Areas:\n"
        s += "\tAi\t{}\n".format(self._Ai)
        s += "\tAe\t{}\n".format(self._Ae)
        s += "\tAt\t{}\n".format(self._At)
        s += "Lengths:\n"
        s += "\tLc\t{}\n".format(self.length_convergent)
        s += "\tLd\t{}\n".format(self.length_divergent)
        s += "\tL\t{}\n".format(self.length)
        return s

    def get_points(self, area_ratio=False, offset=1.2):
        """
        Helper function used to construct a matrix of points representing
        the nozzle for visualization purposes.

        Parameters
        ----------
        area_ratio : Boolean
            If True, represents the area ratio A/A*. Otherwise, represents
            the radius. Default to False.
        offset : float
            Used to construct the container (or the walls) of the nozzle.
            The container radius is equal to offset * max_nozzle_radius.

        Returns
        -------
            points_top : np.ndarray [Nx2]
                Matrix representing the wall at the top of the nozzle.
            points_mid : np.ndarray [Nx2]
                Matrix representing the flow area.
            points_bottom : np.ndarray [Nx2]
                Matrix representing the wall at the bottom of the nozzle.
        """
        L = np.copy(self._length_array)
        r = np.copy(self._area_ratio_array)
        if not area_ratio:
            r = np.sqrt(r * self._At / np.pi)

        max_r = np.max(r)
        container = np.ones_like(L) * offset * max_r
        container = np.concatenate((container, -container[::-1]))
        L = np.concatenate((L, L[::-1]))
        container = np.vstack((L, container)).T
        nozzle = np.concatenate((r, -r[::-1]))
        nozzle = np.vstack((L, nozzle)).T

        return nozzle, container

    def location_divergent_from_area_ratio(self, A_ratio):
        """
        Given an area ratio, compute the location on the divergent where
        this area ratio is located.

        Parameters
        ----------
        A_ratio : float
            A / At

        Returns
        -------
        x : float
            x-coordinate along the divergent length.
        """
        A = A_ratio * self._At
        if A <= 0:
            raise ValueError("The area ratio must be > 0.")
        if A > self._Ae:
            raise ValueError("The provided area ratio is located beyond the exit section.")
        if A < self._At:
            raise ValueError("The provided area ratio is located in the convergent.")

        # compute the location of area ratio
        # https://stackoverflow.com/questions/1029207/interpolation-in-scipy-finding-x-that-produces-y
        # x = self.Length_Array
        length = self._length_array
        area_ratios = self._area_ratio_array
        x = length[length >= 0]
        y = area_ratios[length >= 0]
        y_reduced = y - A_ratio
        f_reduced = interpolate.InterpolatedUnivariateSpline(x, y_reduced)
        # having limited myself to a monotonic curve, I only have one root
        return f_reduced.roots()
