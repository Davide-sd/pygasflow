import numpy as np
import param
from scipy import interpolate
import warnings


class Nozzle_Geometry(param.Parameterized):
    """
    Represents a nozzle geometry. This is meant as a base class.
    """

    # editable parameters
    # I place all editable parameters in this base class in order to be
    # able to access them from De_Laval_Solver, when using param.depends(...),
    # or from NozzleDiagram, when updating the visualization.
    # Not all nozzle geometries might use all these parameters.
    inlet_radius = param.Number(
        default=0.4, bounds=(0, None), softbounds=(0.01, 3),
        step=0.01, label="Ri [m]", doc="Alias ``Ri`` in the constructor.")
    outlet_radius = param.Number(
        default=1.2, bounds=(0, None), softbounds=(0.01, 3),
        step=0.01, label="Re [m]", doc="Alias ``Re`` in the constructor.")
    throat_radius = param.Number(
        default=0.2, bounds=(0, None), softbounds=(0.01, 3),
        step=0.01, label="Rt [m]", doc="Alias ``Rt`` in the constructor.")
    junction_radius_j = param.Number(
        default=0.1, bounds=(0, None), softbounds=(0, 0.5), step=0.01,
        label="Rj [m]",
        doc="""
            Radius of the junction between convergent and divergent.
            Alias ``Rj`` in the constructor.""")
    junction_radius_0 = param.Number(
        default=0, bounds=(0, None), softbounds=(0, 0.5), step=0.01,
        label="R0 [m]",
        doc="""
            Radius of the junction between combustion chamber and convergent.
            Alias ``R0`` in the constructor.""")
    theta_c = param.Number(
        default=40, bounds=(0, 90), softbounds=(0.001, 90),
        inclusive_bounds=(False, False),
        label="Half angle [degrees] of the convergent",
        doc="Half angle [degrees] of the convergent")
    theta_N = param.Number(
        default=15, bounds=(0, 90), softbounds=(0.001, 90),
        inclusive_bounds=(False, False),
        label="Half angle [degrees] of the conical divergent",
        doc="Half angle [degrees] of the conical divergent")
    theta_e = param.Number(
        default=15,
        constant=True,
        label="Half angle [degrees] of the conical divergent at the exit section.",
        doc="Half angle [degrees] of the conical divergent at the exit section.")
    fractional_length = param.Number(
        default=0.8, bounds=(0.6, 1), step=0.01,
        label="Fractional Length, K", doc="""
            Fractional Length of the nozzle with respect to a same exit
            area ratio conical nozzle with 15 deg half-cone angle.
            Alias ``K`` in the constructor.""")
    geometry_type = param.Selector(
        objects=["axisymmetric", "planar"],
        default="axisymmetric",
        label="Geometry:",
        doc="""
            Specify the geometry type of the nozzle:

            * ``"planar"``: the radius indicates the distance from the line
              of symmetry and the nozzle wall. The area is computed with
              ``A = 2 * r``. Note the lack of depth in the formula: this is
              because it simplifies.
            * ``"axisymmetric"``: the area is computed with
              ``A = pi * r**2``."""
    )
    N = param.Integer(
        default=200, bounds=(10, 1000),
        label="Number of points:",
        doc="Number of discretization elements along the length of the nozzle.")
    error_log = param.String("", doc="""
        Visualize on the interactive application any error that raises
        from the computation.""")
    is_interactive_app = param.Boolean(False, doc="""
        If True, exceptions are going to be intercepted and shown on
        error_log, otherwise fall back to the standard behaviour.""")
    title = param.String()

    # for MOC nozzles
    gamma = param.Number(1.4, bounds=(1, 2),
        inclusive_bounds=(False, True),
        step=0.05,
        label="Ratio of specific heats, γ",
        doc="Ratio of specific heats, γ = Cp / Cv"
    )
    n_lines = param.Integer(default=10, bounds=(3, None),
        label="Number of characteristic lines:")

    # read-only parameters
    inlet_area = param.Number(bounds=(0, None), constant=True)
    outlet_area = param.Number(bounds=(0, None), constant=True)
    throat_area = param.Number(bounds=(0, None), constant=True)
    length_convergent = param.Number(allow_None=True)
    length_divergent = param.Number(allow_None=True)
    length = param.Number(bounds=(0, None),
        doc="Total length of the nozzle, convergent + divergent.")
    length_array = param.Array(constant=True)
    wall_radius_array = param.Array(constant=True)
    area_ratio_array = param.Array(constant=True)
    shockwave_location = param.Tuple(
        default=(None, None),
        constant=True,
        doc="Location of the shockwave in the divergent: (loc, radius).")

    def __init__(self, Ri, Re, Rt, R0=0.1, Rj=0.1, **params):
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
        Rj : float
            Radius of the junction between convergent and divergent.
        """
        if (Ri <= Rt) or (Re <= Rt):
            raise ValueError("Must be Ai > At and Ae > At.")
        # aliases for most important parameters
        params.update({
            "inlet_radius": Ri,
            "outlet_radius": Re,
            "throat_radius": Rt,
            "junction_radius_j": Rj,
            "junction_radius_0": R0,
        })
        super().__init__(**params)

    def _compute_area(self, radius):
        # for planar case, the radius corresponds to half the height of
        # the nozzle. Width remain constant along nozzle's length, therefore
        # it simplifies, hence it is not considered in the planar case.
        if self.geometry_type == "axisymmetric":
            return np.pi * radius**2
        return 2 * radius

    def _compute_area_ratio(self, radius):
        area = self._compute_area(radius)
        return area / self.throat_area

    def _compute_radius(self, area):
        if self.geometry_type == "axisymmetric":
            return np.sqrt(area / np.pi)
        return area / 2

    @param.depends(
        "inlet_radius", "outlet_radius", "throat_radius", "geometry_type",
        watch=True, on_init=True
    )
    def _update_areas(self):
        with param.edit_constant(self):
            self.param.update({
                "inlet_area": self._compute_area(self.inlet_radius),
                "outlet_area": self._compute_area(self.outlet_radius),
                "throat_area": self._compute_area(self.throat_radius)
            })

    @param.depends(
        "inlet_radius", "outlet_radius", "throat_radius",
        "geometry_type", "theta_N", "theta_c", "theta_e",
        "N", "fractional_length", "junction_radius_0", "junction_radius_j",
        watch=True, on_init=True
    )
    def _update_geometry(self):
        try:
            # compute the intersection points of the different curves
            # composing the nozzle.
            self._compute_intersection_points()
            x, y = self.build_geometry()
            with param.edit_constant(self):
                self.param.update(dict(
                    length_array=x,
                    wall_radius_array=y,
                    area_ratio_array=self._compute_area_ratio(y)
                ))
            self.error_log = ""
        except ValueError as err:
            self.error_log = f"ValueError: {err}"
            if not self.is_interactive_app:
                raise ValueError(f"{err}")

    def __str__(self):
        s = ""
        s += "Radius:\n"
        s += "\tRi\t{}\n".format(self.inlet_radius)
        s += "\tRe\t{}\n".format(self.outlet_radius)
        s += "\tRt\t{}\n".format(self.throat_radius)
        s += "Areas:\n"
        s += "\tAi\t{}\n".format(self.inlet_area)
        s += "\tAe\t{}\n".format(self.outlet_area)
        s += "\tAt\t{}\n".format(self.throat_area)
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
        flow_area : np.ndarray [Nx2]
            Matrix representing the flow area.
        container : np.ndarray [Nx2]
            Matrix representing the outer walls of the nozzle.
        """
        L = np.copy(self.length_array)
        r = np.copy(self.area_ratio_array)
        if not area_ratio:
            r *= self.throat_area
            if self.geometry_type == "planar":
                r = r / 2
            else:
                r = np.sqrt(r / np.pi)

        max_r = np.max(r)
        container = np.array([
            [L.min(), -offset * max_r],
            [L.max(), -offset * max_r],
            [L.max(), offset * max_r],
            [L.min(), offset * max_r],
        ])
        L = np.concatenate((L, L[::-1]))
        flow_area = np.concatenate((r, -r[::-1]))
        flow_area = np.vstack((L, flow_area)).T

        return flow_area, container

    def plot(self, interactive=True, **kwargs):
        """
        Parameters
        ----------
        interactive : bool
            If False, shows and return a Bokeh figure. If True, returns a
            servable object, which will be automatically rendered inside a
            Jupyter Notebook. If any other interpreter is used, then
            ``nozzle.plot(interactive=True).show()`` might be required in
            order to visualize the interactive application.
        """
        from bokeh.plotting import show as bokeh_show
        from pygasflow.interactive.diagrams.de_laval import NozzleDiagram
        d = NozzleDiagram(nozzle=self, **kwargs)
        if interactive:
            return d.servable()
        bokeh_show(d.figure)
        return d.figure

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
        x : float or None
            x-coordinate along the divergent length. If A_ratio > Ae / At,
            it returns None.
        """
        def deal_with_error(msg, warns=False):
            # NOTE: I'm not going to add it in the error log because it
            # shouldn't be relevant when using DeLavalSection
            # self.error_log += f"\nValueError: {msg}"
            if not self.is_interactive_app:
                if warns:
                    warnings.warn(msg, stacklevel=2)
                else:
                    raise ValueError(msg)

        A = A_ratio * self.throat_area
        if A <= 0:
            with param.edit_constant(self):
                self.shockwave_location = (None, None)
            deal_with_error("The area ratio must be > 0.")
            return
        if A > self.outlet_area:
            with param.edit_constant(self):
                self.shockwave_location = (None, None)
            deal_with_error(
                "The provided area ratio is located beyond the exit section.",
                warns=True
            )
            return
        if A < self.throat_area:
            with param.edit_constant(self):
                self.shockwave_location = (None, None)
            deal_with_error(
                "The provided area ratio is located in the convergent.")
            return

        # compute the location of area ratio
        # https://stackoverflow.com/questions/1029207/interpolation-in-scipy-finding-x-that-produces-y
        length = self.length_array
        area_ratios = self.area_ratio_array
        x = length[length >= 0]
        y = area_ratios[length >= 0]
        y_reduced = y - A_ratio
        f_reduced = interpolate.InterpolatedUnivariateSpline(x, y_reduced)
        # having limited myself to a monotonic curve, I only have one root
        location = f_reduced.roots()[0]
        with param.edit_constant(self):
            self.shockwave_location = (location, self._compute_radius(A))
        return location

