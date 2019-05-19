import numpy as np
from pygasflow.utils.geometry import (
    Tangent_Points_On_Circle, 
    Internal_Tangents_To_Two_Circles
)

def Convergent(Lc, Ri, R0, Rt, factor):
    """
    Simple function to create a smooth curved convergent nozzle. It is composed of:
    * Circular segment at the junction with the combustion chamber (from -Lc <= x <= x0).
    * Straight segment tangent to both circles (from x0 <= x <= x1).
    * Circular segment at the outlet of the nozzle (from x1 <= x <= x0).

    Parameters
    ----------
        Lc : float
            Convergent Length.
        Ri : float
            Inlet section radius.
        R0 : float
            Radius of the junction between combustion chamber and convergent.
        Rt : float
            Throat area.
        factor : float
            Ratio between the radius of the circle on the outlet of the convergent with
            the throat radius.
    
    Returns
    -------
        x0 : float
            x-coordinate of the end of the circular junction of radius R0.
        y0 : float
            y-coordinate of the end of the circular junction of radius R0.
        theta_0 : float 
            slope angle of the straight line in the convergent.
        q0 : float 
            intercept at x=0 of the straight line in the convergent.
        x1 : float 
            x-coordinate of the start of the circular junction of radius (factor * Rt).
        y1 : float
            y-coordinate of the start of the circular junction of radius (factor * Rt).
    """
    # center of the throat circle coming from the convergent
    c1 = np.asarray([0, (1 + factor) * Rt])
    # center of the circle between the junction of the convergent and the combustion chamber
    c0 = np.asarray([-Lc, Ri - R0])

    assert np.linalg.norm(c1 - c0) >= (factor * Rt + R0), "Impossible to build the convergent. Either R0 is too big or Lc is too small."
    
    # compute the two internal tangent lines for circles at c1, c0
    tangent_1, tangent_2 = Internal_Tangents_To_Two_Circles(c1, c0, factor * Rt, R0)

    def find_x0_x1(T1, T2):
        """
        Helper function. Look at the points of each internal tangent and find the coordinates
        x0 and x1.
        """
        # a: tangent point on circle of radius R0 (initial choice)
        a = T1[0, :]
        # b: tangent point on circle of radius factor*Rt
        b = T1[1, :]
        if np.abs(np.linalg.norm(c0 - a) - R0) > 1e-08:
            a = T1[1, :]
            b = T1[0, :]
        # if the y coordinate of the tangent point is greater than the y-coord of the circle center
        if a[1] > Ri - R0:
            return a[0], b[0]
        
        # if we arrive here, the correct tangent point of circle 0 is found on T2
        a = T2[0, :]
        b = T2[1, :]
        if np.abs(np.linalg.norm(c0 - a) - R0) > 1e-08:
            a = T2[1, :]
            b = T2[0, :]
        # if the y coordinate of the tangent point is greater than the y-coord of the circle center
        if a[1] > Ri - R0:
            return a[0], b[0]
    
    x0, x1 = find_x0_x1(tangent_1, tangent_2)

    # TODO: Do I really need this assert???
    assert np.abs(x0) >= np.abs(x1), "Convergent length too short."

    # compute respective y-coordinates
    y0 = np.sqrt(R0**2 - (x0 + Lc)**2) + (Ri - R0)
    y1 = -np.sqrt((factor * Rt)**2 - x1**2) + (1 + factor) * Rt
    # compute the intercept for the straight line of the convergent
    theta_0 = np.arctan((y0 - y1) / (x0 - x1))
    q0 = y0 - np.tan(theta_0) * x0

    return x0, y0, theta_0, q0, x1, y1

class Nozzle_Geometry(object):
    """
    Represents a nozzle geometry.
    """

    def __init__(self, Ai, Ae, At, Lc, Ld):
        """
        Parameters
        ----------
            Ai : float
                Inlet area.
            Ae : float
                Exit (outlet) area.
            At : float
                Throat area.
            Rj : float
                Radius of the junction between convergent and divergent.
            R0 : float
                Radius of the junction between combustion chamber and convergent.
            Lc : float
                Length of the convergent.
            Ld : float
                Length of the divergent. Default to None. If None, theta_N will be used to compute the divergent's length.
            
        """
        self._Ai = Ai
        self._Ae = Ae
        self._At = At

        self._Lc = Lc
        if Lc == None: self._Lc = 0
        self._Ld = Ld
        if Ld == None: self._Ld = 0

        self._length_array, self._area_ratio_array = None, None

    @property
    def Inlet_area(self):
        return self._Ai

    @property
    def Outlet_area(self):
        return self._Ae
    
    @property
    def Critical_area(self):
        return self._At
    
    @property
    def Length_Convergent(self):
        return self._Lc
    
    @property
    def Length_Divergent(self):
        return self._Ld
    
    @property
    def Length(self):
        return self._Lc + self._Ld
    
    @property
    def Length_array(self):
        return self._length_array
    
    @property
    def Area_ratio_array(self):
        return self._area_ratio_array
    
    def __str__(self):
        s = ""
        s += "Ai\t{}\n".format(self._Ai)
        s += "Ae\t{}\n".format(self._Ae)
        s += "At\t{}\n".format(self._At)
        s += "Lc\t{}\n".format(self.Length_Convergent)
        s += "Ld\t{}\n".format(self.Length_Divergent)
        s += "L\t{}\n".format(self.Length)
        return s
    
    def Get_Points(self, area_ratio=False, offset=1.2):
        """ 
        Helper function used to construct a matrix of points representing the nozzle
        for visualization purposes.

        Parameters
        ----------
            area_ratio : Boolean
                If True, represents the area ratio A/A*. Otherwise, represents the radius.
                Default to False.
            offset : float
                Used to construct the container (or the walls) of the nozzle. 
                The container radius is equal to offset * max_nozzle_radius.
        
        Returns
        -------
            points_top : [Nx2]
                Matrix representing the wall at the top of the nozzle.
            points_mid : [Nx2]
                Matrix representing the flow area.
            points_bottom : [Nx2]
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

class CD_Conical_Nozzle(Nozzle_Geometry):
    """
    Convergent-Divergent nozzle with conical divergent.
    """

    def __init__(self, Ai, Ae, At, Rj, R0, Lc, Ld=None, theta_N=40, N=100):
        """
        Parameters
        ----------
            Ai : float
                Inlet area.
            Ae : float
                Exit (outlet) area.
            At : float
                Throat area.
            Rj : float
                Radius of the junction between convergent and divergent.
            R0 : float
                Radius of the junction between combustion chamber and convergent.
            Lc : float
                Length of the convergent.
            Ld : float
                Length of the divergent. Default to None. If None, theta_N will be used to compute the divergent's length.
            theta_N : float
                Half angle [degrees] of the conical divergent. Default to 40 deg. If None, it will be computed from Ld. If both are set, theta_N will be recomputed from Ld.
            N : int
                Number of discretization elements along the length of the nozzle. Default to 100.
        """

        assert Ai > At and Ae > At, "Must be Ai > At and Ae > At."
        assert Ld or theta_N, "Either Ld or theta_N must be provided."
        assert isinstance(N, (int)) and N > 1, "The number of elements for discretization must be N > 1."

        assert Rj > 0, "Junction radius between Convergent and Divergent must be > 0."
        assert R0 > 0, "Junction radius between Combustion Chamber and Convergent must be > 0."

        Rt = np.sqrt(At / np.pi)
        Re = np.sqrt(Ae / np.pi)
        factor = Rj / Rt

        if theta_N and not Ld:
            assert theta_N > 0 and theta_N < 90, "The half cone angle must be 0 < theta_N < 90."
            xN = (factor * Rt) * np.sin(np.deg2rad(theta_N))
            RN = -np.sqrt((factor * Rt)**2 - xN**2) + (1 + factor) * Rt
            xe = xN + (Re - RN) / np.tan(np.deg2rad(theta_N))
            Ld = xe
        # elif not theta_N:
        else:
            c = np.asarray([0, Rt + Rj])
            p = np.asarray([Ld, Re])
            _, q2 = Tangent_Points_On_Circle(c, Rj, p)
            # use the left tangency point as seen from p
            xN, RN = q2
            assert xN <= Ld, "Either the junction radius is too big or Ld is too small. Impossible to continue."
            theta_N = np.rad2deg(np.arctan((Re - RN) / (Ld - xN)))
        
        super().__init__(Ai, Ae, At, Lc, Ld)
        self._theta_N = theta_N
        self._R0 = R0
        self._factor = factor

        x, y = self.Build_Geometry(N)
        self._length_array = x
        self._area_ratio_array = np.pi * y**2 / At
    
    def __str__(self):
        s = "C-D Conical Nozzle\n"
        s += super().__str__()
        s += "theta_N\t{}\n".format(self._theta_N)
        return s
    
    def Build_Geometry(self, N):
        """
        Parameters
        ----------
            N : int
                Number of discretization elements along the length of the nozzle. Default to 100.
        """

        Ri = np.sqrt(self._Ai / np.pi)
        Re = np.sqrt(self._Ae / np.pi)
        Rt = np.sqrt(self._At / np.pi)
        # A_ratio = self._Ae / self._At
        theta_N = np.deg2rad(self._theta_N)
        Lc = self.Length_Convergent
        Ld = self.Length_Divergent
        factor = self._factor
        R0 = self._R0

        # find interesting points for the convergent
        x0, y0, theta_0, q0, x1, y1 = Convergent(Lc, Ri, R0, Rt, factor)

        # point of tangency between the smaller throat junction and the straight line
        xN = (factor * Rt) * np.sin(theta_N)
        RN = -np.sqrt((factor * Rt)**2 - xN**2) + (1 + factor) * Rt
        # compute the intercept for the straight line of the divergent
        qN = RN - np.tan(theta_N) * xN
        
        # Compute the points
        x = np.linspace(-Lc, Ld, N)
        y = np.zeros_like(x)
        # junction between combustion chamber and convergent
        y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
        # straight line in the convergent
        y[np.bitwise_and(x > x0, x < x1)] = np.tan(theta_0) * x[np.bitwise_and(x > x0, x < x1)] + q0
        # junction between convergent and divergent
        y[np.bitwise_and(x >= x1, x <= xN)] = -np.sqrt((factor * Rt)**2 - x[np.bitwise_and(x >= x1, x <= xN)]**2) + (1 + factor) * Rt
        # straight line of the divergent
        y[x > xN] = np.tan(theta_N) * x[x > xN] + qN

        return x, y
    
class CD_TOP_Nozzle(Nozzle_Geometry):
    """
    Convergent-Divergent nozzle based on Rao's Thrust Optimized Parabolic contours
    """

    def __init__(self, Ai, Ae, At, R0, Lc, theta_N=40, theta_e=20, N=100):
        """
        Parameters
        ----------
            Ai : float
                Inlet area.
            Ae : float
                Exit (outlet) area.
            At : float
                Throat area.
            R0 : float
                Radius of the junction between combustion chamber and convergent.
            Lc : float
                Length of the convergent.
            theta_N : float
                Angle [degrees] of the inflection point between the circular curve and the parabola on the divergent. Default to 40 deg.
            theta_e : float
                Angle [degrees] of the inflection point at the exit area. Default to 20 deg.
            N : int
                Number of discretization elements along the length of the nozzle. Default to 100.
        """

        assert Ai > 0 and Ae > 0 and At > 0, "All input areas must be > 0."
        assert Ai > At and Ae > At, "Must be Ai > At and Ae > At."
        assert isinstance(N, (int)) and N > 1, "The number of elements for discretization must be N > 1."
        assert theta_N > 0 and theta_N < 90, "The half cone angle must be 0 < theta_N < 90."
        assert theta_e > 0 and theta_e < theta_N, "The angle at the exit of the parabola must be 0 < theta_e < theta_N."
        assert R0 > 0, "Junction radius between Combustion Chamber and Convergent must be > 0."

        super().__init__(Ai, Ae, At, Lc, None)
        self._theta_N = theta_N
        self._theta_e = theta_e
        self._R0 = R0

        x, y = self.Build_Geometry(N)
        self._length_array = x
        self._area_ratio_array = np.pi * y**2 / At
    
    def __str__(self):
        s = "C-D TOP Nozzle\n"
        s += super().__str__()
        s += "theta_N\t{}\n".format(self._theta_N)
        s += "theta_e\t{}\n".format(self._theta_e)
        return s
    
    def Build_Geometry(self, N):
        """
        Parameters
        ----------
            N : int
                Number of discretization elements along the length of the nozzle. Default to 100.
        """

        Ri = np.sqrt(self._Ai / np.pi)
        Re = np.sqrt(self._Ae / np.pi)
        Rt = np.sqrt(self._At / np.pi)
        A_ratio = self._Ae / self._At
        theta_N = np.deg2rad(self._theta_N)
        theta_e = np.deg2rad(self._theta_e)
        Lc = self.Length_Convergent
        R0 = self._R0
        
        # Rao used a radius of 1.5 * Rt for curve 1 (at the left of the throat)
        factor = 1.5
        # find interesting points for the convergent
        x0, y0, theta_0, q0, x1, y1 = Convergent(Lc, Ri, R0, Rt, factor)

        # point of tangency between the smaller throat junction and the parabola
        xN = 0.382 * Rt * np.sin(theta_N)
        RN = -np.sqrt((0.382 * Rt)**2 - xN**2) + 1.382 * Rt
        
        # percent length of same AR conical nozzle. Used in Rao nozzle designs.
        K = 0.8
        # length of the divergent
        Ln = K *(np.sqrt(A_ratio) - 1) * Rt / np.tan(theta_e)
        self._Ld = Ln

        # solve Rao system of equations to get the parabola parameters
        # and tangency at point N with curve 2
        A = np.asarray([
            [2 * RN, 1, 0],
            [RN**2, RN, 1],
            [Re**2, Re, 1]
        ])
        b = np.asarray([1 / np.tan(theta_N), xN, Ln])
        x = np.linalg.inv(A).dot(b)
        # parabola parameters
        a, b, c = x

        # Compute the points
        x = np.linspace(-Lc, Ln, N)
        y = np.zeros_like(x)

        # junction between combustion chamber and convergent
        y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
        # straight line in the convergent
        y[np.bitwise_and(x > x0, x < x1)] = np.tan(theta_0) * x[np.bitwise_and(x > x0, x < x1)] + q0
        # curve 1: junction between convergent and divergent, left of throat
        y[np.bitwise_and(x >= x1, x <= 0)] = -np.sqrt((factor * Rt)**2 - x[np.bitwise_and(x >= x1, x <= 0)]**2) + (1 + factor) * Rt
        # curve 2: junction between convergent and divergent, right of throat
        y[np.bitwise_and(x > 0, x < xN)] = -np.sqrt((0.382 * Rt)**2 - x[np.bitwise_and(x > 0, x < xN)]**2) + 1.382 * Rt
        # parabola: TODO here I went with luck. Why + solution seems to work???
        y[x > xN] = (-b + np.sqrt(b**2 - 4 * a * (c - x[x > xN]))) / (2 * a)

        return x, y
    

def main():
    # Ai = 63.61725123519331
    # Ae = 176.71458676442586
    # At = 7.0685834705770345
    # Lc = 5
    # theta_N = 40

    # geom = CD_Conical_Nozzle(Ai, Ae, At, 2.5, 01.5, Lc, None, theta_N)
    # # geom = CD_TOP_Nozzle(Ai, Ae, At, 2.5, Lc)
    # print(geom)

    Ai=10
    Ae=15
    At=2
    Lc=1.5
    Ld=5

    # geom = CD_Conical_Nozzle(Ai, Ae, At, 1, .5, Lc, Ld)
    geom = CD_TOP_Nozzle(Ai, Ae, At, 0.35, Lc)
    print(geom)

    radius_nozzle, radius_container = geom.Get_Points(False)
    ar_nozzle, ar_container = geom.Get_Points(True)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(nrows=2, sharex=True)

    ax[0].add_patch(patches.Polygon(radius_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))
    ax[0].add_patch(patches.Polygon(radius_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))
    ax[0].set_xlim(min(radius_container[:, 0]), max(radius_container[:, 0]))
    ax[0].set_ylim(0, max(radius_container[:, 1]))
    ax[0].set_ylabel("r [m]")

    ax[1].add_patch(patches.Polygon(ar_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))
    ax[1].add_patch(patches.Polygon(ar_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))
    ax[1].set_xlim(min(ar_container[:, 0]), max(ar_container[:, 0]))
    ax[1].set_ylim(0, max(ar_container[:, 1]))
    ax[1].set_xlabel("L [m]")
    ax[1].set_ylabel("$A/A^{*}$")

    plt.show()

if __name__ == "__main__":
    main()