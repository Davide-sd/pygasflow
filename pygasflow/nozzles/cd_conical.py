import numpy as np

from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles.utils import (
    convergent,
    nozzle_length
)

class CD_Conical_Nozzle(Nozzle_Geometry):
    """
    Convergent-Divergent nozzle with conical divergent.
    """

    def __init__(self, Ri, Re, Rt, Rj, R0, theta_c, theta_N=15, geometry_type="axisymmetric", N=100):
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
            theta_c : float
                Half angle [degrees] of the convergent.
            theta_N : float
                Half angle [degrees] of the conical divergent. Default to 15 deg.
            geometry_type : string
                Specify the geometry type of the nozzle. Can be either
                'axisymmetric' or 'planar'. 
                If 'planar' is specified, Ri, Re, Rt will be considered as
                half of the height of the respective sections (therefore, R is the
                distance from the line of symmetry and the nozzle wall).
                To compute the cross section area, "axisymmetric" uses the formula 
                A = pi * r**2, whereas "planar" uses the formula A = 2 * r. Note the
                lack of width in the planar formula, this is because in the area ratios
                it simplifies, hence it is not considered here.                
            N : int
                Number of discretization elements along the length of the nozzle. 
                Default to 100.
        """

        assert Ri > Rt and Re > Rt, "Must be Ai > At and Ae > At."
        assert isinstance(N, (int)) and N > 1, "The number of elements for discretization must be N > 1."
        assert Rj > 0, "Junction radius between Convergent and Divergent must be > 0."
        assert R0 > 0, "Junction radius between Combustion Chamber and Convergent must be > 0."
        assert theta_N > 0 and theta_N < 90, "The half cone angle of the divergent must be 0 < theta_N < 90."
        assert theta_c > 0 and theta_c < 90, "The half cone angle of the convergent must be 0 < theta_N < 90."

        super().__init__(Ri, Re, Rt, None, None, geometry_type)
        self._theta_N = theta_N
        self._theta_c = theta_c
        self._R0 = R0
        self._Rj = Rj

        # compute the intersection points of the different curves
        # composing the nozzle.
        self._compute_intersection_points()

        x, y = self.build_geometry(N)
        self._length_array = x
        self._wall_radius_array = y
        self._area_ratio_array = 2 * y / self._At
        if self._geometry_type == "axisymmetric":
            self._area_ratio_array = np.pi * y**2 / self._At
    
    def __str__(self):
        s = "C-D Conical Nozzle\n"
        s += super().__str__()
        s += "Angles:\n"
        s += "\ttheta_c\t{}\n".format(self._theta_c)
        s += "\ttheta_N\t{}\n".format(self._theta_N)
        return s
    
    def _compute_intersection_points(self):
        Ri, Rt, Re = self._Ri, self._Rt, self._Re
        R0 = self._R0
        Rj = self._Rj

        # divergent length
        self._Ld = nozzle_length(Rt, Re, Rj, 1, self._theta_N)
        
        # find interesting points for the convergent
        x0, y0, x1, y1, xc, yc = convergent(self._theta_c, Ri, R0, Rt, Rj / Rt)
        # convergent length
        self._Lc = xc
        # offset to the left, I want x=0 to be throat section
        x0 -= xc
        x1 -= xc

        # point of tangency between the smaller throat junction and the straight line
        xN = Rj * np.sin(np.deg2rad(self._theta_N))
        RN = -np.sqrt(Rj**2 - xN**2) + (Rt + Rj)

        self._intersection_points = {
            "S": [-self._Lc, Ri],   # start point
            "0": [x0, y0],  # combustion chamber circle - convergent straight line
            "1": [x1, y1],  # convergent straight line - throat circle
            "N": [xN, RN],  # throat circle - divergent straight line
            "E": [self._Ld, Re] # end point
        }
    
    @property
    def intersection_points(self):
        return self._intersection_points

    def build_geometry(self, N):
        """
        Discretize the length of the nozzle and compute the nozzle profile.

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
        Rj = self._Rj

        Lc = self.length_convergent
        Ld = self.length_divergent
        
        x0, y0 = self._intersection_points["0"]
        x1, y1 = self._intersection_points["1"]
        xN, RN = self._intersection_points["N"]

        theta_c = np.deg2rad(self._theta_c)
        theta_N = np.deg2rad(self._theta_N)
        # intercept of the straight line to the throat section
        q = y1 + x1 * np.tan(theta_c)
        # intercept for the straight line of the divergent
        qN = RN - np.tan(theta_N) * xN
        
        # Compute the points
        x = np.linspace(-Lc, Ld, N)
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
    

def main():
    Ai = 5
    Ae = 20
    At = 2
    
    geom = CD_Conical_Nozzle(Ai, Ae, At, 0.5, 1, 30, 25)
    print(geom)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    N = 1000
    x, y = geom.build_geometry(N)
    plt.plot(x, y)
    for k in geom.intersection_points.keys():
        plt.plot(*geom.intersection_points[k], 'v')
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()