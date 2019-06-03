import numpy as np
from pygasflow.nozzles.geometry import Internal_Tangents_To_Two_Circles

def Nozzle_Length(Rt, Re, R, K=1, alpha=15):
    """
    Compute the length of the nozzle, from the throat section to the
    exit section, with respect to a conical nozzle.
    (equation (4-7), Modern Engineering for Design of Liquid
    Propellant Rocket engines)

    Parameters
    ----------
        Rt : float
            Throat radius.
        Re : float
            Radius of the exit section.
        R : float
            Radius of the circular segment connecting the throat to the
            divergent.
        K : float
            Fractional length with respect to a 15 deg half-cone angle
            conical nozzle. Default to 1.
        alpha : float
            Half-cone angle of the divergent in degrees. Default to 15.
    
    Returns
    -------
        Ln : float
    """
    alpha = np.deg2rad(alpha)
    # area ratio
    eps = (Re / Rt)**2
    return K * (Rt * (np.sqrt(eps) - 1) + R * (1 / np.cos(alpha) - 1)) / np.tan(alpha)


# Quadratic Bézier parameterization
# https://www.benjaminmunro.com/liquid-oxygen-methane-engine-development
def Quadratic_Bezier_Parabola(P0, P2, theta_N, theta_e, t):
    """
    This is the Quadratic Bézier parameterization of a parabola.
    Given two points of a parabola and the slopes at each point,
    compute the parabola coordinates given this parameterization.

    Parameters
    ----------
        P0 : array [1x2]
            Point representing the start of the parabolic section. (xN, RN)
        P2 : array [1x2]
            Point representing the end of the parabolic section. (Ln, Re)
        theta_N : float
            Slope in radians at point P0.
        theta_e : float
            Slope in radians at point P2.
        t : array [nx1]
            Parameter of the parabola. Must be 0 <= t <= 1.
    
    Returns
    -------
        xy : array [nx2]
            Coordinate matrix. First column represents x-coords,
            second column represents y-coords.
    """

    xN, RN = P0
    Ln, Re = P2

    # coordinate of point P1, intersection of the straight lines tangent
    # to the parabola at point (xN, RN) and (Ln, Re)
    qx = (Re - RN + xN * np.tan(theta_N) - Ln * np.tan(theta_e)) / (np.tan(theta_N) - np.tan(theta_e))
    qy = np.tan(theta_N) * (qx - xN) + RN
    P1 = np.asarray([[qx, qy]])

    # make sure it's the correct shape for matrix multiplication
    # points: make it 1x2
    P0 = P0.reshape(1, 2)
    P2 = P2.reshape(1, 2)
    # t: array (list) of parameters. Make it Nx1
    t = t.reshape(len(t), 1)
    
    # Bernstein polynomials of second degree
    a = (1 - t)**2
    b = 2 * t * (1 - t)
    c = t**2
    
    # [Nx1] [1x2] = [Nx2]
    xy =  np.matmul(a, P0) + np.matmul(b, P1) + np.matmul(c, P2)
    return xy


# from a pdf file found on this mailing list:
# https://www.freelists.org/post/arocket/Parabolic-Nozzle-Approximation-Function
# pdf file created by Narayanan Komerath, 2004.
# Slides 11 -> 15.
def Rotated_Parabola(P0, P1, theta_N, theta_e, x):
    """
    Given two points of a parabola and the slopes at each point,
    compute the coordinates of the rotated parabola.

    Parameters
    ----------
        P0 : array [1x2]
            Point representing the start of the parabolic segment. (xN, RN)
        P1 : array [1x2]
            Point representing the end of the parabolic segment. (Ln, Re)
        theta_N : float
            Slope in radians at point P0.
        theta_e : float
            Slope in radians at point P1.
        x : array_like
            x-coordinates of the parabola

    Returns
    -------
        y : array_like
            y-coordinates of the parabola
    """
    xN, RN = P0
    Ln, Re = P1

    xe = Ln - xN
    ye = Re - RN

    # equation G
    num = ye * (np.tan(theta_N) + np.tan(theta_e)) - 2 * xe * np.tan(theta_e) * np.tan(theta_N)
    den = 2 * ye - xe * (np.tan(theta_N) + np.tan(theta_e))
    P = num / den
    # equation F
    num = (ye - P * xe)**2 * (np.tan(theta_N) - P)
    den = xe * np.tan(theta_N) - ye
    S = num / den
    # equation C
    # NOTE: here I needed to add the minus sign on front of S, otherwise
    # the plot would have been wrong.
    Q = -S / (2 * (np.tan(theta_N) - P))
    # equation A
    T = Q**2

    return RN + P * (x - xN) + Q + np.sqrt(S * (x - xN) + T)


def Convergent(theta, Ri, R0, Rt, factor):
    """
    Helper function to compute the important points of the convergent.
    It's possible to create both conical or curved convergents. 
    The curved configuration is composed of:
    * Circular segment at the junction with the combustion chamber (from -Lc <= x <= x0).
    * Straight segment tangent to both circles (from x0 <= x <= x1).
    * Circular segment at the outlet of the nozzle (from x1 <= x <= x0).

    Note: to create a conical convergent, set R0=0 and factor=0.

    Parameters
    ----------
        theta : float
            Convergent angle in degrees. Must be 0 < theta < 90.
        Ri : float
            Inlet section radius. Must be >= 0.
        R0 : float
            Radius of the junction between combustion chamber and convergent.
            Must be >= 0.
        Rt : float
            Throat radius.
            Must be >= 0.
        factor : float
            Ratio between the radius of the circle on the outlet of the convergent with
            the throat radius. Must be >= 0.
    
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

    assert theta > 0 and theta < 90, "The convergent angle must be 0 < theta < 90."
    assert Ri >= 0, "Inlet section radius must be >= 0."
    assert R0 >= 0, "Radius of the junction between combustion chamber and convergent must be >= 0."
    assert Rt >= 0, "Throat radius must be >= 0."
    assert factor >= 0, "Ratio of the radius' must be >= 0."

    R = factor * Rt

    theta_deg = theta
    theta = np.deg2rad(theta)

    # start the computation with reference system at the inlet area
    # intercept of the convergent straight line
    q = Ri - R0 * (1 - np.cos(theta) - np.sin(theta) * np.tan(theta))

    # coordinates of the tangent point between straight line and circle
    # downstream of the combustion chamber
    x0 = R0 * np.sin(theta)
    y0 = (Ri - R0) + R0 * np.cos(theta)

    # y-coord of the circle at throat section
    yc = R + Rt

    # compute the coordinates of the intersection point between the straight
    # line and the circle at throat section
    x1, y1 = x0, y0
    if theta_deg != 90:
        y1 = yc - R * np.cos(theta)
        x1 = (q - y1) / np.tan(theta)
    
    xc = x1 + R * np.sin(theta)

    # must obey this condition
    if x1 < x0:
        raise ValueError("The provided combination of theta, R0, factor, Rt " +
        "is not allowed because the x-coordinate of the tangent point on the " +
        "circle at throat section is less than the x-coordinate of the tangent " +
        "point on the circle at the combustion chamber section.")
    
    return x0, y0, x1, y1, xc, yc

def main():
    Lc = 5
    Rt = 1
    Ri = 3 * Rt
    R0 = 0.5 * Rt
    R0 = 0
    factor = 0
    N = 100

    x0, y0, theta_0, q0, x1, y1 = Convergent(Lc, Ri, R0, Rt, factor)

    # Compute the points
    x = np.linspace(-Lc, 0, N)
    y = np.zeros_like(x)
    # junction between combustion chamber and convergent
    y[x <= x0] = np.sqrt(R0**2 - (x[x <= x0] + Lc)**2) + (Ri - R0)
    # straight line in the convergent
    y[np.bitwise_and(x > x0, x < x1)] = np.tan(theta_0) * x[np.bitwise_and(x > x0, x < x1)] + q0
    # junction between convergent and divergent
    y[x >= x1] = -np.sqrt((factor * Rt)**2 - x[x >= x1]**2) + (1 + factor) * Rt
    
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()