import numpy as np

def External_Tangents_To_Two_Circles(c1, c2, r1, r2):
    """
    Helper function that compute the two external tangent lines to two circles.

    Parameters
    ----------
        c1 : array_like [1x2]
            Coordinates of the center of the first circle
        c2 : array_like [1x2]
            Coordinates of the center of the second circle
        r1 : float
            Radius of the first circle
        r2 : float
            Radius of the second circle
    
    Returns
    -------
        L : array_like [2x2] [[x1, y1], [x2, y2]]
            The two points of the left tangent.
        R : array_like [2x2] [[x1, y1], [x2, y2]]
            The two points of the right tangent.
    """
    # https://it.mathworks.com/matlabcentral/answers/162912-tangent-to-two-circles
    x21, y21 = c2 - c1
    d2 = x21**2 + y21**2

    # d = c2 - c1
    # d2 = d.dot(d)

    r21 = (r2 - r1) / d2

    assert d2 >= (r2 - r1)**2, "There are no tangent lines."

    s21 = np.sqrt(d2 - (r2 - r1)**2) / d2
    # left unit vector
    u1 = np.asarray([-x21 * r21 - y21 * s21, -y21 * r21 + x21 * s21])
    # right unit vector
    u2 = np.asarray([-x21 * r21 + y21 * s21, -y21 * r21 - x21 * s21])
    # left line tangency points
    L1 = c1 + r1 * u1
    L2 = c2 + r2 * u1
    # right line tangency points
    R1 = c1 + r1 * u2
    R2 = c2 + r2 * u2

    L = np.vstack((L1, L2))
    R = np.vstack((R1, R2))

    return L, R

def Internal_Tangents_To_Two_Circles(c1, c2, r1, r2):
    """
    Helper function that compute the two internal tangent lines to two circles.

    Parameters
    ----------
        c1 : array_like [1x2]
            Coordinates of the center of the first circle
        c2 : array_like [1x2]
            Coordinates of the center of the second circle
        r1 : float
            Radius of the first circle
        r2 : float
            Radius of the second circle
    
    Returns
    -------
        L : array_like [2x2] [[x1, y1], [x2, y2]]
            The two points of the left tangent.
        R : array_like [2x2] [[x1, y1], [x2, y2]]
            The two points of the right tangent.
    """
    # https://www.lucidar.me/en/mathematics/tangent-line-segments-to-circle/
    x1, y1 = c1
    x2, y2 = c2
    x21, y21 = c2 - c1
    d = np.sqrt(x21**2 + y21**2)

    assert d**2 >= (r2 + r1)**2, "There are no tangent lines."
    
    # Compute the lenght of the tangents
    L = np.sqrt(d**2 - (r1 + r2)**2)

    # Compute the parameters
    ra = np.sqrt(L**2 + r2**2)
    Sigma_a = (1/4) * np.sqrt((d + r1 + ra) * (d + r1 - ra) * (d - r1 + ra) * (-d + r1 + ra))
    rb = np.sqrt(L**2 + r1**2)
    Sigma_b = (1/4) * np.sqrt ((d + r2 + rb) * (d + r2 - rb) * (d - r2 + rb) * (-d + r2 + rb))
    
    # Compute the first tangent
    x11 = (x1 + x2) / 2 + (x2 - x1) * (r1**2 - ra**2) / (2 * d**2) + 2 *(y1 - y2) * Sigma_a / d**2
    y11 = (y1 + y2) / 2 + (y2 - y1) * (r1**2 - ra**2) / (2 * d**2) - 2 *(x1 - x2) * Sigma_a / d**2
    x21 = (x2 + x1) / 2 + (x1 - x2) * (r2**2 - rb**2) / (2 * d**2) + 2 *(y2 - y1) * Sigma_b / d**2
    y21 = (y2 + y1) / 2 + (y1 - y2) * (r2**2 - rb**2) / (2 * d**2) - 2 *(x2 - x1) * Sigma_b / d**2   
    
    L = np.asarray([[x11, y11], [x21, y21]])
    
    # Compute second tangent
    x12 = (x1 + x2) / 2 + (x2 - x1) * (r1**2 - ra**2) / (2 * d**2) - 2 * (y1 - y2) * Sigma_a / d**2
    y12 = (y1 + y2) / 2 + (y2 - y1) * (r1**2 - ra**2) / (2 * d**2) + 2 * (x1 - x2) * Sigma_a / d**2
    x22 = (x2 + x1) / 2 + (x1 - x2) * (r2**2 - rb**2) / (2 * d**2) - 2 * (y2 - y1) * Sigma_b / d**2
    y22 = (y2 + y1) / 2 + (y1 - y2) * (r2**2 - rb**2) / (2 * d**2) + 2 * (x2 - x1) * Sigma_b / d**2   
    
    R = np.asarray([[x12, y12], [x22, y22]])

    return L, R

def Tangent_Points_On_Circle(c, r, p):
    """
    Helper function that compute the tangent line from a point to a circle.

    Parameters
    ----------
        c : array_like [1x2]
            Coordinates of the center of the circle
        r : float
            Radius of the circle
        p : array_like [1x2]
            Coordinates of the point
    
    Returns
    -------
        q1 : array_like [1x2] [x1, y1]
            Coordinates of the point of tangency to the right of the circle as
            viewed from the point p.
        q2 : array_like [1x2] [x1, y1]
            Coordinates of the point of tangency to the left of the circle as
            viewed from the point p.
    """
    # https://it.mathworks.com/matlabcentral/answers/86365-how-can-i-draw-a-tangent-line-from-a-given-point-to-a-circle-in-matlab
    d = p - c
    d2 = d.dot(d)
    q0 = c + r**2 / d2 * d
    t = r / d2 * np.sqrt(d2 - r**2) * d.dot(np.asarray([[0, 1], [-1, 0]]))
    # tangency point to the right as viewed from p
    q1 = q0 + t
    # tangency point to the left as viewed from p
    q2 = q0 - t

    return q1, q2