import numpy as np
from scipy.optimize import fsolve

# TODO: include phi_1 and phi_2, as in the paper.

def cotan(x):
    return 1 / np.tan(x)

def arccotan(x):
    # https://mathworld.wolfram.com/InverseCotangent.html
    return np.arctan(1 / x)

def sharp_cone_solver(Rb, theta_c, alpha, beta, L=None, Cpt2=2, phi_1=0, phi_2=2*np.pi, to_dict=False):
    """Compute axial/normal/lift/drag/moments coefficients over a sharp cone
    or a slice.

    Parameters
    ----------
    Rb : float or array_like
        Radius of the base of the cone.
    theta_c : float or array_like
        Half-cone angle [radians].
    alpha : float or array_like
        Angle of attack [radians].
    beta : float or array_like
        Angle of sideslip [radians].
    L : None or float or array_like
        Characteristic length to compute the moments. If None (default value)
        it will be set to Rb.
    Cpt2 : float or array_like, optional
        Pressure coefficient at the stagnation point behind a shock wave.
        Default to 2 (newtonian theory).
    phi_1 : float
        Initial angle of the slice. Default to 0 rad.
    phi_2 : float
        Final angle of the slice. Default to 2*pi rad.
    to_dict : bool, optional
        Default value to False, which would return a tuple of results. If True,
        a dictionary will be returned, whose keys are listed in the Returns
        section.

    Returns
    -------
    CN : float or array_like
        Normal force coefficient.
    CA : float or array_like
        Axial force coefficient.
    CY : float or array_like
        Side-force coefficient.
    CL : float or array_like
        Lift coefficient.
    CD : float or array_like
        Drag coefficient.
    L/D : float or array_like
        Lift/Drag ratio.
    Cl : float or array_like
        Rolling-moment coefficient.
    Cm : float or array_like
        Pitching-moment coefficient.
    Cn : float or array_like
        Yawing-moment coefficient.

    References
    ----------
    
    * "Tables of aerodynamic coefficients obtained from developed newtonian
      expressions for complete and partial conic and spheric bodies at combined
      angle of attack and sideslip with some comparison with hypersonic
      experimental data", by William R. Wells and William O. Armstrong, 1962.
    * Hypersonic Aerothermodynamics, John J. Bertin

    See Also
    --------
    pressure_coefficient

    Examples
    --------

    Compute the aerodynamic characteristics for a sharp cone having radius 1,
    for different angle of attacks with no sideslip:

    >>> import numpy as np
    >>> from pygasflow.atd.newton.sharp_cone import sharp_cone_solver
    >>> theta_c = np.deg2rad(10)
    >>> alpha = np.linspace(0, np.pi)
    >>> beta = np.deg2rad(0)
    >>> res = sharp_cone_solver(1, theta_c, alpha, beta, to_dict=True)

    Compute the aerodynamic characteristics for a slice of sharp cone with
    ``phi_1=90 deg`` and ``phi_2=270 deg``:

    >>> res = sharp_cone_solver(1, theta_c, alpha, beta, to_dict=True,
    ...     phi_1=np.pi/2, phi_2=1.5*np.pi)

    Notes
    -----

    The reference system is:

    .. code-block:: text
    
                -z
                 |
                 |
       y ---------
                /|
               / |
              /  |
           x /   |
                 z
    
    ``phi_1`` and ``phi_2`` are the angles starting from the -z axis, rotating
    counter-clockwise.
    """
    # substitutions to shorten the expressions
    lamb = np.cos(alpha) * np.cos(beta)
    tau = np.sqrt(1 - lamb**2)
    nu = -np.sin(beta)
    omega = np.sin(alpha) * np.cos(beta)

    # different from paper because:
    # https://stackoverflow.com/questions/71554667/wrong-solution-to-a-quadratic-equation/
    A = lamb * np.sin(theta_c)
    B = nu * np.cos(theta_c)
    C = omega * np.cos(theta_c)
    t1 = (-B + np.sqrt(B**2 - (A**2 - C**2)))/(A + C)
    t2 = (-B - np.sqrt(B**2 - (A**2 - C**2)))/(A + C)
    phi_i = 2 * np.arctan(t1)
    phi_f = 2 * np.arctan(t2)

    phi_i[np.isnan(phi_i) & (alpha < np.pi / 2)] = phi_f[np.isnan(phi_f) & (alpha < np.pi / 2)] = 0
    phi_i[np.isnan(phi_i) & (alpha > np.pi / 2)] = np.pi
    phi_f[np.isnan(phi_f) & (alpha > np.pi / 2)] = -np.pi
    phi_f += 2 * np.pi

    def CN_integral(phi):
        # eq (39)
        return (
            lamb**2 * np.sin(2 * theta_c) * np.sin(phi)
            + 2 * lamb * nu * np.cos(theta_c)**2 * np.sin(phi)**2
            - 2 * lamb * omega * (phi + np.sin(phi) * np.cos(phi)) * np.cos(theta_c)**2
            + 2 / 3 * nu**2 * np.cos(theta_c)**2 * cotan(theta_c) * np.sin(phi)**3
            + 4 / 3 * nu * omega * np.cos(theta_c)**2 * cotan(theta_c) * np.cos(phi)**3
            + 2 / 3 * omega**2 * np.cos(theta_c)**2 * cotan(theta_c) * np.sin(phi) * (np.cos(phi)**2 + 2))
    
    def CA_integral(phi):
        # eq (40)
        return (
            2 * phi * lamb**2 * np.sin(theta_c)**2
            - 2 * lamb * nu * np.sin(2 * theta_c) * np.cos(phi)
            - 2 * lamb * omega * np.sin(2 * theta_c) * np.sin(phi)
            + (phi - np.sin(phi) * np.cos(phi)) * nu**2 * np.cos(theta_c)**2
            - 2 * nu * omega * np.cos(theta_c)**2 * np.sin(phi)**2
            + (phi + np.sin(phi) * np.cos(phi)) * omega**2 * np.cos(theta_c)**2)
    
    def CY_integral(phi):
        # eq (41)
        return (
            - lamb**2 * np.sin(2 * theta_c) * np.cos(phi)
            + (phi - np.sin(phi) * np.cos(phi)) * 2 * lamb * nu * np.cos(theta_c)**2
            - 2 * lamb * omega * np.cos(theta_c)**2 * np.sin(phi)**2
            - (np.sin(phi)**2 + 2) * (2 / 3 * nu**2 * np.cos(theta_c)**2 * cotan(theta_c) * np.cos(phi))
            - 4 / 3 * nu * omega * np.cos(theta_c)**2 * cotan(theta_c) * np.sin(phi)**3
            - 2 / 3 * omega**2 * np.cos(theta_c)**2 * cotan(theta_c) * np.cos(phi)**3
        )
    
    S = (np.pi - np.abs(phi_1)) * Rb**2
    CN_func = lambda p_f, p_i: -Cpt2 * Rb**2 / (4 * S) * (CN_integral(p_f) - CN_integral(p_i))
    CA_func = lambda p_f, p_i: Cpt2 * Rb**2 / (4 * S) * (CA_integral(p_f) - CA_integral(p_i))
    CY_func = lambda p_f, p_i: Cpt2 * Rb**2 / (4 * S) * (CY_integral(p_f) - CY_integral(p_i))

    if (phi_1 >= 0) and (phi_2 > phi_1):
        phi_i[phi_i < phi_1] = phi_1
        phi_f[phi_f > phi_2] = phi_2
        CN = CN_func(phi_f, phi_i)
        CA = CA_func(phi_f, phi_i)
        CY = CY_func(phi_f, phi_i)
    else:

        phi_1 = 2 * np.pi + phi_1
        phi_i[phi_i > phi_2] = phi_2
        phi_f[phi_f < phi_1] = phi_1
        CN = CN_func(phi_2, phi_i) + CN_func(phi_f, phi_1)
        CA = CA_func(phi_2, phi_i) + CA_func(phi_f, phi_1)
        CY = CY_func(phi_2, phi_i) + CY_func(phi_f, phi_1)
    
    # TODO: wrong as they do not consider sideslip
    # eq (8-7) Hypersonic Aerothermodynamics
    CL = CN * np.cos(alpha) - CA * np.sin(alpha)
    # eq (8-9) Hypersonic Aerothermodynamics
    CD = CN * np.sin(alpha) + CA * np.cos(alpha)

    l = Rb / np.tan(theta_c)
    if L is None:
        L = Rb
    
    # eq (42)
    Cm = l / (3 * L) * (1 - np.tan(theta_c)**2) * CN
    # eq (43)
    Cn = l / (3 * L) * (1 - np.tan(theta_c)**2) * CY
    # eq (38)
    Cl = np.zeros_like(Cm)

    if to_dict:
        return {
            "CN": CN,
            "CA": CA,
            "CY": CY,
            "CL": CL,
            "CD": CD,
            "L/D": CL / CD,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn
        }

    return CN, CA, CY, CL, CD, CL / CD, Cl, Cm, Cn
