import numpy as np

def axial_coeff(Cpt2, L, alpha, theta_c, x=None, beta=None):
    """Compute the axial force coefficient for a cone.

    Parameters
    ----------
    Cpt2 : float
        Pressure coefficient at the stagnation point.
    L : float
        Length of the cone.
    alpha : float
        Angle of attack [radians].
    theta_c : float
        Half-cone angle [radians].
    x : None, tuple or float
        Integration range along the axial direction. It can be:

        * None (default): the integration range will be ``[0, L]``.
        * A float, x_max: the integration range will be ``[0, x_max]``.
        * A tuple, ``(x_min, x_max)``.
    beta : tuple or float
        Integration range along the circumferential direction. It can be:

        * None (default): the integration range will be ``[0, 2 * pi]``.
        * A float, beta_max: the integration will cover the range
          ``[-beta_max, beta_max]``.
        * A tuple, ``(beta_min, beta_max)``.

        The angle must be in radians.
    
    Returns
    -------
    out : float

    Examples
    --------

    A sharp cone (theta_c=9 deg) is located in a stream with an angle of attack
    of 60 deg. Assuming Cpt2=1 and the length of the cone L=1, compute the
    axial force coefficient:

    >>> import numpy as np
    >>> from pygasflow.atd.fm.cone import axial_coeff
    >>> from pygasflow.atd.newton import shadow_region
    >>> alpha = np.deg2rad(60)
    >>> theta_c = np.deg2rad(9)
    >>> beta = shadow_region(alpha, theta_c)[0]
    >>> axial_coeff(Cpt2, L, alpha, theta_c, L, beta=beta)
    0.22862239941379614

    """
    def x_integral(x):
        return x**2 / 2
    
    def beta_integral(tc, a, b_min, b_max):
        return -1/4*b_min * np.sin(2*tc) * np.sin(2*a) + (1/4)*(b_max * np.sin(b_max) + np.cos(b_max)) * np.sin(2*tc) * np.sin(2*a) * np.sin(b_max) + (1/4)*(b_max * np.cos(b_max)**2 - 1/2 * np.sin(2*b_min)) * np.sin(2*tc) * np.sin(2*a) + (np.sin(b_max) * np.cos(b_max)**2 - np.sin(b_min) * np.cos(b_min)**2) * np.sin(a)**2 * np.cos(tc)**2 + (np.sin(b_max) - np.sin(b_min)) * np.sin(tc)**2 * np.cos(a)**2 + (2/3)*(np.sin(b_max)**3 - np.sin(b_min)**3) * np.sin(a)**2 * np.cos(tc)**2
    
    def beta_integral(tc, a, b):
        return b * np.sin(tc)**2 * np.cos(a)**2 + ((1/2)*b + (1/4) * np.sin(2*b)) * np.sin(a)**2 * np.cos(tc)**2 + (1/2) * np.sin(2*tc) * np.sin(2*a) * np.sin(b)
    
    if isinstance(x, (tuple, list)):
        x_min, x_max = x
    elif x is None:
        x_min, x_max = 0, L
    else:
        x_min, x_max = 0, x
    
    if isinstance(beta, (tuple, list)):
        beta_min, beta_max = beta
    elif beta is None:
        beta_min, beta_max = 0, 2 * np.pi
    else:
        beta_min, beta_max = -beta, beta

    return Cpt2 / np.pi / L**2 * (x_integral(x_max) - x_integral(x_min)) * (beta_integral(theta_c, alpha, beta_max) - beta_integral(theta_c, alpha, beta_min))


def normal_coeff(Cpt2, L, alpha, theta_c, x=None, beta=None):
    """Compute the normal force coefficient for a cone.

    Parameters
    ----------
    Cpt2 : float
        Pressure coefficient at the stagnation point.
    L : float
        Length of the cone.
    alpha : float
        Angle of attack [radians].
    theta_c : float
        Half-cone angle [radians].
    x : None, tuple or float
        Integration range along the axial direction. It can be:

        * None (default): the integration range will be ``[0, L]``.
        * A float, x_max: the integration range will be ``[0, x_max]``.
        * A tuple, ``(x_min, x_max)``.
    beta : tuple or float
        Integration range along the circumferential direction. It can be:

        * None (default): the integration range will be ``[0, 2 * pi]``.
        * A float, beta_max: the integration will cover the range
          ``[-beta_max, beta_max]``.
        * A tuple, ``(beta_min, beta_max)``.

        The angle must be in radians.
    
    Returns
    -------
    out : float

    Examples
    --------

    A sharp cone (theta_c=9 deg) is located in a stream with an angle of attack
    of 60 deg. Assuming Cpt2=1 and the length of the cone L=1, compute the
    axial force coefficient:

    >>> import numpy as np
    >>> from pygasflow.atd.fm.cone import normal_coeff
    >>> from pygasflow.atd.newton import shadow_region
    >>> alpha = np.deg2rad(60)
    >>> theta_c = np.deg2rad(9)
    >>> beta = shadow_region(alpha, theta_c)[0]
    >>> normal_coeff(Cpt2, L, alpha, theta_c, L, beta=beta)
    1.203768855996552

    """
    def beta_integral(tc, a, b):
        return (1/8)*(2*b + np.sin(2*b)) * np.sin(2*tc) * np.sin(2*a) + (1/3)*(np.cos(b)**2 + 2) * np.sin(a)**2 * np.sin(b) * np.cos(tc)**2 + np.sin(tc)**2 * np.sin(b) * np.cos(a)**2
    
    def x_integral(x):
        return x**2 / 2
    
    if isinstance(x, (tuple, list)):
        x_min, x_max = x
    elif x is None:
        x_min, x_max = 0, L
    else:
        x_min, x_max = 0, x
    
    if isinstance(beta, (tuple, list)):
        beta_min, beta_max = beta
    elif beta is None:
        beta_min, beta_max = 0, 2 * np.pi
    else:
        beta_min, beta_max = -beta, beta

    return Cpt2 * (x_integral(x_max) - x_integral(x_min)) * (beta_integral(theta_c, alpha, beta_max) - beta_integral(theta_c, alpha, beta_min)) / np.pi / L**2 / np.tan(theta_c)


def lift_coeff(C_A, C_N, alpha):
    """Compute the lift coefficient for a cone.

    Parameters
    ----------
    C_A : float or array_like
        Axial force coefficient.
    C_N : float or array_like
        Normal force coefficient.
    alpha : float or array_like
        Angle of attack [radians].
    
    Returns
    -------
    out : float or array_like
    """
    return C_N * np.cos(alpha) - C_A * np.sin(alpha)


def drag_coeff(C_A, C_N, alpha):
    """Compute the drag coefficient for a cone.

    Parameters
    ----------
    C_A : float or array_like
        Axial force coefficient.
    C_N : float or array_like
        Normal force coefficient.
    alpha : float or array_like
        Angle of attack [radians].
    
    Returns
    -------
    out : float or array_like
    """
    return C_N * np.sin(alpha) + C_A * np.cos(alpha)


def pitching_moment_coeff(*args, x=None, beta=None, xM=0, ref_length=None):
    """Compute the pitching moment coefficient about some axial location on
    the cone.

    There are two modes of operations:

    * ``pitching_moment_coeff(Cpt2, alpha, theta_c)``: compute the pitching
      moment with respect to the apex by integrating the pressure coefficient
      all over the cone surface.
    * ``pitching_moment_coeff(Cpt2, L, alpha, theta_c, **kwargs)``: integrate
      the pressure coefficient on a specified axial and circumferential range.

    Parameters
    ----------
    Cpt2 : float or array_like
        Pressure coefficient at the stagnation point.
    L : float or array_like
        Length of the cone
    alpha : float or array_like
        Angle of attack [radians].
    theta_c : float or array_like
        Half-cone angle [radians].
    x : None, float or tuple
        Integration range along the axial direction. It can be:

        * None: the integration range will be ``[0, L]``.
        * A float, x_max: the integration range will be ``[0, x_max]``.
        * A tuple, ``(x_min, x_max)``.
    beta : float or array_like
        Integration range along the circumferential direction. It can be:

        * None: the integration range will be ``[0, 2 * pi]``.
        * A float, beta_max: the integration will cover the range
          ``[-beta_max, beta_max]``.
        * A tuple, ``(beta_min, beta_max)``.

        The angle must be in radians.
        Default value to 2*np.pi, so the integration will cover ``[0, 2*pi]``.
    xM : float or array_like
        Reference location where to compute the pitching moment. Default to 0,
        the apex.
    ref_length : None or float
        Reference length. If None (default value), it will be set to the radius
        of the base of the cone.

    Returns
    -------
    out : float

    References
    ----------

    * Hypersonic Aerothermodynamics, Johnn J. Bertin
    * Aerodynamics for Engineers, John J. Bertin, Russel M. Cummings

    Examples
    --------

    Compute the pitching moment of a sharp cone (theta_c=9 deg) in a free
    stream with angle of attack 2 deg. Assume Cpt2=1:

    >>> import numpy as np
    >>> from pygasflow.atd.fm.cone import pitching_moment
    >>> pitching_moment(1, np.deg2rad(2), np.deg2rad(9))
    -0.14680834725345368

    A sharp cone (theta_c=9 deg) is located in a stream with an angle of attack
    of 60 deg. Assuming Cpt2=1 and the length of the cone L=1, compute the
    the pitching moment at a point located 0.6355*L from the apex:

    >>> from pygasflow.atd.newton import shadow_region
    >>> Cpt2 = 1
    >>> L = 1
    >>> alpha = np.deg2rad(60)
    >>> theta_c = np.deg2rad(9)
    >>> beta = shadow_region(alpha, theta_c)[0]
    >>> pitching_moment(Cpt2, L, alpha, theta_c, beta=beta, xM=0.6355 * L)
    -0.3639814411715981

    """
    if (len(args) > 4) or (len(args) < 3):
        raise ValueError(
            "Wrong number of arguments. Please read the documentation.")
    elif len(args) == 3:
        # eq (8.20)
        Cpt2, alpha, theta_c = args
        return -Cpt2 * np.sin(2 * alpha) / (3 * np.tan(theta_c))

    # eq (8.15)
    Cpt2, L, alpha, theta_c = args

    def beta_integral(tc, a, b):
        return (1/8)*(2*b + np.sin(2*b)) * np.sin(2*tc) * np.sin(2*a) + (1/3)*(np.cos(b)**2 + 2) * np.sin(a)**2 * np.sin(b) * np.cos(tc)**2 + np.sin(tc)**2 * np.sin(b) * np.cos(a)**2
    
    def x_integral(tc, x_M, x):
        return x**3 / 3 - x**2 * x_M * np.cos(tc)**2 / 2
    
    if isinstance(x, (tuple, list)):
        x_min, x_max = x
    elif x is None:
        x_min, x_max = 0, L
    else:
        x_min, x_max = 0, x
    
    if isinstance(beta, (tuple, list)):
        beta_min, beta_max = beta
    elif beta is None:
        beta_min, beta_max = 0, 2 * np.pi
    else:
        beta_min, beta_max = 0, beta
    
    if ref_length is None:
        mult = -Cpt2 / np.pi / L**3 / np.sin(theta_c)**2
    else:
        mult = -Cpt2 / np.pi / L**2 / ref_length * (2 / np.sin(2 * theta_c))
    
    return mult * (x_integral(theta_c, xM, x_max) - x_integral(theta_c, xM, x_min)) * (beta_integral(theta_c, alpha, beta_max) - beta_integral(theta_c, alpha, beta_min))


def center_of_pressure(Cpt2, L, alpha, theta_c, beta=np.pi, xM=0):
    """Compute the center of pressure for a sharp cone.

    Parameters
    ----------
    Cpt2 : float or array_like
        Pressure coefficient at the stagnation point.
    L : float or array_like
        Length of the cone
    alpha : float or array_like
        Angle of attack [radians].
    theta_c : float or array_like
        Half-cone angle [radians].
    beta : float or array_like
        Upper limit of the integration along the circumferential direction
        [radians]. Default value to np.pi, so the integration will cover
        ``[-pi, pi]``. If a number is provided, the integration range will be
        ``[-beta_u, beta_u]``.
    xM : float or array_like
        Reference location where to compute the pitching moment. Default to 0,
        the apex.

    Returns
    -------
    x_cp : float
        X-coordinate with respect to the apex.
    y_cp : float
        Y-coordinate.
    
    References
    ----------

    * Aerodynamics for Engineers, John J. Bertin, Russel M. Cummings
    """
    beta_u = beta
    if beta is None:
        beta_u = np.pi

    # eq (12.52) and (12.53) modified to account for beta_u and xM
    x_cp = 2 * L / 3 - xM
    y_cp = 1/9*L*(3*(2*beta_u + np.sin(2*beta_u)) * np.sin(2*theta_c) * np.sin(2*alpha) - 24*((np.sin(theta_c)**2 - np.cos(theta_c)**2 * np.cos(beta_u)**2) * np.sin(alpha)**2 - np.sin(theta_c)**2) * np.sin(beta_u) + 16 * np.sin(alpha)**2 * np.sin(beta_u)**3 * np.cos(theta_c)**2) * np.tan(theta_c)/(4*beta_u * np.sin(theta_c)**2 * np.cos(alpha)**2 + (2*beta_u + np.sin(2*beta_u)) * np.sin(alpha)**2 * np.cos(theta_c)**2 + 2 * np.sin(2*theta_c) * np.sin(2*alpha) * np.sin(beta_u))

    return x_cp, y_cp


def sharp_cone_solver(Cpt2, L, alpha, theta_c, beta=None, xM=0, ref_length=None, to_dict=False):
    """Compute axial/normal/lift/drag/pitching moment coefficients and center
    of pressure over a sharp cone.

    Parameters
    ----------
    Cpt2 : float or array_like
        Pressure coefficient at the stagnation point.
    L : float
        Length of the cone.
    alpha : float or array_like
        Angle of attack [radians].
    theta_c : float or array_like
        Half-cone angle [radians].
    beta : float or array_like, optional
        Upper limit of the integration along the circumferential direction
        [radians]. Default value to np.pi, so the integration will cover
        ``[-pi, pi]``.
    xM : float or array_like, optional
        Reference location where to compute the pitching moment. Default to 0,
        the apex.
    ref_length : None or float, optional
        Reference length to compute the pitching moment. If None (default
        value), it will be set to the radius of the base of the cone.
    to_dict : bool, optional
        Default value to False, which would return a tuple of results. If True,
        a dictionary will be returned, whose keys are listed in the Returns
        section.
    
    Returns
    -------
    CA : float or array_like
        Axial force coefficient.
    CN : float or array_like
        Normal force coefficient.
    CL : float or array_like
        Lift coefficient.
    CD : float or array_like
        Drag coefficient.
    L/D : float or array_like
        Lift over Drag ratio.
    CM : float or array_like
        Pitching moment coefficient.
    xp : float or array_like
        Axial location of the center of pressure.
    yp : float or array_like
        Location of the center of pressure in the normal direction.
    
    See Also
    --------
    axial_coeff, normal_coeff, lift_coeff, drag_coeff, pitching_moment_coeff,
    center_of_pressure

    """
    CA = axial_coeff(Cpt2, L, alpha, theta_c, beta=beta)
    CN = normal_coeff(Cpt2, L, alpha, theta_c, beta=beta)
    CL = lift_coeff(CA, CN, alpha)
    CD = drag_coeff(CA, CN, alpha)
    L_D = CL / CD
    CM = pitching_moment_coeff(Cpt2, L, alpha, theta_c, beta=beta, xM=xM, ref_length=ref_length)
    xp, yp = center_of_pressure(Cpt2, L, alpha, theta_c, beta=beta, xM=xM)

    if to_dict:
        return {
            "CA": CA,
            "CN": CN,
            "CL": CL,
            "CD": CD,
            "L/D": L_D,
            "CM": CM,
            "xp": xp,
            "yp": yp
        }
    return CA, CN, CL, CD, L_D, CM, xp, yp


def blunted_cone_solver(Rn, Rb, theta_c, alpha, Cpt2=2):
    # def beta_integral_sphere(b):
    #     return b * np.cos(alpha)**2 * np.cos(phi)**2 + (b + np.sin(2*b)/2) * np.sin(alpha)**2 * np.sin(phi)**2/2 + np.sin(2*alpha) * np.sin(b) * np.sin(2*phi)/2
    # def beta_integral_CA_cone(b):
    #     return b * np.sin(theta_c)**2 * np.cos(alpha)**2 + (b/2 + np.sin(b) * np.cos(b)/2) * np.sin(alpha)**2 * np.cos(theta_c)**2 + np.sin(2*alpha) * np.sin(b) * np.sin(2*theta_c)/2
    
    R_ratio = Rn / Rb
    CA = 2 * Cpt2 * R_ratio**2 * ((0.25 * np.cos(alpha)**2 * (1 - np.sin(theta_c)**4) + 0.125 * np.sin(alpha)**2 * np.cos(theta_c)**4) + np.tan(theta_c) * (np.cos(alpha)**2 * np.sin(theta_c)**2 + 0.5 * np.sin(alpha)**2 * np.cos(theta_c)**2) * ((1 / R_ratio - np.cos(theta_c)) / np.tan(theta_c) * np.cos(theta_c) + (1 / R_ratio - np.cos(theta_c))**2 / (2 * np.tan(theta_c))))

    CN = 2 * Cpt2 * R_ratio**2 * (0.25 * np.sin(alpha) * np.cos(alpha) * np.cos(theta_c)**4 + np.sin(alpha) * np.cos(alpha) * np.sin(theta_c) * np.cos(theta_c) * ((1 / R_ratio - np.cos(theta_c)) / np.tan(theta_c) * np.cos(theta_c) + (1 / R_ratio - np.cos(theta_c))**2 / (2 * np.tan(theta_c))))

    CL = lift_coeff(CA, CN, alpha)
    CD = drag_coeff(CA, CN, alpha)
    L_D = CL / CD

    return CA, CN, CL, CD, L_D
