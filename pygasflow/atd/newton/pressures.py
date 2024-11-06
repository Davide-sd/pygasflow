import numpy as np
from pygasflow.common import pressure_coefficient as cp
from pygasflow.shockwave import rayleigh_pitot_formula


def pressure_coefficient(theta_b, alpha=0, beta=0, Mfs=None, gamma=1.4):
    """Compute the pressure coefficient, Cp, with a Newtonian flow model.

    There are four modes of operation:

    * Compute Cp with a Newtonian model:
      ``pressure_coefficient(theta_b)``
    * Compute Cp with a modified Newtonian model:
      ``pressure_coefficient(theta_b, Mfs=Mfs, gamma=gamma)``
    * Compute Cp with a Newtonian model for an axisymmetric configuration:
      ``pressure_coefficient(theta_b, alpha=alpha, beta=beta)``
    * Compute Cp with a modified Newtonian model for an axisymmetric
      configuration:
      ``pressure_coefficient(theta_b, alpha=alpha, beta=beta, Mfs=Mfs, gamma=gamma)``

    Note that for an axisymmetric configuration, the freestream flow does not
    impinge on those portions of the body surface which are inclined away
    from the freestream direction and which may, therefore, be thought
    of as lying in the "shadow of the freestream".

    Parameters
    ----------
    theta_b : float or array_like
        Local body slope [radians]. Note that for a flat plate `theta_b`
        corresponds to the angle of attack.
    alpha : float or array_like, optional
        Angle of attack [radians]. Default to 0 deg.
    beta : float or array_like, optional
        Angular position of a point on the surface of the body [radians].
        Default to 0 deg.
    Mfs : float or array_like, optional
        Free stream Mach number. Must be > 1.
    gamma : float or array_like, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Cp : float or array_like
        Pressure coefficient.

    Examples
    --------

    Compute Cp with a Newtonian model for a body with an angle of attack of
    5deg:

    >>> from pygasflow.atd.newton.pressures import pressure_coefficient
    >>> from numpy import deg2rad
    >>> pressure_coefficient(deg2rad(5))
    np.float64(0.015192246987791938)

    Compute Cp with a modified Newtonian model for a body with an angle of
    attack of 5deg with a free stream Mach number of 20 in air:

    >>> pressure_coefficient(deg2rad(5), Mfs=20, gamma=1.4)
    np.float64(0.01395744352416113)

    Compute Cp with a Newtonian flow model for a sharp cone (axisymmetric
    configuration) with theta_b=15deg, exposed to an hypersonic stream of
    Helium with free stream Mach number of 14.9 and specific heat ratio 5/3,
    with an angle of attack of 10deg, at a point located beta=45deg:

    >>> pressure_coefficient(deg2rad(15), alpha=deg2rad(10), beta=deg2rad(45))
    np.float64(0.2789909245623247)

    On the same axisymmetric configuration, compute Cp with a modified
    Netwonian flow mode:

    >>> pressure_coefficient(deg2rad(15), alpha=deg2rad(10), beta=deg2rad(45), Mfs=14.9, gamma=5.0/3.0)
    np.float64(0.2454586025665183)

    References
    ----------

    * "Basic of Aerothermodynamics", by Ernst H. Hirschel
    * "Hypersonic Aerothermodynamics" by John J. Bertin

    """
    # newtonian flow: eq (6.143)
    cp_max = 2
    if Mfs is not None:
        # modified newtonian flow: eq (6.152)
        cp_max = cp(Mfs, gamma=gamma, stagnation=True)

    # eq (6.7) - (6.8) from "Hypersonic Aerothermodynamics" by John J. Bertin
    return cp_max * (np.cos(alpha) * np.sin(theta_b) + np.sin(alpha) * np.cos(theta_b) * np.cos(beta))**2


def modified_newtonian_pressure_ratio(Mfs, theta_b, alpha=0, beta=0, gamma=1.4):
    """Compute the pressure ratio Ps / Pt2, between the static pressure in the
    shock layer and the pressure at the stagnation point.

    Parameters
    ----------
    Mfs : float or array_like, optional
        Free stream Mach number. Must be > 1.
    theta_b : float or array_like
        Local body slope [radians].
    alpha : float or array_like or None, optional
        Angle of attack [radians]. Default to 0 deg.
    beta : float or array_like or None, optional
        Angular position of a point on the surface of the body [radians].
        Default to 0 deg.
    gamma : float or array_like, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    ps_pt2 : float or array_like
        Ratio Ps / Pt2.

    Examples
    --------

    Compute Ps / Pt2 for a body with the following local body slope: [90deg,
    60deg, 30deg, 0deg], immersed on a free stream with Mach number 10 having
    0deg angle of attack:

    >>> import numpy as np
    >>> from pygasflow.atd.newton.pressures import modified_newtonian_pressure_ratio
    >>> theta_b = np.deg2rad([90, 60, 30, 0])
    >>> modified_newtonian_pressure_ratio(10, theta_b)
    array([1.        , 0.75193473, 0.25580419, 0.00773892])

    Compute Ps / Pt2 for a body with the following local body slope: [90deg,
    60deg, 30deg, 0deg], immersed on a free stream with Mach number 10 having
    33deg angle of attack:

    >>> theta_b = np.deg2rad([90, 60, 30, 0])
    >>> modified_newtonian_pressure_ratio(10, theta_b, alpha=np.deg2rad(33))
    array([0.70566393, 0.99728214, 0.79548767, 0.30207499])

    References
    ----------

    * "Hypersonic Aerothermodynamics" by John J. Bertin

    """
    pt2_pinf = rayleigh_pitot_formula(Mfs, gamma)
    # eq (6.7)
    cos_eta = np.cos(alpha) * np.sin(theta_b) + np.sin(alpha) * np.cos(theta_b) * np.cos(beta)
    cos_eta_square = cos_eta**2
    sin_eta_square = 1 - cos_eta_square
    # modified eq (6.9)
    return cos_eta_square + (1 / pt2_pinf) * sin_eta_square


def shadow_region(alpha, theta, beta=0):
    """Compute the boundaries in the circumferential direction (``phi``) in
    which the pressure coefficient ``Cp=0`` for an axisymmetric object.
    The shadow region is identified by Cp<0.

    Parameters
    ----------
    alpha : float or array_like
        Angle of attack [radians].
    theta : float or array_like
        Local body slope [radians].
    beta : float or array_like
        Sideslip angle [radians]. Default to 0 (no sideslip).

    Returns
    -------
    phi_i : float or array_like
        Lower limit of the shadow region [radians]. If NaN, there is no shadow
        region.
    phi_f : float or array_like
        Upper limit of the shadow region [radians]. If NaN, there is no shadow
        region.
    func : callable
        A lambda function with signature (alpha, theta, beta, phi), which can
        be used to test the configuration. It represents the angle between the
        velocity vector and the normal vector to the surface (see Notes).

    Notes
    -----
    The newtonian pressure coefficient is given by:

    Cp = Cpt2 * cos(eta)**2

    where cos(eta) is the angle between the vecocity vector and and normal
    vector to the surface:

    cos(eta) = cos(alpha) * cos(beta) * sin(theta) - cos(theta) * sin(phi) * sin(beta) - cos(phi) * cos(theta) * sin(alpha) * cos(beta)

    This function solves cos(eta) = 0 for phi, the angle in the circumferential
    direction.

    Let's consider an axisymmetric geometry, for example a cone. The positive
    x-axis starts from the base and move to the apex. The base of the cone lies
    on the y-z plane, where the positive z-axis points down. The angle ``phi``
    starts from the negative z-axis and is positive in the counter-clockwise
    direction.

    .. code-block:: text

                  |
          phi_i  _|_   phi_f
               /\ | /\ 
              /  \|/  \ 
        +y <-|---------|----
              \   |   /
               \ _|_ /
                  |
                  v
                  +z

    The pressure coefficient is positive for ``phi_1 <= phi <= phi_f``.
    The shadow region (where Cp<0) is identified by ``0 <= phi <= phi_1`` and
    ``phi_f <= phi <= 2 * pi``.

    Examples
    --------

    >>> import numpy as np
    >>> from pygasflow.atd.newton import shadow_region
    >>> alpha = np.deg2rad(35)
    >>> beta = np.deg2rad(0)
    >>> theta_c = np.deg2rad(9)
    >>> phi_i, phi_f, func = shadow_region(alpha, theta_c, beta)
    >>> print(phi_i, phi_f)
    1.342625208348352 4.940560098831234

    References
    ----------

    * "Tables of aerodynamic coefficients obtained from developed newtonian
      expressions for complete and partial conic and spheric bodies at combined
      angle of attack and sideslip with some comparison with hypersonic
      experimental data", by William R. Wells and William O. Armstrong, 1962.

    """
    # substitutions to shorten the expressions
    lamb = np.cos(alpha) * np.cos(beta)
    nu = -np.sin(beta)
    omega = np.sin(alpha) * np.cos(beta)

    # different from paper because:
    # https://stackoverflow.com/questions/71554667/wrong-solution-to-a-quadratic-equation/
    A = lamb * np.sin(theta)
    B = nu * np.cos(theta)
    C = omega * np.cos(theta)
    t1 = (-B + np.sqrt(B**2 - (A**2 - C**2)))/(A + C)
    t2 = (-B - np.sqrt(B**2 - (A**2 - C**2)))/(A + C)
    phi_i = 2 * np.arctan(t1)
    phi_f = 2 * np.arctan(t2) + 2 * np.pi

    func = lambda a, t, b, p: np.cos(a) * np.cos(b) * np.sin(t) - np.cos(t) * np.sin(p) * np.sin(b) - np.cos(p) * np.cos(t) * np.sin(a) * np.cos(b)
    return phi_i, phi_f, func


def pressure_coefficient_tangent_cone(theta_c, gamma=1.4):
    """Compute the pressure coefficient with the tangent-cone method.

    Parameters
    ----------
    theta_c : float or array_like
        Cone half-angle [radians].
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Cp : float or array_like
        Pressure coefficient

    Examples
    --------

    >>> from pygasflow.atd.newton.pressures import pressure_coefficient_tangent_cone
    >>> from numpy import deg2rad
    >>> pressure_coefficient_tangent_cone(deg2rad(10), 1.4)
    np.float64(0.06344098329442194)

    """
    return 2 * (gamma + 1) * (gamma + 7) / (gamma + 3)**2 * theta_c**2

def pressure_coefficient_tangent_wedge(theta_w, gamma=1.4):
    """Compute the pressure coefficient with the tangent-wedge method.

    Parameters
    ----------
    theta_w : float or array_like
        Wedge angle [radians].
    gamma : float, optional
        Specific heats ratio. Default to 1.4. Must be gamma > 1.

    Returns
    -------
    Cp : float or array_like
        Pressure coefficient

    Examples
    --------

    >>> from pygasflow.atd.newton.pressures import pressure_coefficient_tangent_wedge
    >>> from numpy import deg2rad
    >>> pressure_coefficient_tangent_wedge(deg2rad(10), 1.4)
    np.float64(0.07310818074881005)

    """
    return (gamma + 1) * theta_w**2
