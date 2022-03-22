import numpy as np
from pygasflow.common import pressure_coefficient as cp
from pygasflow.shockwave import rayleigh_pitot_formula
import warnings


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

    >>> from pygasflow.newton import pressure_coefficient
    >>> from numpy import deg2rad
    >>> pressure_coefficient(deg2rad(5))
    0.015192246987791938

    Compute Cp with a modified Newtonian model for a body with an angle of
    attack of 5deg with a free stream Mach number of 20 in air:

    >>> pressure_coefficient(deg2rad(5), Mfs=20, gamma=1.4)
    0.01395744352416113

    Compute Cp with a Newtonian flow model for a sharp cone (axisymmetric
    configuration) with theta_b=15deg, exposed to an hypersonic stream of
    Helium with free stream Mach number of 14.9 and specific heat ratio 5/3,
    with an angle of attack of 10deg, at a point located beta=45deg:

    >>> pressure_coefficient(deg2rad(15), alpha=deg2rad(10), beta=deg2rad(45))
    0.2789909245623247

    On the same axisymmetric configuration, compute Cp with a modified
    Netwonian flow mode:

    >>> pressure_coefficient(deg2rad(15), alpha=deg2rad(10), beta=deg2rad(45), Mfs=14.9, gamma=5.0/3.0)
    0.2454586025665183

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
    >>> from pygasflow.newton import modified_newtonian_pressure_ratio
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


def shadow_region(alpha, theta):
    """Compute the upper limit of the shadow region in which the pressure
    coefficient is Cp=0.

    This function solves:

    cos(alpha) * sin(theta) + sin(alpha) * cos(theta) * cos(beta) = 0

    for beta in [0, 2*pi].

    Parameters
    ----------
    alpha : float or array_like
        Angle of attack [radians].
    theta : float or array_like
        Local body slope [radians].

    Returns
    -------
    beta : tuple (s1, s2)
        Upper limit of the angular position of a point on the surface of the
        body where Cp=0 [radians]. In the interval beta in [0, 2*pi] there are
        two solutions, s1 and s2. The upper limit is s1.

    Examples
    --------

    Compute the upper limit in which Cp=0 for a sharp cone (theta_c=15deg)
    with an angle of attack of 20deg:

    >>> import numpy as np
    >>> from pygasflow.newton import shadow_region
    >>> sol = shadow_region(np.deg2rad(20), np.deg2rad(15))
    >>> np.rad2deg(sol[0])
    137.40738758045816

    References
    ----------

    * "Hypersonic Aerothermodynamics" by John J. Bertin

    """
    t = -np.tan(theta) / np.tan(alpha)
    t[t > 1] = 1
    t[t < -1] = -1
    s1 = np.arccos(t)

    if hasattr(s1, "__iter__"):
        idx = np.iscomplex(s1)
        s1[idx] = np.nan
    else:
        if np.iscomplex(s1):
            s1 = np.nan
    s2 = 2 * np.pi - s1
    return s1, s2


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

    >>> from pygasflow.atd.tangent_cone_wedge import pressure_coefficient_tangent_cone
    >>> from numpy import deg2rad
    >>> pressure_coefficient_tangent_cone(deg2rad(10), 1.4)
    0.06344098329442194

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

    >>> from pygasflow.atd.tangent_cone_wedge import pressure_coefficient_tangent_wedge
    >>> from numpy import deg2rad
    >>> pressure_coefficient_tangent_wedge(deg2rad(10), 1.4)
    0.07310818074881005

    """
    return (gamma + 1) * theta_w**2
