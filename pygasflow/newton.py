import numpy as np
from pygasflow.common import pressure_coefficient as cp
import warnings

def pressure_coefficient(alpha, theta=None, beta=None, Mfs=None, gamma=None):
    """Compute the pressure coefficient, Cp, with a Newtonian flow model.

    There are four modes of operation:

    * Compute Cp with a Newtonian model:
      ``pressure_coefficient(alpha)``
    * Compute Cp with a modified Newtonian model:
      ``pressure_coefficient(alpha, Mfs=Mfs, gamma=gamma)``
    * Compute Cp with a Newtonian model for an axisymmetric configuration:
      ``pressure_coefficient(alpha, theta=theta, beta=beta)``
    * Compute Cp with a modified Newtonian model for an axisymmetric
      configuration:
      ``pressure_coefficient(alpha, theta=theta, beta=beta, Mfs=Mfs, gamma=gamma)``

    Note that for an axisymmetric configuration, the freestream flow does not
    impinge on those portions of the body surface which are inclined away
    from the freestream direction and which may, therefore, be thought
    of as lying in the "shadow of the freestream".

    Parameters
    ----------
    alpha : float or array_like
        Angle of attack [radians].
    theta : float or array_like, optional
        Local body slope [radians]. Note that for a flat plate alpha=theta,
        hence ``alpha`` must be provided.
    beta : float or array_like, optional
        Angular position of a point on the surface of the body [radians].
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
    0.013957443524161136

    Compute Cp with a Newtonian flow model for a sharp cone (axisymmetric
    configuration) with theta=15deg, exposed to an hypersonic stream of
    Helium with free stream Mach number of 14.9 and specific heat ratio 5/3,
    with an angle of attack of 10deg, at a point located beta=45deg:

    >>> pressure_coefficient(deg2rad(10), theta=deg2rad(15), beta=deg2rad(45))
    0.2789909245623247

    On the same axisymmetric configuration, compute Cp with a modified
    Netwonian flow mode:

    >>> pressure_coefficient(deg2rad(10), theta=deg2rad(15), beta=deg2rad(45), Mfs=14.9, gamma=5.0/3.0)
    0.24545860256651808

    References
    ----------

    * "Basic of Aerothermodynamics", by Ernst H. Hirschel
    * "Hypersonic Aerothermodynamics" by John J. Bertin

    """
    # newtonian flow: eq (6.143)
    cp_max = 2
    if (Mfs is not None) and (gamma is not None):
        # modified newtonian flow: eq (6.152)
        cp_max = cp(Mfs, gamma=gamma, stagnation=True)
    elif (Mfs is not None):
        warnings.warn(
            "`Mfs` has been provided, but `gamma` was not. Setting "
            "`gamma=1.4` and proceeding with a modified Newtonian flow "
            "model.")
        cp_max = cp(Mfs, gamma=1.4, stagnation=True)

    if (beta is None) and (theta is None):
        return cp_max * np.sin(alpha)**2
    elif (beta is not None) and (theta is not None):
        # eq (6.7) - (6.8) from "Hypersonic Aerothermodynamics" by John J. Bertin
        return cp_max * (np.cos(alpha) * np.sin(theta) + np.sin(alpha) * np.cos(theta) * np.cos(beta))**2
    raise ValueError("Combination of input parameters not supported.")
