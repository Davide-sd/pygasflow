"""
This module contains functions to estimate the **wall shear stress and
related coefficients for a incompressible/compressible, laminar/turbulent
flat plate**.

The two most important functions are: ``wss_ic``, ``wss_c``, as they are
wrapper functions that allows to estimates all parameters with a single call.
"""

import numpy as np
from pygasflow.utils.common import (
    _should_solver_return_dict,
    FlowResultsDict,
    FlowResultsList,
    _print_results_helper
)


def wss_lam_ic(rho, u, Re):
    """Compute the wall shear stress for a laminar incompressible flat plate.

    Parameters
    ----------
    rho : float or array_like
        Free stream density.
    u : float or array_like
        Free stream velocity.
    Re : float or array_like
        Free stream Reynolds number computed at some location.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    skin_friction
    """
    # eq (7.135)
    return 0.332 * rho * u**2 / np.sqrt(Re)


def wss_tur_ic(rho, u, Re):
    """Compute the wall shear stress for a turbulent incompressible flat plate.

    Parameters
    ----------
    rho : float or array_like
        Free stream density.
    u : float or array_like
        Free stream velocity.
    Re : float or array_like
        Free stream Reynolds number computed at some location.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    skin_friction
    """
    # eq (7.139)
    return 0.0296 * rho * u**2 / Re**0.2


def wss_lam_c(rhoinf, uinf, Reinf, Ts_Tinf, omega=0.65):
    """Compute the wall shear stress for a laminar compressible flat plate.

    Parameters
    ----------
    rhoinf : float or array_like
        Free stream density.
    uinf : float or array_like
        Free stream velocity.
    Reinf : float or array_like
        Free stream Reynolds number computed at some location.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    skin_friction
    """
    # eq (7.143)
    return wss_lam_ic(rhoinf, uinf, Reinf) * Ts_Tinf**(0.5 * (omega - 1))


def wss_tur_c(rhoinf, uinf, Reinf, Ts_Tinf, omega=0.65):
    """Compute the wall shear stress for a turbulent compressible flat plate.

    Parameters
    ----------
    rhoinf : float or array_like
        Free stream density.
    uinf : float or array_like
        Free stream velocity.
    Reinf : float or array_like
        Free stream Reynolds number computed at some location.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like

    See Also
    --------
    skin_friction
    """
    # eq (7.144)
    return wss_tur_ic(rhoinf, uinf, Reinf) * Ts_Tinf**(0.2 * (omega - 4))


def skin_friction(tau_w, q):
    """Compute the skin friction coefficient, cf.

    Parameters
    ----------
    tau_w : float or array_like
        Wall Shear Stress.
    q : float or array_like
        Dynamic pressure.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.137)
    return tau_w / q


def friction_drag_lam_c(Re, Ts_Tinf, omega=0.65):
    """Compute the friction drag coefficient, C_Df, for a laminar compressible
    flat plate.

    Parameters
    ----------
    Re : float or array_like
        Free stream Reynolds number computed with a reference length.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.167)
    return 1.328 / Re**0.5 * Ts_Tinf**(-0.175)


def friction_drag_tur_c(Re, Ts_Tinf, omega=0.65):
    """Compute the friction drag coefficient, C_Df, for a turbulent
    compressible flat plate.

    Parameters
    ----------
    Re : float or array_like
        Free stream Reynolds number computed with a reference length.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.168)
    return 0.074 / Re**0.2 * Ts_Tinf**(-0.67)


def friction_drag_lam_ic(Re):
    """Compute the friction drag coefficient, C_Df, for a laminar
    incompressible flat plate.

    Parameters
    ----------
    Re : float or array_like
        Free stream Reynolds number computed with a reference length.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.167)
    return 1.328 / Re**0.5


def friction_drag_tur_ic(Re):
    """Compute the friction drag coefficient, C_Df, for a turbulent
    incompressible flat plate.

    Parameters
    ----------
    Re : float or array_like
        Free stream Reynolds number computed with a reference length.

    Returns
    -------
    out : float or array_like
    """
    # eq (7.168)
    return 0.074 / Re**0.2


def wss_ic(rho, u, Re, laminar=True, to_dict=None):
    """Compute the wall shear stress and friction coefficients for an
    incompressible flat plate.

    Parameters
    ----------
    rho : float or array_like
        Free stream density.
    u : float or array_like
        Free stream velocity.
    Re : float or array_like
        Free stream Reynolds number computed at some location.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    tau_w : float or array_like
        Wall shear stress.
    cf : float or array_like
        Skin friction coefficient.
    CDf : float or array_like
        Friction drag coefficient.
    
    Examples
    --------

    Compute the wall shear stree, the skin friction coefficient and the
    friction drag coefficient for the following incompressible laminar flow:

    >>> from pygasflow.atd.avf.wall_shear_stress_fp import wss_ic
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> rho_inf = 0.0184101009 * ureg.kg / ureg.m**3
    >>> u_inf = 1810.2429106370967 * ureg.m / ureg.s
    >>> Re_inf = 169116173.74147075
    >>> res = wss_ic(rho_inf, u_inf, Re_inf, laminar=True, to_dict=True)
    >>> res.show()
    key     quantity              
    ------------------------------
    tau_w   τw [kg / m / s ** 2]       1.54019387
    cf      cf                         0.00005106
    CDf     CDf                        0.00010212

    See Also
    --------
    wss_c
    """
    to_dict = _should_solver_return_dict(to_dict)
    q = 0.5 * rho * u**2
    tau_w = wss_lam_ic(rho, u, Re) if laminar else wss_tur_ic(rho, u, Re)
    results = [
        tau_w,
        skin_friction(tau_w, q),
        friction_drag_lam_ic(Re) if laminar else friction_drag_tur_ic(Re)
    ]
    keys = ["tau_w", "cf", "CDf"]
    if to_dict is False:
        return FlowResultsList(
            results,
            printer=_printer_wrapper(keys)
        )
    return FlowResultsDict(
        **{k: v for k, v in zip(keys, results)},
        printer=_printer_wrapper(keys)
    )


def wss_c(rhoinf, uinf, Reinf, Ts_Tinf, laminar=True, omega=0.65, to_dict=None):
    """Compute the wall shear stress and friction coefficients for a
    compressible flat plate.

    Parameters
    ----------
    rhoinf : float or array_like
        Free stream density.
    uinf : float or array_like
        Free stream velocity.
    Reinf : float or array_like
        Free stream Reynolds number computed at some location.
    Ts_Tinf : float or array_like
        Temperature ratio T* / Tinf between the reference temperature T* and
        the free stream temperature Tinf.
    laminar : bool, optional
        Default to True, which computes the results for the laminar case.
        Set ``laminar=False`` to compute turbulent results.
    omega : float, optional
        Exponent of the viscosity power law. Default to 0.65, corresponding to
        T > 400K. Set ``omega=1`` otherwise.
    to_dict : bool, optional
        If False, the function returns a list of results. If True, it returns
        a dictionary in which the keys are listed in the Returns section.
        Default to False (return a list of results).

    Returns
    -------
    tau_w : float or array_like
        Wall shear stress.
    cf : float or array_like
        Skin friction coefficient.
    CDf : float or array_like
        Friction drag coefficient.
    
    Examples
    --------

    Compute the wall shear stree, the skin friction coefficient and the
    friction drag coefficient for the following compressible laminar flow:

    >>> from pygasflow.atd.avf.wall_shear_stress_fp import wss_c
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> T_inf = 226.509084 * ureg.K
    >>> rho_inf = 0.0184101009 * ureg.kg / ureg.m**3
    >>> u_inf = 1810.2429106370967 * ureg.m / ureg.s
    >>> Re_inf = 169116173.74147075
    >>> T_star = 921.8977042109084 * ureg.K
    >>> res = wss_c(rho_inf, u_inf, Re_inf, T_star / T_inf, 
    ...     laminar=True, to_dict=True)
    >>> res.show()
    key     quantity              
    ------------------------------
    tau_w   τw [kg / m / s ** 2]       1.20474712
    cf      cf                         0.00003994
    CDf     CDf                        0.00007988

    See Also
    --------
    wss_ic
    """
    to_dict = _should_solver_return_dict(to_dict)
    q = 0.5 * rhoinf * uinf**2
    f1 = wss_lam_c if laminar else wss_tur_c
    f2 = friction_drag_lam_c if laminar else friction_drag_tur_c
    tau_w = f1(rhoinf, uinf, Reinf, Ts_Tinf, omega=omega)
    results = [
        tau_w,
        skin_friction(tau_w, q),
        f2(Reinf, Ts_Tinf, omega=omega)
    ]
    keys = ["tau_w", "cf", "CDf"]
    if to_dict is False:
        return FlowResultsList(
            results,
            printer=_printer_wrapper(keys)
        )
    return FlowResultsDict(
        **{k: v for k, v in zip(keys, results)},
        printer=_printer_wrapper(keys)
    )


def _printer_wrapper(keys):
    labels_map = {
        "tau_w": "τw",
        "cf": "cf",
        "CDf": "CDf",
    }
    labels = [labels_map[k] for k in keys]

    def _printer(results):
        _print_results_helper(results, labels)

    return _printer
