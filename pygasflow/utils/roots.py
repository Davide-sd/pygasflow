import numpy as np
from scipy.optimize import bisect

def apply_bisection(ratio, func, flag="sub"):
    """ Helper function used for applying the bisection method to find the
    roots of a given function.

    Parameters
    ----------
    ratio : np.array_like
        Ratio (or parameter) passed to the function.
    func : callable
        Function returning a number.
    flag : str
        Can be either ``"sub"`` or ``"super"``.

    Returns
    -------
    roots : np.array_like
        The zero of the given function.
    """
    if flag == "sub":
        mach_range = [np.spacing(1), 1]
    else:
        # realmax = np.finfo(np.float64).max

        # TODO: evaluate if this mach range is sufficient for all gamma and ratios.
        mach_range = [1, 100]

    # Since I'm using
    M = np.zeros_like(ratio)
    for i, r in enumerate(ratio):
        M[i] = bisect(func, *mach_range, args=(r))
    return M
