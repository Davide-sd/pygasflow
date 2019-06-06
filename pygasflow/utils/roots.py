import numpy as np
from scipy.optimize import bisect

def Apply_Bisection(ratio, func, flag="sub"):
    """ Helper function used for applying the bisection method to find the 
    roots of a given function.

    Args:
        ratio:  Ratio (or parameter) passed to the function.
        func:   Function returning a number.
        flag:   Can be either "sub" or "super".
    
    Return:
        The zero of the given function.
    """
    if flag == "sub":
        mach_range = [np.spacing(1), 1]
    else:
        # realmax = np.finfo(np.float64).max

        # TODO: evaluate if this mach range is sufficient for all gamma and ratios.
        mach_range = [1, 100]

    if ratio.shape:
        M = np.zeros_like(ratio)
        for i, r in enumerate(ratio):
            M[i] = bisect(func, *mach_range, args=(r))
        return M
    
    # TODO: do I need to check for the type? ndarray vs float/int?!?!?
    # since this function is very likely to be called by a decorated function,
    # ratio should be of type ndarray. Therefore I return an array.
    return np.asarray(bisect(func, *mach_range, args=(ratio)))
