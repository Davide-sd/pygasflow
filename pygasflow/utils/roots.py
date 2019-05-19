import numpy as np
from pygasflow.utils.decorators import execution_time
from scipy.optimize import bisect

def Apply_Bisection(ratio, func, flag="sub"):
    """ Helper function used for applying the bisection method to find the 
    roots of a given function.

    Args:
        ratio:  Ratio (or parameter) passed to the function.
        func:   Function returning a number.
        flag:   Can be either "sub" or "sup".
    
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

#TODO 2: do I really get a performance boost by using Brent's Method?
#   https://math.stackexchange.com/questions/1464795/what-are-the-difference-between-some-basic-numerical-root-finding-methods
def iterate_Mach(r, gamma, func, flag, tol=1e-08):
    """
    Find the root of the function with the bisection method.

    Args:
        r:      Target value for the given ratio
        gamma:  Specific heats ratio
        func:   Function to evaluate
        flag:   'sub' or 'sup'
        tol:    Tolerance to use. Default to 1e-08.
    
    Return:
        The root of the function (ie, the Mach number).
    """
    if flag == "sub":
        return iterate_Mach_subsonic(r, gamma, func, tol)
    return iterate_Mach_supersonic(r, gamma, func, tol)

def iterate_Mach_subsonic(r, gamma, func, tol=1e-08):
    """
    Find the root of the function with the bisection method for subsonic case.

    Args:
        r:      Target value for the given ratio
        gamma:  Specific heats ratio
        func:   Function to evaluate
        tol:    Tolerance to use. Default to 1e-08.
    
    Return:
        The root of the function.
    """
    fdiff = 5
    xlo = 1e-08
    xhi = 1

    while fdiff >= tol:
        x = (xlo + xhi) / 2
        y = func(x, gamma)

        if (y > r):
            xlo = x
        else:
            xhi = x

        fdiff = np.abs(r - y)
    return x

def iterate_Mach_supersonic(r, gamma, func, tol=1e-08):
    """
    Find the root of the function with the bisection method for supersonic case.

    Args:
        r:      Target value for the given ratio
        gamma:  Specific heats ratio
        func:   Function to evaluate
        tol:    Tolerance to use. Default to 1e-08.
    
    Return:
        The root of the function.
    """
    fdiff = 5
    xlo = 1
    xhi = 100

    while fdiff >= tol:
        x = (xlo + xhi) / 2
        y = func(x, gamma)

        if (y > r):
            xhi = x
        else:
            xlo = x

        fdiff = np.abs(r - y)
    return x

@execution_time
def iterate_ShockWave_location(pr, gamma, func, area_ratio, tol=1e-08):
    """
    Find the root of the function with the bisection method.

    Args:
        pr:         Target value for the given pressure ratio
        gamma:      Specific heats ratio
        func:       Function to evaluate
        area_ratio: Initial estimate for the area ratio
        tol:        Tolerance to use. Default to 1e-08.
    
    Return:
        The root of the function.
    """
    fdiff = 5
    x_lo = 1
    x_hi = area_ratio

    while fdiff > tol:
        x = (x_lo + x_hi) / 2
        # computed_pr is the computed Pe_P0_ratio
        computed_pr = func(x, gamma)
        if computed_pr > pr:
            x_lo = x
        else:
            x_hi = x
        fdiff = np.abs(computed_pr - pr)
    return x