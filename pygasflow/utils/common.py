import numpy as np
from packaging import version
curr_numpy_ver = version.parse(np.__version__)
np_2_0_0 = version.parse("2.0.0")


def convert_to_ndarray(x):
    """
    Check if the input parameter is of type np.ndarray.
    If not, convert it to np.ndarray and make sure it is at least
    1 dimensional.
    """
    if not isinstance(x, np.ndarray):
        if curr_numpy_ver >= np_2_0_0:
            return np.atleast_1d(np.asarray(x, dtype=np.float64))
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
    if x.ndim == 0:
        if curr_numpy_ver >= np_2_0_0:
            return np.atleast_1d(np.asarray(x, dtype=np.float64))
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
    else:
        # this is mandatory, otherwise some function computes wrong
        # results if integer arguments are provided
        x = x.astype(np.float64)
    return x


def ret_correct_vals(x):
    """ Many functions implemented in this package requires their input
    arguments to be Numpy arrays, hence a few decorators take care of the
    conversion before applying the function.
    However, If I pass a scalar value to a function, I would like it to return
    a scalar value, and not a Numpy one-dimensional or zero-dimensional array.
    These function extract the scalar array from a 0-D or 1-D Numpy array.
    """
    if isinstance(x, tuple):
        # Many functions return a tuple of elements. If I give in input a single
        # mach number, it may happens that the function return a tuple of 1-D
        # Numpy arrays. But I want a tuple of numbers. Hence, the following lines
        # of code extract the values from the 1-D array and return a modified
        # tuple of elements.
        new_x = []
        for e in x:
            new_x.append(ret_correct_vals(e))
        return new_x
    elif isinstance(x, dict):
        # Many functions may return a dictionary of elements. Each value may
        # be a 1-D one-element array. If that's the case, extract that number.
        x = {k: ret_correct_vals(v) for k, v in x.items()}
    if isinstance(x, np.ndarray) and (x.ndim == 1) and (x.size == 1):
        return x[0]
    elif isinstance(x, np.ndarray) and (x.ndim == 0):
        return x[()]
    return x
