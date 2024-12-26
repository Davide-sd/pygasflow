import numpy as np
from packaging import version
import warnings
import pygasflow
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
    elif isinstance(x, ShockResults):
        for k in x:
            x[k] = ret_correct_vals(x[k])
    elif isinstance(x, dict):
        # Many functions may return a dictionary of elements. Each value may
        # be a 1-D one-element array. If that's the case, extract that number.
        x = {k: ret_correct_vals(v) for k, v in x.items()}
    if isinstance(x, np.ndarray) and (x.ndim == 1) and (x.size == 1):
        return x[0]
    elif isinstance(x, np.ndarray) and (x.ndim == 0):
        return x[()]
    return x


class ShockResults(dict):
    """This class implements the deprecation of old keys for the results
    of pygasflow.solvers.shockwave_solver and
    pygasflow.solvers.conical_shockwave_solver.

    Please, don't use this class outside of this module: it will be removed in
    the future.
    """
    deprecation_map = {
        "m": "mu",
        "m1": "mu",
        "mn1": "mnu",
        "m2": "md",
        "mn2": "mnd",
        "pc_p1": "pc_pu",
        "rhoc_rho1": "rhoc_rhou",
        "Tc_T1": "Tc_Tu",
    }

    def __getitem__(self, k):
        if k in self.deprecation_map:
            warnings.warn(
                f"Key '{k}' is deprecated and will be removed in the future."
                f" Use '{self.deprecation_map[k]}' instead.",
                stacklevel=1
            )
            return super().__getitem__(self.deprecation_map[k])
        return super().__getitem__(k)


def _should_solver_return_dict(to_dict):
    """Initially, solvers only returned a list of results. To retrieve a
    particular quantity, users had to specify an index, which is readily
    available in the solver's documentation.

    With later version of the module, solvers can return a dictionary of
    results by setting ``to_dict=True`` in the function call. Dictionaries
    make it easier to retrieve a particular result (like downstream Mach
    number, or pressure ratio) because users only needs to remember a few keys
    like "mn", "pr", etc.

    By default, many solvers return a list of results instead of a dictionary.
    This is to maintain back-compatibility.

    However, setting ``to_dict=True`` on each solver call is a PITA. Hence,
    a shortcut is needed: set it only once (after importing the module),
    and then all solvers will automatically return a dictionary.

    Parameters
    ----------
    to_dict : bool
        Value provided in the function call.

    Returns
    -------
    to_dict : bool
        If ``to_dict=None`` in the function call (default behavior) it returns
        the value of ``pygasflow.defaults.solver_to_dict``. Otherwise it
        returns the user-provided value in the function call.
    """
    if to_dict is not None:
        return to_dict
    return pygasflow.defaults.solver_to_dict


def _print_results_helper(
    data, labels, label_formatter=None, number_formatter=None,
    blank_line=False
):
    """Helper function to print results computed by some solver.
    """
    if len(labels) != len(data):
        raise ValueError(
            f"len(labels)={len(labels)} while len(data)={len(data)}."
            " They must be the same. You are likely using a wrong printing"
            " function for the solver that produced `data`."
        )
    if number_formatter is None:
        number_formatter = pygasflow.defaults.print_number_formatter
    if label_formatter is None:
        label_formatter = "{:12}"

    data = list(data)
    if hasattr(data[0], "__iter__"):
        for l, d in zip(labels, data):
            s = label_formatter.format(l)
            s += "".join([number_formatter.format(n) for n in d])
            print(s)
    else:
        s = label_formatter + number_formatter
        for l, d in zip(labels, data):
            print(s.format(l, d))
    if blank_line:
        print()
