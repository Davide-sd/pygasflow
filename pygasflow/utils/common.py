import numpy as np
from packaging import version
import warnings
import pygasflow
from numbers import Number
curr_numpy_ver = version.parse(np.__version__)
np_2_0_0 = version.parse("2.0.0")


def convert_to_ndarray(x):
    """
    Check if the input parameter is of type np.ndarray.
    If not, convert it to np.ndarray and make sure it is at least
    1 dimensional.
    """
    if not isinstance(x, np.ndarray):
        units = x.units if _is_pint_quantity(x) else 1
        x = x.magnitude if _is_pint_quantity(x) else x
        if curr_numpy_ver >= np_2_0_0:
            return np.atleast_1d(np.asarray(x, dtype=np.float64)) * units
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64)) * units
    if x.ndim == 0:
        if curr_numpy_ver >= np_2_0_0:
            return np.atleast_1d(np.asarray(x, dtype=np.float64))
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
    else:
        # this is mandatory, otherwise some function computes wrong
        # results if integer arguments are provided
        x = x.astype(np.float64)
    return x


def _get_pint_unit_registry():
    ureg = pygasflow.defaults.pint_ureg
    if ureg is None:
        raise ValueError(
            "A `pint.Quantity was detected. However,"
            " the module doesn't know how to construct new quantities"
            " because no unit registry was set. Please, set the following"
            " attribute to the appropriate instance of `pint.UnitRegistry`:"
            " `pygasflow.defaults.pint_ureg`"
        )
    return ureg


def ret_correct_vals(x, units=None):
    """
    Many functions implemented in this package requires their input
    arguments to be Numpy arrays, hence a few decorators take care of the
    conversion before applying the function.
    However, If I pass a scalar value to a function, I would like it to return
    a scalar value, and not a Numpy one-dimensional or zero-dimensional array.
    These function extract the scalar array from a 0-D or 1-D Numpy array.

    Parameters
    ==========
    x :
        The numerical values.
    unit : str or None
        If ``pint`` is installed, ``unit`` can be a string representing the
        unit to be applied to ``x``.
    """
    if units is None:
        units = 1
    else:
        ureg = _get_pint_unit_registry()
        if units == "deg":
            units = ureg.deg

    if isinstance(x, FlowResultsList):
        # Many functions return a tuple of elements. If I give in input a single
        # mach number, it may happens that the function return a tuple of 1-D
        # Numpy arrays. But I want a tuple of numbers. Hence, the following lines
        # of code extract the values from the 1-D array and return a modified
        # tuple of elements.
        for i, v in enumerate(x):
            x[i] = ret_correct_vals(v)
        return x
    elif isinstance(x, tuple):
        return [ret_correct_vals(v) for v in x]
    elif isinstance(x, FlowResultsDict):
        for k in x:
            x[k] = ret_correct_vals(x[k])
        return x
    elif isinstance(x, dict):
        # Many functions may return a dictionary of elements. Each value may
        # be a 1-D one-element array. If that's the case, extract that number.
        for k, v in x.items():
            x[k] = ret_correct_vals(v)
    if isinstance(x, np.ndarray) or _is_pint_quantity(x):
        if (x.ndim == 1) and (x.size == 1):
            return x[0] * units
        elif (x.ndim == 0):
            return x[()] * units
        else:
            return x * units
    return x


def _sanitize_angle(angle):
    """
    Extract the appropriate numerical value of the angle.

    NOTE: this step is very important. Turns out that you can
    convert a pint unitless quantity to degrees, thus introducing
    weird errors. For example, try `ureg.Quantity(0).to("deg")`
    """
    if _is_pint_quantity(angle):
        if angle.unitless:
            angle = angle.magnitude
        else:
            angle = angle.to("deg").magnitude
    return angle


class _ShowMixin:
    @staticmethod
    def _default_printer(results):
        if isinstance(results, dict):
            labels = list(results.keys())
        else:
            raise NotImplementedError(
                f"The current `results` is of type {type(results)},"
                " from which labels cannot be extracted."
            )
        _print_results_helper(results, labels)

    def show(self):
        if self.printer is None:
            self.printer = self._default_printer
        self.printer(self)


class FlowResultsList(list, _ShowMixin):
    """
    This class implements a list with a `show()` method,
    which renders a nice table of results. This allow users to
    avoid calling `print_<flow_type>_results()`, which is combersome.
    """
    def __init__(self, iterable=(), printer=None):
        self.printer = printer
        super().__init__(iterable)


ShockResultsList = FlowResultsList


class FlowResultsDict(dict, _ShowMixin):
    """
    This class implements a dictionary with a `show()` method,
    which renders a nice table of results. This allow users to
    avoid calling `print_<flow_type>_results()`, which is combersome.
    """

    def __init__(self, *args, **kwargs):
        self.printer = kwargs.pop("printer", None)
        super().__init__(*args, **kwargs)


class ShockResults(FlowResultsDict):
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

    if isinstance(data, dict):
        keys = list(data.keys())
        max_length = max(len(k) for k in keys)
        max_length = max(8, max_length + 2)
        data = data.values()
        key_formatter = "{:" + str(max_length) + "}"
        key_label = "key"
    else:
        keys = list(range(len(data)))
        key_formatter = "{:<6}"
        key_label = "idx"

    if number_formatter is None:
        number_formatter = pygasflow.defaults.print_number_formatter
    if label_formatter is None:
        labels_length = []
        for l, d in zip(labels, data):
            if _is_pint_quantity(d) and (not d.dimensionless):
                unit = f"{d.units:~}"  # short notation (e.g., deg, m/s)
                if len(unit) > 0:
                    label_with_unit = f"{l} [{unit}]"
                else:
                    label_with_unit = f"{l}"

                labels_length.append(len(label_with_unit))
            else:
                labels_length.append(len(l))
        max_length = max(13, max(labels_length) + 2)
        label_formatter = "{:" + str(max_length) + "}"

    # header line
    header = (
        key_formatter.format(key_label) +
        label_formatter.format("quantity")
    )
    print(header)
    print("-" * len(header))

    for k, l, d in zip(keys, labels, data):
        if _is_pint_quantity(d):
            mag = d.magnitude
            unit = f"{d.units:~}"  # short notation (e.g., deg, m/s)
            if len(unit) > 0:
                label_with_unit = f"{l} [{unit}]"
            else:
                label_with_unit = f"{l}"
        else:
            mag = d
            label_with_unit = l

        # Ensure magnitude is an array for unified handling
        mag = np.atleast_1d(mag)
        # Format each numerical element
        formatted_numbers = [number_formatter.format(m) for m in mag]

        # Build the line
        s = (
            key_formatter.format(k) +
            label_formatter.format(label_with_unit) +
            "".join(formatted_numbers)
        )
        print(s)

    if blank_line:
        print()


def _is_pint_quantity(val):
    try:
        import pint
        return isinstance(val, pint.Quantity)
    except ImportError:
        return False


def _is_scalar(val):
    """Verify if the provided numerical value is scalar or iterable.

    Parameters
    ==========
    val : number, list, tuple, array

    Returns
    =======
    result : boolean
    """
    is_scalar = isinstance(val, Number)
    if not is_scalar:
        if _is_pint_quantity(val):
            is_scalar = isinstance(val.magnitude, Number)
    return is_scalar


def _parse_pint_units(input_string: str):
    ureg = pygasflow.defaults.pint_ureg
    return ureg.parse_units(input_string)


def _check_mix_of_units_and_dimensionless(quantities):
    check = [_is_pint_quantity(t) for t in quantities]
    if not (all(check) or (not any(check))):
        raise ValueError(
            "Detected a mix of units and dimensionless quantities."
            " The evaluation can't proceed. Please, check the"
            " dimensions of the provided quantities."
        )

def canonicalize_pint_dimensions(quantity):
    """
    Round exponent on all units to 6 decimal places.
    Used to override unit power when there's floating point mismatch.

    Parameters
    ----------
    quantity : pint.Quantity

    Returns
    -------
    new_quantity : pint.Quantity
    
    Examples
    --------

    >>> import pint
    >>> import pygasflow
    >>> from pygasflow.atd.avf.heat_flux_sp import heat_flux_fay_riddell
    >>> from pygasflow.utils.common import canonicalize_pint_dimensions
    >>> ureg = pint.UnitRegistry()
    >>> ureg.formatter.default_format = "~"
    >>> ureg.define("pound_mass = 0.45359237 kg = lbm")
    >>> pygasflow.defaults.pint_ureg = ureg
    >>> lbf, lbm, Btu, ft, s = ureg.lbf, ureg.lbm, ureg.Btu, ureg.ft, ureg.s
    >>> Pr = 0.7368421052631579
    >>> u_grad = 12871.540335275073 * 1 / s
    >>> rho_w = 1.2611943627968788e-05 * lbf * s ** 2 / ft ** 4
    >>> rho_e = 6.525428485981234e-07 * lbf * s ** 2 / ft ** 4
    >>> mu_w = 1.0512765233552152e-06 * lbf * s / ft ** 2
    >>> mu_e = 4.9686546490717815e-06 * lbf * s / ft ** 2
    >>> h_t2 = 11586.824574050748 * Btu / lbm
    >>> h_w = 599.5031167908519 * Btu / lbm
    >>> q = heat_flux_fay_riddell(u_grad, Pr, rho_w, mu_w, rho_e, mu_e, h_t2, h_w, sphere=True)
    >>> q
    <Quantity(2.36807802, 'force_pound * second ** 1 * british_thermal_unit / foot ** 3 / pound_mass')>

    Note the `s ** 1`. It should be just `s`. We see the exponent `1` because
    in pint sees it as a floating point number. This in turns can cause
    conversion errors, for example:

    >>> q.to("Btu / ft**2 / s")
    DimensionalityError: Cannot convert from 'force_pound * second ** 1 * british_thermal_unit / foot ** 3 / pound_mass' ([mass] / [length] ** 4.44089e-16 / [time] ** 3) to 'british_thermal_unit / foot ** 2 / second' ([mass] / [time] ** 3)

    Hence, we need to correct for this floating-point exponents in the units:

    >>> new_q = canonicalize_pint_dimensions(q)
    >>> new_q
    2.368078016743907 Btu * lbf * s / ft ** 3 / lbm

    Note that `s ** 1` disappeared, in favor of `s`. We can now convert
    this new quantity to another quantity with different units:

    >>> new_q.to("Btu / ft**2 / s")
    76.19065709613399 Btu / ft ** 2 / s

    """
    if not _is_pint_quantity(quantity):
        # Only operate on Quantities
        return quantity

    ureg = _get_pint_unit_registry()
    import pint

    new_units = pint.util.UnitsContainer({
        unit: round(exp, 6) for unit, exp in quantity.unit_items()
    })

    return ureg.Quantity(quantity.magnitude, new_units)
