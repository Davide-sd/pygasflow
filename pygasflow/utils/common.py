import numpy as np


def convert_to_ndarray(x):
    """
    Check if the input parameter is of type np.ndarray.
    If not, convert it to np.ndarray and make sure it is at least
    1 dimensional.
    """
    if not isinstance(x, np.ndarray):
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
    if x.ndim == 0:
        return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
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


class Ideal_Gas(object):
    def __init__(self, R=287.058, gamma=1.4):
        self._R = R
        self._gamma = gamma
        self._cp = self._gamma * self._R / (self._gamma - 1)
        self._cv = self._cp - R

    @property
    def R(self):
        return self._R

    @property
    def gamma(self):
        return self._gamma

    @property
    def cp(self):
        return self._cp

    @property
    def cv(self):
        return self._cv

    def solve(self, **args):
        if not args:
            raise ValueError("Need some input arguments.")

        P, T, rho = None, None, None
        # convert all keywords to lower case
        args = {k.lower(): v for k,v in args.items()}

        if "p" in args.keys():
            P = np.asarray(args["p"])
        if "t" in args.keys():
            T = np.asarray(args["t"])
        if "rho" in args.keys():
            rho = np.asarray(args["rho"])


        if P != None and T != None and rho != None:
            if (P.size != T.size) or (P.size != rho.size):
                raise ValueError("P, T, rho must have the same number of elements")
            if np.any(np.abs(P - rho * self.R * T) <= 1e-08):
                raise ValueError("The input arguments appear not to follow ideal gas low")
            return None

        if P != None and T != None:
            if P.size != T.size:
                raise ValueError("P, T must have the same number of elements")
            return P / self.R / T

        if P != None and rho != None:
            if P.size != rho.size:
                raise ValueError("P, rho must have the same number of elements")
            return P / self.R / rho

        if T != None and rho != None:
            if T.size != rho.size:
                raise ValueError("T, rho must have the same number of elements")
            return rho * self.R * T

class Flow_State(object):
    def __init__(self, **args):
        if args is None:
            raise ValueError("Must be arguments to create a flow!!!")

        self._name = ""
        self._mach = 0
        self._normal_mach = 0
        self._pressure = 0
        self._static_temperature = 0
        self._density = 0
        self._total_pressure = 0
        self._total_temperature = 0

        # convert all keywords to lower case
        args = {k.lower(): v for k,v in args.items()}


        if "name" in args.keys(): self.name = args["name"]
        if "m" in args.keys(): self.mach = args["m"]
        if "mn" in args.keys(): self.normal_mach = args["mn"]
        if "p" in args.keys(): self.pressure = args["p"]
        if "t" in args.keys(): self.static_temperature = args["t"]
        if "rho" in args.keys(): self.density = args["rho"]
        if "p0" in args.keys(): self.total_pressure = args["p0"]
        if "t0" in args.keys(): self.total_temperature = args["t0"]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def mach(self):
        return self._mach

    @mach.setter
    def mach(self, mach):
        self._mach = mach

    @property
    def normal_mach(self):
        return self._normal_mach

    @normal_mach.setter
    def normal_mach(self, normal_mach):
        self._normal_mach = normal_mach

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        self._pressure = pressure

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    @property
    def static_temperature(self):
        return self._static_temperature

    @static_temperature.setter
    def static_temperature(self, static_temperature):
        self._static_temperature = static_temperature

    @property
    def total_temperature(self):
        return self._total_temperature

    @total_temperature.setter
    def total_temperature(self, total_temperature):
        self._total_temperature = total_temperature

    @property
    def total_pressure(self):
        return self._total_pressure

    @total_pressure.setter
    def total_pressure(self, total_pressure):
        self._total_pressure = total_pressure

    def __str__(self):
        s = "State {}\n".format(self.name)
        s += "\tM\t{}\n".format(self.mach)
        s += "\tP\t{}\n".format(self.pressure)
        s += "\tT\t{}\n".format(self.static_temperature)
        s += "\trho\t{}\n".format(self.density)
        s += "\tP0\t{}\n".format(self.total_pressure)
        s += "\tT0\t{}\n".format(self.total_temperature)
        return s

    def __mul__(self, a):
        # new values
        m, pn, rn, tn, p0n, t0n = None, None, None, None, None, None

        if "m" in a.keys():
            m = a["m"]
        if "pressure_ratio" in a.keys():
            pn = self.pressure * a["pressure_ratio"]
        if "density_ratio" in a.keys():
            rn = self.density * a["density_ratio"]
        if "temperature_ratio" in a.keys():
            tn = self.static_temperature * a["temperature_ratio"]
        if "total_pressure_ratio" in a.keys():
            p0n = self.total_pressure * a["total_pressure_ratio"]
        if "total_temperature_ratio" in a.keys():
            t0n = self.total_temperature * a["total_temperature_ratio"]

        b = Flow_State(
            name = "",
            m = m,
            p = pn,
            rho = rn,
            t = tn,
            p0 = p0n,
            t0 = t0n,
        )

        return b
