import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import interpolate
from pygasflow.utils.common import ret_correct_vals

class Rao_Parabola_Angles(object):
    """
    This class is used to generate the relation from the extracted data
    of plot 4-16, page 77, Modern Engineering for Design of
    Liquid-Propellant Rocket Engines, D.K. Huzel and D.H. Huang.

    Useful for finding the Rao's parabola angles given the fractional
    length and the area ratio, or vice-versa.

    The plot relate the Rao's TOP nozzle:

    * theta_n, initial parabola angle
    * theta_e, final parabola angle

    with the expansion ratio, epsilon, varying the fractional nozzle
    length Lf.

    Note that the data has been manually extracted from the plot in the
    book, hence it is definitely a huge "approximation".
    If you have the original Rao's plot, you can probably extract better
    data.

    Examples
    --------

    Visualize the plot:

    .. plot::
       :context: reset
       :format: python
       :include-source: True

       from pygasflow import Rao_Parabola_Angles
       p = Rao_Parabola_Angles()
       p.plot()

    Compute the angles at the end points of Rao's parabola for a nozzle with
    fractional length 68 and area ratio 35:

    >>> from pygasflow import Rao_Parabola_Angles
    >>> p = Rao_Parabola_Angles()
    >>> print(p.angles_from_Lf_Ar(68, 35))
    (36.11883335948745, 10.695233384809715)

    Compute the area ratio for a nozzle with fractional length 68 and an angle
    at the start of the parabola of 35 degrees:

    >>> p.area_ratio_from_Lf_angle(68, theta_n=35)
    24.83022334667575

    """

    def __init__(self):
        # path of the folder containing this file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # path of the folder containing the data of the plot
        data_dir = os.path.join(current_dir, "plot-4-16-data")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))]

        # substitute "," with "." for decimal separator
        func = lambda s: float(s.replace(b",", b"."))
        c = {
            0: func,
            1: func,
        }

        theta_n = dict()
        theta_e = dict()

        for f in data_files:
            data = np.loadtxt(f, delimiter=";", converters=c)
            basename = os.path.basename(f).split(".")[0]
            basename = basename.split(" ")
            lf = basename[1].split("_")[1]

            # interpolation spline
            # InterpolatedUnivariateSpline forces smoothing = 0 on the spline
            s = interpolate.InterpolatedUnivariateSpline(data[:, 0], data[:, 1])

            if basename[0] == "theta_n":
                theta_n[int(lf)] = {
                    "data": data,
                    "spline": s
                }
            else:
                theta_e[int(lf)] = {
                    "data": data,
                    "spline": s
                }
        self._theta_n = theta_n
        self._theta_e = theta_e

    def _find_Lf_range(self, Lf):
        """
        Compute the interval where to interpolate the fractional length.

        Parameters
        ----------
        Lf : float
            Fractional length in percent. Must be 60 <= Lf <= 100.

        Returns
        -------
        Lf_inf : float
        Lf_sup : float
        """
        Lf /= 10
        Lf_inf = int(np.floor(Lf)) * 10
        Lf_sup = int(np.ceil(Lf)) * 10
        return Lf_inf, Lf_sup

    def angles_from_Lf_Ar(self, Lf, Ar):
        """
        Compute the angles at the end points of Rao's parabola.
        Note that linear interpolation is used if Lf is different
        than 60, 70, 80, 90, 100.

        Parameters
        ----------
        Lf : float
            Fractional length in percent. Must be 60 <= Lf <= 100.
        Ar : float
            Area ratio. Must be 5 <= Ar <= 50.

        Returns
        -------
        theta_n : float
            Angle in degrees at the start of the parabola.
        theta_e : float
            Angle in degrees at the end of the parabola (at the exit section).
        """
        if (Lf < 60) or (Lf > 100):
            raise ValueError("Fractional length must be 60 <= Lf <= 100.")
        if (Ar < 5) or (Ar > 50):
            raise ValueError("Area ratio must be 5 <= Ar <= 50.")

        Lf_inf, Lf_sup = self._find_Lf_range(Lf)

        if Lf == Lf_inf or Lf == Lf_sup:
            thn = self._theta_n[Lf]["spline"](Ar)
            the = self._theta_e[Lf]["spline"](Ar)
        else:
            # use linear interpolation in the range (Lf_inf, Lf_sup)
            thn_inf = self._theta_n[Lf_inf]["spline"](Ar)
            thn_sup = self._theta_n[Lf_sup]["spline"](Ar)
            the_inf = self._theta_e[Lf_inf]["spline"](Ar)
            the_sup = self._theta_e[Lf_sup]["spline"](Ar)

            # I could use interpolate.interp1d, but it would end up in more lines of code
            thn = thn_inf + (Lf - Lf_inf) / (Lf_sup - Lf_inf) * (thn_sup - thn_inf)
            the = the_inf + (Lf - Lf_inf) / (Lf_sup - Lf_inf) * (the_sup - the_inf)
        return thn, the

    def area_ratio_from_Lf_angle(self, Lf=60, **kwargs):
        """
        Compute the Area Ratio given Lf, theta_n or theta_e.

        Parameters
        ----------
        Lf : float
            Fractional length in percent. Must be 60 <= Lf <= 100.

        kwargs : float.
            It can either be:

            theta_n : float
                The angle in degrees at the start of the parabola.
            theta_e : float
                The angle in degrees at the end of the parabola.

        Returns
        -------
        Ar : float
            The area ratio corresponding to the given angle and Lf.
            Note that linear interpolation is used if Lf is different
            than 60, 70, 80, 90, 100.
        """
        if (Lf < 60) or (Lf > 100):
            raise ValueError("Fractional length must be 60 <= Lf <= 100.")

        # convert all keywords to lower case
        kwargs = {k.lower(): v for k,v in kwargs.items()}
        angle, angle_name, theta_dict = None, None, None

        if "theta_n" in kwargs.keys():
            angle = kwargs["theta_n"]
            angle_name = "theta_n"
            theta_dict = self._theta_n
        if "theta_e" in kwargs.keys():
            angle = kwargs["theta_e"]
            angle_name = "theta_e"
            theta_dict = self._theta_e
        if angle == None:
            raise ValueError("Either theta_n or theta_e must be given in input.")

        # function to compute area ratio given the angle and fractional length
        # https://stackoverflow.com/questions/1029207/interpolation-in-scipy-finding-x-that-produces-y
        def func(data):
            x = data[:, 0]
            y = data[:, 1]
            y_reduced = y - angle
            f_reduced = interpolate.InterpolatedUnivariateSpline(x, y_reduced)
            return f_reduced.roots()

        if Lf in [60, 70, 80, 90, 100]:
            Ar = func(theta_dict[int(Lf)]["data"])
        else:
            Lf_inf, Lf_sup = self._find_Lf_range(Lf)

            def Min_Max(arr):
                return min(arr), max(arr)

            # TODO: isn't it better to compute min, max for each curve in the
            # __init__ and save the values in the dictionary?

            # for both cases (theta_n, theta_e) we must satisfy the same condition
            _min_inf, _max_inf = Min_Max(theta_dict[Lf_inf]["data"][:, 1])
            _min_sup, _max_sup = Min_Max(theta_dict[Lf_sup]["data"][:, 1])
            if angle < _min_inf or angle > _max_sup:
                raise ValueError("Could not interpolate the value for the given angle.\n" +
                    "\tIt must be {} <= {} <= {}".format(_min_inf, angle_name, _max_sup)
                )

            # compute area rataio corresponding to the inf and sup limits of Lf
            Ar_inf = func(theta_dict[Lf_inf]["data"])
            Ar_sup = func(theta_dict[Lf_sup]["data"])

            # use linear interpolation in the range (Lf_inf, Lf_sup)
            Ar = Ar_inf + (Lf - Lf_inf) / (Lf_sup - Lf_inf) * (Ar_sup - Ar_inf)

        return ret_correct_vals(Ar)

    def plot(self, N=30):
        """
        Plot the relation.

        Parameters
        ----------
        N : int
            Number of interpolated point for each curve. Default to 30.
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # colors
        cmap = plt.get_cmap('hsv')
        cNorm  = colors.Normalize(vmin=0, vmax=len(self._theta_e.keys()))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        c = [scalarMap.to_rgba(i) for i in range(len(self._theta_e.keys()))]

        x = np.linspace(5, 50, N)

        # need to loop in this way to insert the line in ordered Lf
        k1 = list(self._theta_e.keys())
        k1.sort()
        for k in k1:
            ax1.plot(x, self._theta_e[k]["spline"](x), label="Lf = {}%".format(k))

        # use a different color map
        k2 = list(self._theta_n.keys())
        k2.sort()
        for k, c in zip(k2, c):
            ax2.plot(x, self._theta_n[k]["spline"](x), label="Lf = {}%".format(k), color=c)

        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 50)
        ax2.set_ylim(0, 50)
        plt.title("plot 4-16")
        ax1.set_xlabel(r"Area Ratio, $\varepsilon$")
        ax1.set_ylabel(r"$\theta_e$ [deg]")
        ax2.set_ylabel(r"$\theta_n$ [deg]")

        # TODO: definitely an hacky way to do it...
        ax1.yaxis.set_label_coords(-0.05, 0.3)
        ax2.yaxis.set_label_coords(1.05, 0.7)

        # two different legends
        ax1.legend(loc='lower left')
        legend1 = plt.legend(loc='upper left')
        ax2.add_artist(legend1)

        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='-', alpha=0.7)
        ax1.grid(which='minor', linestyle=':', alpha=0.5)
        plt.show()
