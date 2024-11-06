import numpy as np
import pandas as pd
import panel as pn
import param
from pygasflow.interactive.diagrams import ObliqueShockDiagram
from pygasflow.interactive.pages.base import Common, _combine, stylesheet
from pygasflow.solvers import shockwave_solver
from itertools import product


class ObliqueShockPage(Common, pn.viewable.Viewer):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Upstream Mach number": "m1",
            "Upstream normal Mach number": "mn1",
            "Downstream normal Mach number": "mn2",
            "Pressure Ratio P2/P1": "pressure",
            "Density Ratio rho2/rho1": "density",
            "Temperature Ratio T2/T1": "temperature",
            "Total Pressure Ratio P02/P01": "total_pressure",
            "Shock wave angle, β, [degrees]": "beta",
            "Deflection angle, θ, [degrees]": "theta"
        },
        doc="Fisrt input parameter to be used in the shockwave computation."
    )

    input_parameter_2 = param.Selector(
        label="Select parameter:",
        objects={
            "Shock wave angle, β, [degrees]": "beta",
            "Deflection angle, θ, [degrees]": "theta",
            "Upstream normal Mach number": "mn1"
        },
        default="theta",
        doc="Second input parameter to be used in the shockwave computation."
    )

    input_value_2 = param.String(
        default="15",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    input_flag = param.Selector(
        label="Solution to compute:",
        objects=["weak", "strong", "both"],
    )

    _columns = {
        "gamma": "gamma",
        "m1": "Upstream Mach",
        "mn1": "Upstream normal Mach",
        "m2": "Downstream Mach",
        "mn2": "Downstream normal Mach",
        "beta": "β [deg]",
        "theta": "θ [deg]",
        "Solution": "Solution",
        "pr": "P2/P1",
        "dr": "rho2/rho1",
        "tr": "T2/T1",
        "tpr": "P02/P01"
    }

    _float_formatters_exclusion = ["Solution"]

    _internal_map = {
        "m1": "m1",
        "mn1": "mn1",
        "m2": "m2",
        "mn2": "mn2",
        "pressure": "pr",
        "density": "dr",
        "temperature": "tr",
        "total_pressure": "tpr",
        "beta": "beta",
        "theta": "theta",
    }

    def __init__(self, **params):
        params.setdefault("_filename", "oblique_shockwave")
        params.setdefault("_diagram", ObliqueShockDiagram)
        params.setdefault("page_title", "Oblique Shock")
        params.setdefault("page_description",
            "Change in properties of a 1D flow caused by an"
            " oblique shock wave. The streamlines experience some"
            " deflection angle, θ, at the shock.")
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter in ["m1", "mn1"]:
            self.input_value = "2"
        if self.input_parameter == "mn2":
            self.input_value = "0.5773502691896257"
        if self.input_parameter == "beta":
            self.input_value = "90"
        if self.input_parameter == "theta":
            self.input_value = "0"
        if self.input_parameter == "pressure":
            self.input_value = "4.5"
        elif self.input_parameter ==  "density":
            self.input_value = "2.666666666666667"
        elif self.input_parameter == "temperature":
            self.input_value = "1.6874999999999998"
        elif self.input_parameter == "total_pressure":
            self.input_value = "0.7208738614847455"

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.input_parameter,
            self.param.input_value,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.input_parameter_2,
            self.param.input_value_2,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.gamma,
            self.param.input_flag,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.num_decimal_places
        )

    @param.depends(
        "input_parameter", "input_value",
        "input_parameter_2", "input_value_2",
        "input_flag", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        value = self._parse_input_string(self.input_value)
        value_2 = self._parse_input_string(self.input_value_2)
        gamma = self._parse_input_string(self.gamma)

        values = product(value, value_2, gamma)

        list_of_results = []
        info = [""]
        flags = ["weak"]
        if self.input_flag == "strong":
            flags = ["strong"]
        elif self.input_flag == "both":
            flags = ["weak", "strong"]

        # NOTE: isentropic_solver uses numpy vectorization for performance.
        # Here, I sacrifice performance in favor of functionality. For example,
        # suppose the user inputs "0.5, 2" for the pressure ratio. If I were to
        # use vectorized operations, an error will be raised and no results
        # will be returned because all pressure ratios must be 0 <= P/P0 <= 1.
        # Instead, by looping over each provided value, I can catch the faulty
        # value and insert an appropriate row in the dataframe.
        for v1, v2, g in values:
            for f in flags:
                try:
                    results = shockwave_solver(
                        self.input_parameter, v1,
                        self.input_parameter_2, v2,
                        gamma=g, flag=f, to_dict=True)
                except ValueError as err:
                    results = {k: np.nan for k in self._columns}
                    results[self._internal_map[self.input_parameter]] = v1
                    results[self._internal_map[self.input_parameter_2]] = v2
                    error_msg = "%s" % err
                    info.append("ValueError: %s" % err)
                results["Solution"] = [f] * len(np.atleast_1d(results["m1"]))
                results["gamma"] = g * np.ones_like(results["m1"])
                list_of_results.append(results)

        results = _combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())
