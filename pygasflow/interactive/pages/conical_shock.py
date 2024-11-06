import numpy as np
import pandas as pd
import panel as pn
import param
from pygasflow.interactive.diagrams import ConicalShockDiagram
from pygasflow.interactive.pages.base import Common, _combine, stylesheet
from pygasflow.solvers import conical_shockwave_solver
from itertools import product


class ConicalShockPage(Common, pn.viewable.Viewer):
    input_mach_value = param.String(
        default="2",
        label="Upstream Mach Number:",
        doc="Comma separated list of values."
    )

    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Mach number at the cone's surface": "mc",
            "Half cone angle, θ_c, [degrees]": "theta_c",
            "Shock wave angle, β, [degrees]": "beta"
        },
        default="theta_c",
        doc="Second input parameter to be used in the shockwave computation."
    )

    input_value = param.String(
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
        "m": "Upstream Mach",
        "mc": "M_c",
        "theta_c": "θ_c [deg]",
        "beta": "β [deg]",
        "delta": "δ [deg]",
        "Solution": "Solution",
        "pr": "P2/P1",
        "dr": "rho2/rho1",
        "tr": "T2/T1",
        "tpr": "P02/P01",
        "rhoc_rho1": "rho_c/rho_1",
        "Tc_T1": "T_c/T1"
    }

    _float_formatters_exclusion = ["Solution"]

    _internal_map = {
        "m": "m",
        "mc": "mc",
        "theta_c": "theta_c",
        "beta": "beta",
        "delta": "delta",
        "mn2": "mn2",
        "pressure": "pr",
        "density": "dr",
        "temperature": "tr",
        "total_pressure": "tpr",
    }

    def __init__(self, **params):
        params.setdefault("_filename", "conical_shockwave")
        params.setdefault("_diagram", ConicalShockDiagram)
        params.setdefault("page_title", "Conical Shock")
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "mc":
            self.input_value = "1.7068679592453362"
        if self.input_parameter == "theta_c":
            self.input_value = "15"
        if self.input_parameter == "beta":
            self.input_value = "33.91469752764406"

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.input_mach_value,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.input_parameter,
            self.param.input_value,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.gamma,
            self.param.input_flag,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.num_decimal_places
        )

    @param.depends(
        "input_mach_value",
        "input_parameter", "input_value",
        "input_flag", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        mach_value = self._parse_input_string(self.input_mach_value)
        other_value = self._parse_input_string(self.input_value)
        gamma = self._parse_input_string(self.gamma)

        values = product(mach_value, other_value, gamma)

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
                    results = conical_shockwave_solver(
                        v1, self.input_parameter, v2,
                        gamma=g, flag=f, to_dict=True)
                except ValueError as err:
                    results = {k: np.nan for k in self._columns}
                    results[self._internal_map[self.input_parameter]] = v1
                    results[self._internal_map[self.input_parameter_2]] = v2
                    error_msg = "%s" % err
                    info.append("ValueError: %s" % err)
                results["Solution"] = [f] * len(np.atleast_1d(results["m"]))
                results["gamma"] = g * np.ones_like(results["m"])
                list_of_results.append(results)

        results = _combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())
