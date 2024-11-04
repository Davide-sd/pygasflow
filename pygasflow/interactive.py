import numpy as np
import param
import pandas as pd
import panel as pn
from pygasflow.solvers import (
    isentropic_solver, fanno_solver, rayleigh_solver,
    shockwave_solver, conical_shockwave_solver
)
from pygasflow.diagrams import (
    isentropic, fanno, rayleigh
)
from io import StringIO
from itertools import product
from bokeh.models.widgets.tables import NumberFormatter


pn.extension()


def combine(list_of_results):
    types = [type(t) for t in list_of_results]
    if not all(isinstance(t, dict) for t in list_of_results):
        raise TypeError(
            "All `results` must be instances of `dict`. Instead, this"
            " this `list_of_results` was received: %s" % types
        )

    new_results = {}
    for k in list_of_results[0]:
        arr = np.concatenate([
            d[k] if hasattr(d[k], "__iter__") else np.atleast_1d(d[k])
            for d in list_of_results
        ])
        new_results[k] = arr
    return new_results

theme = "dark"
tab_header_hover_bg = "#404040" if theme == "dark" else "#e6e6e6"
stylesheet = """
.bk-clearfix hr {
    border:none;
    border-top: 1px solid var(--design-secondary-color, var(--panel-secondary-color));
}
:host(.bk-above) .bk-header .bk-tab {font-size: 1.25em;}
.bk-active {
    color: white !important;
    background-color: var(--pn-tab-active-color) !important;
    font-weight: bold;
}
"""


class Common(param.Parameterized):
    input_value = param.String(
        default="2",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    gamma = param.String(
        default="1.4",
        label="Ratio of specific heats, γ:",
        doc="""
            Comma separated list of ratio of specific heats.
            Each value must be greater than 1."""
    )

    results = param.DataFrame(
        doc="Stores the results of the computation."
    )

    num_decimal_places = param.Integer(
        4, bounds=(2, None), label="Number of decimal places:"
    )

    computation_info = param.String(default="")

    # list of column names to be excluded by float formatter
    _float_formatters_exclusion = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_sidebar_controls()
        self._create_tabulator()

    def _parse_input_string(self, s):
        return np.fromstring(s, sep=",")

    def _df_to_csv_callback(self):
        sio = StringIO()
        self.results.copy().to_csv(sio, index=False)
        sio.seek(0)
        return sio

    def _create_tabulator(self):
        # NOTE: tabulator needs to be created at __init__ and stored in an
        # attribute in order to be able to modify the formatters when
        # num_decimal_places is changed
        self._tabulator = pn.widgets.Tabulator(
            self.param.results,
            text_align="center",
            header_align="center",
            disabled=True,
            stylesheets=[
                ":host .tabulator {font-size: 12px;}",
                ":host .tabulator-row .tabulator-cell {padding:8px 4px;}"
                ":host .tabulator-row {min-height: 4px;}",
                ":host .tabulator .tabulator-header .tabulator-col.tabulator-sortable.tabulator-col-sorter-element:hover {background-color: %s; !important}" % tab_header_hover_bg
            ],
            formatters={
                name: NumberFormatter(format="0." + "".join(["0"] * self.num_decimal_places))
                for name in self._columns.values()
                if name not in self._float_formatters_exclusion
            }
        )

    @param.depends("num_decimal_places", watch=True, on_init=True)
    def _update_formatters(self):
        if hasattr(self, "_tabulator"):
            self._tabulator.formatters = {
                name: NumberFormatter(format="0." + "".join(
                    ["0"] * self.num_decimal_places))
                for name in self._columns.values()
                if name not in self._float_formatters_exclusion
            }

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.input_parameter,
            self.param.input_value,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.gamma,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.num_decimal_places
        )

    def __panel__(self):
        return pn.Column(
            pn.Card(
                pn.pane.Bokeh(
                    type(self)._diagram(),
                    # height=400,
                    # height=400,
                    # width=800,
                    # sizing_mode="stretch_width"
                ),
                title="Diagram",
                sizing_mode='stretch_width',
                collapsed=True
            ),
            pn.Row(
                pn.pane.Str(self.param.computation_info)
            ),
            pn.Row(
                pn.widgets.FileDownload(
                    callback=self._df_to_csv_callback,
                    filename=self._filename + ".csv"
                )
            ),
            self._tabulator
        )
        return pn.panel(self.param)

    def _compute(self):
        value = self._parse_input_string(self.input_value)
        gamma = self._parse_input_string(self.gamma)
        values = product(value, gamma)

        list_of_results = []
        info = [""]

        # NOTE: isentropic_solver uses numpy vectorization for performance.
        # Here, I sacrifice performance in favor of functionality. For example,
        # suppose the user inputs "0.5, 2" for the pressure ratio. If I were to
        # use vectorized operations, an error will be raised and no results
        # will be returned because all pressure ratios must be 0 <= P/P0 <= 1.
        # Instead, by looping over each provided value, I can catch the faulty
        # value and insert an appropriate row in the dataframe.
        for v, g in values:
            try:
                # NOTE: there must be some holoviz panel machinery at play:
                # it converts `_solver` from an attribute pointing to
                # a function, to a method. Hence, the following way to invoke
                # the function.
                results = type(self)._solver(
                    self.input_parameter, v, gamma=g, to_dict=True)
            except ValueError as err:
                results = {k: np.nan for k in self._columns}
                results[self._internal_map[self.input_parameter]] = v
                error_msg = "%s" % err
                info.append(
                    "ValueError: %s Instead, %s was received." % (
                        err, g if "specific heats" in error_msg else v
                ))
            results["gamma"] = g * np.ones_like(results["m"])
            list_of_results.append(results)

        results = combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())


class Isentropic(Common, pn.viewable.Viewer):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Mach number": "m",
            "Pressure Ratio P/P0": "pressure",
            "Density Ratio rho/rho0": "density",
            "Temperature Ratio T/T0": "temperature",
            "Critical Area Ratio A/A* (subsonic case)": "crit_area_sub",
            "Critical Area Ratio A/A* (supersonic case)": "crit_area_super",
            "Mach Angle [degrees]": "mach_angle",
            "Prandtl-Meyer Angle [degrees]": "prandtl_meyer"
        },
        doc="The input parameter to be used in the isentropic computation."
    )

    _columns = {
        "gamma": "gamma",
        "m": "Mach",
        "pr": "P/P0",
        "dr": "rho/rho0",
        "tr": "T/T0",
        "prs": "P/P*",
        "drs": "rho/rho*",
        "trs": "T/T*",
        "urs": "U/U*",
        "ars": "A/A*",
        "ma": "Mach Angle [deg]",
        "pm": "Prandtl-Meyer Angle [deg]"
    }

    _internal_map = {
        "m": "m",
        "pressure": "pr",
        "density": "dr",
        "temperature": "tr",
        "crit_area_sub": "ars",
        "crit_area_super": "ars",
        "mach_angle": "ma",
        "prandtl_meyer": "pm"
    }

    _filename = "isentropic"

    _solver = isentropic_solver

    _diagram = isentropic

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m":
            self.input_value = "2"
        if self.input_parameter == "pressure":
            self.input_value = "0.12780452546295096"
        elif self.input_parameter ==  "density":
            self.input_value = "0.2300481458333117"
        elif self.input_parameter == "temperature":
            self.input_value = "0.5555555555555556"
        elif self.input_parameter in ["crit_area_sub", "crit_area_super"]:
            self.input_value = "1.6875000000000002"
        elif self.input_parameter == "mach_angle":
            self.input_value = "30"
        elif self.input_parameter == "prandtl_meyer":
            self.input_value = "26.379760813416457"

    @param.depends(
        "input_parameter", "input_value", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        self._compute()


class Fanno(Common, pn.viewable.Viewer):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Mach number": "m",
            "Critical Pressure Ratio P/P*": "pressure",
            "Critical Density Ratio rho/rho*": "density",
            "Critical Temperature Ratio T/T*": "temperature",
            "Critical Total Pressure Ratio P0/P0* (subsonic case)": "total_pressure_sub",
            "Critical Total Pressure Ratio P0/P0* (supersonic case)": "total_pressure_super",
            "Critical Velocity Ratio U/U*": "velocity",
            "Critical Friction parameter 4fL*/D (subsonic case)": "friction_sub",
            "Critical Friction parameter 4fL*/D (supersonic case)": "friction_super",
            "Entropy parameter (s*-s)/R (subsonic case)": "entropy_sub",
            "Entropy parameter (s*-s)/R (subsonic case)": "entropy_super"
        },
        doc="The input parameter to be used in the fanno computation."
    )

    _columns = {
        "gamma": "gamma",
        "m": "Mach",
        "prs": "P/P*",
        "drs": "rho/rho*",
        "trs": "T/T*",
        "tprs": "P0/P0*",
        "urs": "U/U*",
        "fps": "4fL*/D",
        "eps": "(s*-s)/R"
    }

    _internal_map = {
        "m": "m",
        "pressure": "prs",
        "density": "drs",
        "temperature": "trs",
        "total_pressure_sub": "tprs",
        "total_pressure_super": "tprs",
        "velocity": "urs",
        "friction_sub": "fps",
        "friction_super": "fps",
        "entropy_sub": "eps",
        "entropy_super": "eps"
    }

    _filename = "fanno"

    _solver = fanno_solver

    _diagram = fanno

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m":
            self.input_value = "2"
        if self.input_parameter == "pressure":
            self.input_value = "0.408248290463863"
        elif self.input_parameter ==  "density":
            self.input_value = "0.6123724356957945"
        elif self.input_parameter == "temperature":
            self.input_value = "0.6666666666666667"
        elif self.input_parameter in ["total_pressure_sub", "total_pressure_super"]:
            self.input_value = "1.6875000000000002"
        elif self.input_parameter == "velocity":
            self.input_value = "1.632993161855452"
        elif self.input_parameter in ["friction_sub", "friction_super"]:
            self.input_value = "0.3049965025814798"
        elif self.input_parameter in ["entropy_sub", "entropy_super"]:
            self.input_value = "0.523248143764548"

    @param.depends(
        "input_parameter", "input_value", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        self._compute()


class Rayleigh(Common, pn.viewable.Viewer):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Mach number": "m",
            "Critical Pressure Ratio P/P*": "pressure",
            "Critical Density Ratio rho/rho*": "density",
            "Critical Velocity Ratio U/U*": "velocity",
            "Critical Temperature Ratio T/T* (subsonic case)": "temperature_sub",
            "Critical Temperature Ratio T/T* (supersonic case)": "temperature_super",
            "Critical Total Pressure Ratio P0/P0* (subsonic case)": "total_pressure_sub",
            "Critical Total Pressure Ratio P0/P0* (supersonic case)": "total_pressure_super",
            "Critical Total Temperature Ratio T0/T0* (subsonic case)": "total_temperature_sub",
            "Critical Total Temperature Ratio T0/T0* (supersonic case)": "total_temperature_super",
            "Entropy parameter (s*-s)/R (subsonic case)": "entropy_sub",
            "Entropy parameter (s*-s)/R (subsonic case)": "entropy_super"
        },
        doc="The input parameter to be used in the rayleigh computation."
    )

    _columns = {
        "gamma": "gamma",
        "m": "Mach",
        "prs": "P/P*",
        "drs": "rho/rho*",
        "trs": "T/T*",
        "tprs": "P0/P0*",
        "ttrs": "T0/T0*",
        "urs": "U/U*",
        "eps": "(s*-s)/R"
    }

    _internal_map = {
        "m": "m",
        "pressure": "prs",
        "density": "drs",
        "temperature_sub": "trs",
        "temperature_super": "trs",
        "total_pressure_sub": "tprs",
        "total_pressure_super": "tprs",
        "total_temperature_sub": "ttrs",
        "total_temperature_super": "ttrs",
        "velocity": "urs",
        "entropy_sub": "eps",
        "entropy_super": "eps"
    }

    _filename = "rayleigh"

    _solver = rayleigh_solver

    _diagram = rayleigh

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m":
            self.input_value = "2"
        if self.input_parameter == "pressure":
            self.input_value = "0.36363636363636365"
        elif self.input_parameter ==  "density":
            self.input_value = "0.6875"
        elif self.input_parameter == "temperature":
            self.input_value = "0.5289256198347108"
        elif self.input_parameter in ["total_pressure_sub", "total_pressure_super"]:
            self.input_value = "1.5030959785260414"
        elif self.input_parameter in ["total_temperature_sub", "total_temperature_super"]:
            self.input_value = "0.793388429752066"
        elif self.input_parameter == "velocity":
            self.input_value = "1.4545454545454546"
        elif self.input_parameter in ["entropy_sub", "entropy_super"]:
            self.input_value = "1.2175752061512626"

    @param.depends(
        "input_parameter", "input_value", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        self._compute()


class ShockWave(Common, pn.viewable.Viewer):
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

    _filename = "shockwave"

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

    def __panel__(self):
        return pn.Column(
            pn.Row(
                pn.pane.Str(self.param.computation_info)
            ),
            pn.Row(
                pn.widgets.FileDownload(
                    callback=self._df_to_csv_callback,
                    filename=self._filename + ".csv"
                )
            ),
            self._tabulator
        )
        return pn.panel(self.param)

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

        results = combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())


class ConicalShockWave(Common, pn.viewable.Viewer):
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

    _filename = "conical_shockwave"

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

    def __panel__(self):
        return pn.Column(
            pn.Row(
                pn.pane.Str(self.param.computation_info)
            ),
            pn.Row(
                pn.widgets.FileDownload(
                    callback=self._df_to_csv_callback,
                    filename=self._filename + ".csv"
                )
            ),
            self._tabulator
        )
        return pn.panel(self.param)

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

        results = combine(list_of_results)
        rows = [results[k] for k in self._columns]
        data = np.vstack(rows).T
        self.computation_info = "\n".join(info)
        self.results = pd.DataFrame(data, columns=self._columns.values())


class Flow(pn.viewable.Viewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = {
            "Isentropic": Isentropic(),
            "Fanno": Fanno(),
            "Rayleigh": Rayleigh(),
            "Shock Wave": ShockWave(),
            "Conical Shock Wave": ConicalShockWave()
        }
        self.tabs = pn.Tabs(
            *list(self.components.items()),
            stylesheets=[stylesheet]
        )

    def __panel__(self):
        return self.tabs
