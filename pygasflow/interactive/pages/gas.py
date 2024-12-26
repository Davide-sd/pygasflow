import numpy as np
import pandas as pd
import panel as pn
import param
from pygasflow.interactive.diagrams import GasDiagram
from pygasflow.interactive.pages.base import (
    BasePage,
    BaseSection,
    _combine,
    stylesheet,
    _parse_input_string
)
from pygasflow.solvers import (
    gas_solver,
    ideal_gas_solver,
    sonic_condition
)
from pygasflow.common import sound_speed
from itertools import product


class GasSection(BaseSection):
    input_parameter_1 = param.String("cp")

    input_value_1 = param.Array(np.array([1004.675]))

    input_parameter_2 = param.String("cv")

    input_value_2 = param.Array(np.array([717.625]))

    input_temperature = param.Array(np.array([298]))

    def __init__(self, **params):
        params.setdefault("solver", gas_solver)
        params.setdefault("title", "Gas")
        params.setdefault("diagrams", [GasDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="gas",
                columns_map={
                    "gamma": "gamma",
                    "r": "R",
                    "Cp": "Cp",
                    "Cv": "Cv",
                    "drs": "rho0 / rho*",
                    "prs": "p0 / p*",
                    "ars": "a0 / a*",
                    "trs": "T0 / T*",
                    "T": "T [K]",
                    "a": "Sound Speed [m/s]",
                },
            ),
        ])
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)
        self.compute()

    @param.depends(
        "input_parameter_1", "input_value_1",
        "input_parameter_2", "input_value_2",
        "input_temperature", watch=True
    )
    def compute(self):
        info = []
        list_of_results = []

        for v1, v2, t in product(
            self.input_value_1, self.input_value_2, self.input_temperature
        ):
            try:
                results = self.solver(
                    self.input_parameter_1, v1,
                    self.input_parameter_2, v2,
                    to_dict=True
                )
                results["T"] = t
                results["a"] = sound_speed(results["gamma"], results["R"], t)
                results.update(
                    sonic_condition(results["gamma"], to_dict=True)
                )
            except ValueError as err:
                info.append(f"ValueError: {err}")
                results = {
                    self.input_parameter_1: v1,
                    self.input_parameter_2: v2,
                    "drs": np.nan,
                    "prs": np.nan,
                    "ars": np.nan,
                    "trs": np.nan,
                    "T": t,
                    "a": np.nan,
                }

            list_of_results.append(results)

        results = _combine(list_of_results)
        self.error_log = "\n".join(info)
        df = pd.DataFrame( results )
        df.rename(columns=self.tabulators[0].columns_map, inplace=True)
        df = df.reindex(columns=self.tabulators[0].columns_map.values())
        self.results = df

    @param.depends("results", watch=True, on_init=True)
    def update_dataframe(self):
        self.tabulators[0].results = self.results


class IdealGasSection(BaseSection):
    input_wanted = param.String("p")

    input_value_1 = param.Array(np.array([1.2259]))

    input_value_2 = param.Array(np.array([287.05]))

    input_value_3 = param.Array(np.array([288]))

    # to compute sonic condition and sound speed
    gamma = param.Array(np.array([1.4]))

    def __init__(self, **params):
        params.setdefault("solver", ideal_gas_solver)
        params.setdefault("title", "Ideal Gas")
        params.setdefault("tabulators", [
            dict(
                filename="ideal_gas",
                columns_map={
                    "gamma": "gamma",
                    "p": "P [Pa]",
                    "rho": "rho [Kg / m^3]",
                    "R": "R [J / (Kg K)]",
                    "drs": "rho0/rho*",
                    "prs": "P0/P*",
                    "ars": "a0/a*",
                    "trs": "T0/T*",
                    "T": "T [K]",
                    "a": "Sound Speed [m/s]",
                },
            ),
        ])
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)
        self.compute()

    @param.depends(
        "input_wanted", "input_value_1",
        "input_value_2", "input_value_3",
        "gamma",watch=True
    )
    def compute(self):
        info = []
        list_of_results = []

        for v1, v2, v3, g in product(
            self.input_value_1, self.input_value_2,  self.input_value_3,
            self.gamma
        ):
            if self.input_wanted == "p":
                parameters = dict(rho=v1, R=v2, T=v3)
            elif self.input_wanted == "rho":
                parameters = dict(p=v1, R=v2, T=v3)
            elif self.input_wanted == "R":
                parameters = dict(p=v1, rho=v2, T=v3)
            else:
                parameters = dict(p=v1, rho=v2, R=v3)

            try:
                results = self.solver(
                    self.input_wanted,
                    **parameters,
                    to_dict=True
                )
                results["gamma"] = g
                results["a"] = sound_speed(g, results["R"], results["T"])
                results.update(
                    sonic_condition(g, to_dict=True)
                )
            except ValueError as err:
                info.append(f"ValueError: {err}")
                results = parameters.copy().update({
                    "drs": np.nan,
                    "prs": np.nan,
                    "ars": np.nan,
                    "trs": np.nan,
                    "T": np.nan,
                    "a": np.nan,
                    "gamma": g
                })

            list_of_results.append(results)

        results = _combine(list_of_results)
        self.error_log = "\n".join(info)
        df = pd.DataFrame( results )
        df.rename(columns=self.tabulators[0].columns_map, inplace=True)
        df = df.reindex(columns=self.tabulators[0].columns_map.values())
        self.results = df

    @param.depends("results", watch=True, on_init=True)
    def update_dataframe(self):
        self.tabulators[0].results = self.results


class GasPage(BasePage, pn.viewable.Viewer):
    input_parameter_1 = param.ListSelector(
        label="Select input parameters:",
        objects={
            "γ: ratio of specific heats": "gamma",
            "R: mass-specific gas constant": "r",
            "Cp: specific heat at constant pressure": "cp",
            "Cv: specific heat at constant volume": "cv",
        },
        default=["gamma", "r"],
        doc="Parameters for the solver."
    )

    input_value_1 = param.String(
        default="1.4",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    input_value_2 = param.String(
        default="1.4",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    input_wanted = param.Selector(
        label="Select the parameter to compute:",
        objects={
            "Static pressure": "p",
            "Density": "rho",
            "Mass-specific gas constant, R": "r",
            "Temperature": "t",
        },
        default="t"
    )

    gamma = param.String(
        default="1.4",
        label="Ratio of specific heats, γ:",
        doc="""
            Comma separated list of ratio of specific heats.
            Each value must be greater than 1."""
    )

    input_temperature = param.String(
        default="298",
        label="Temperature:",
        doc="Comma separated list of values."
    )

    input_value_3 = param.String(
        default="1.4",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    input_value_4 = param.String(
        default="1.4",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    input_value_5 = param.String(
        default="1.4",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    def __init__(self, **params):
        params.setdefault("page_title", "Gas")
        params.setdefault("page_description", "Compute gas-related quantities.")
        params.setdefault("sections", [
            GasSection(
                wrap_in_card=True, theme=params.get("theme", "default")),
            IdealGasSection(
                wrap_in_card=True, theme=params.get("theme", "default")),
        ])
        super().__init__(**params)

    @param.depends("input_parameter_1", watch=True, on_init=True)
    def update_inputs_labels_gas_section(self):
        rev_dict = {
            v: k for k, v in self.param.input_parameter_1.objects.items()
        }
        default_values = {
            "gamma": "1.4",
            "r": "287.05",
            "cp": "1004.675",
            "cv": "717.625",
        }
        self.param.input_value_1.label = rev_dict[self.input_parameter_1[0]]
        self.param.input_value_2.label = rev_dict[self.input_parameter_1[1]]
        self.param.update({
            "input_value_1": default_values[self.input_parameter_1[0]],
            "input_value_2": default_values[self.input_parameter_1[1]],
        })

    @param.depends("input_wanted", watch=True, on_init=True)
    def update_inputs_labels_ideal_gas_section(self):
        rev_dict = {
            v: k for k, v in self.param.input_wanted.objects.items()
        }
        if self.input_wanted == "p":
            self.param.input_value_3.label = rev_dict["rho"]
            self.param.input_value_4.label = rev_dict["r"]
            self.param.input_value_5.label = rev_dict["t"]
            self.param.update({
                "input_value_3": "1.2259",
                "input_value_4": "287.05",
                "input_value_5": "288",
            })
        elif self.input_wanted == "rho":
            self.param.input_value_3.label = rev_dict["p"]
            self.param.input_value_4.label = rev_dict["r"]
            self.param.input_value_5.label = rev_dict["t"]
            self.param.update({
                "input_value_3": "101345.64336",
                "input_value_4": "287.05",
                "input_value_5": "288",
            })
        elif self.input_wanted == "r":
            self.param.input_value_3.label = rev_dict["p"]
            self.param.input_value_4.label = rev_dict["rho"]
            self.param.input_value_5.label = rev_dict["t"]
            self.param.update({
                "input_value_3": "101345.64336",
                "input_value_4": "1.2259",
                "input_value_5": "288",
            })
        else:
            self.param.input_value_3.label = rev_dict["p"]
            self.param.input_value_4.label = rev_dict["rho"]
            self.param.input_value_5.label = rev_dict["r"]
            self.param.update({
                "input_value_3": "101345.64336",
                "input_value_4": "1.2259",
                "input_value_5": "287.05",
            })

    @param.depends(
        "input_parameter_1", "input_value_1", "input_value_2",
        "input_temperature", "gamma",
        watch=True, on_init=True
    )
    def update_gas_section(self):
        self.sections[0].param.update({
            "input_parameter_1": self.input_parameter_1[0],
            "input_parameter_2": self.input_parameter_1[1],
            "input_value_1": _parse_input_string(self.input_value_1),
            "input_value_2": _parse_input_string(self.input_value_2),
            "input_temperature": _parse_input_string(self.input_temperature),
        })

    @param.depends(
        "input_wanted", "input_value_3", "input_value_4",
        "input_value_5", "gamma",
        watch=True, on_init=True
    )
    def update_ideal_gas_section(self):
        self.sections[1].param.update({
            "input_wanted": self.input_wanted,
            "input_value_1": _parse_input_string(self.input_value_3),
            "input_value_2": _parse_input_string(self.input_value_4),
            "input_value_3": _parse_input_string(self.input_value_5),
            "gamma": _parse_input_string(self.gamma),
        })

    def _create_sidebar_controls(self):
        tmp_panel = pn.Param(
            self,
            widgets={
                "input_parameter_1": {
                "type": pn.widgets.MultiChoice,
                "max_items": 2
            }},
        )
        self.controls = pn.Column(
            "For Gas section:",
            tmp_panel.widget("input_parameter_1"),
            self.param.input_value_1,
            self.param.input_value_2,
            self.param.input_temperature,
            pn.layout.Divider(stylesheets=[stylesheet]),
            "For Ideal Gas section:",
            self.param.input_wanted,
            self.param.input_value_3,
            self.param.input_value_4,
            self.param.input_value_5,
            self.param.gamma,
            pn.layout.Divider(stylesheets=[stylesheet]),
            "For both sections:",
            self.param.num_decimal_places
        )

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(
                self.param.page_description,
                styles={
                    "font-size": "1.2em"
                }
            ),
            *self.sections
        )
