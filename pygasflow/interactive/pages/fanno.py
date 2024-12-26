import param
from pygasflow.solvers import fanno_solver
from pygasflow.interactive.diagrams import FannoDiagram
from pygasflow.interactive.pages.base import (
    FlowPage,
    FlowSection,
    _parse_input_string
)


class FannoSection(FlowSection):
    def __init__(self, **params):
        params.setdefault("solver", fanno_solver)
        params.setdefault("title", "Fanno Section")
        params.setdefault("diagrams", [FannoDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="fanno",
                columns_map={
                    "gamma": "gamma",
                    "m": "Mach",
                    "prs": "P/P*",
                    "drs": "rho/rho*",
                    "trs": "T/T*",
                    "tprs": "P0/P0*",
                    "urs": "U/U*",
                    "fps": "4fL*/D",
                    "eps": "(s*-s)/R"
                },
            ),
        ])
        params.setdefault("internal_map", {
            "total_pressure_sub": "tprs",
            "total_pressure_super": "tprs",
            "friction_sub": "fps",
            "friction_super": "fps",
            "entropy_sub": "eps",
            "entropy_super": "eps"
        })
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)
        self.compute()


class FannoPage(FlowPage):
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
            "Entropy parameter (s*-s)/R (supersonic case)": "entropy_super"
        },
        doc="The input parameter to be used in the fanno computation."
    )

    def __init__(self, **params):
        params.pop("_theme", "")
        params.setdefault("page_title", "Fanno")
        params.setdefault("page_description", "1D flow with friction.")
        params.setdefault("sections", [
            FannoSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m":
            self.input_value = "2"
        elif self.input_parameter == "pressure":
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
        # update all parameters simoultaneously
        new_params = dict(
            gamma=_parse_input_string(self.gamma),
            input_parameter=self.input_parameter,
            input_value=_parse_input_string(self.input_value)
        )
        self.sections[0].param.update(**new_params)
