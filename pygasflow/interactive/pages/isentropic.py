import param
from pygasflow.solvers import isentropic_solver
from pygasflow.interactive.diagrams import IsentropicDiagram
from pygasflow.interactive.pages.base import (
    FlowPage,
    FlowSection,
    _parse_input_string
)


class IsentropicSection(FlowSection):
    def __init__(self, **params):
        params.setdefault("solver", isentropic_solver)
        params.setdefault("title", "Isentropic Section")
        params.setdefault("diagrams", [IsentropicDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="isentropic",
                columns_map={
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
                },
            ),
        ])
        params.setdefault("internal_map", {
            "crit_area_sub": "ars",
            "crit_area_super": "ars",
        })
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)
        self.compute()


class IsentropicPage(FlowPage):
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

    def __init__(self, **params):
        params.pop("_theme", "")
        params.setdefault("page_title", "Isentropic")
        params.setdefault("page_description",
            "Adiabatic and reversible 1D flow.")
        params.setdefault("sections", [
            IsentropicSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m":
            self.input_value = "2"
        elif self.input_parameter == "pressure":
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
        # update all parameters simoultaneously
        new_params = dict(
            gamma=_parse_input_string(self.gamma),
            input_parameter=self.input_parameter,
            input_value=_parse_input_string(self.input_value)
        )
        self.sections[0].param.update(**new_params)
