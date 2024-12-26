import param
from pygasflow.solvers import normal_shockwave_solver
from pygasflow.interactive.diagrams import NormalShockDiagram
from pygasflow.interactive.pages.base import (
    FlowPage,
    FlowSection,
    _parse_input_string
)


class NormalShockSection(FlowSection):
    def __init__(self, **params):
        params.setdefault("solver", normal_shockwave_solver)
        params.setdefault("title", "Normal Shock Section")
        params.setdefault("diagrams", [NormalShockDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="normal_shock",
                columns_map={
                    "gamma": "gamma",
                    "mu": "Upstream Mach",
                    "md": "Downstream Mach",
                    "pr": "P2/P1",
                    "dr": "rho2/rho1",
                    "tr": "T2/T1",
                    "tpr": "P02/P01"
                },
            ),
        ])
        params.setdefault("wrap_in_card", False)
        params.setdefault("input_parameter", "mu")
        super().__init__(**params)
        self.compute()


class NormalShockPage(FlowPage):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Upstream Mach number": "mu",
            "Downstream Mach number": "md",
            "Pressure Ratio P2/P1": "pressure",
            "Density Ratio rho2/rho1": "density",
            "Temperature Ratio T2/T1": "temperature",
            "Total Pressure Ratio P02/P01": "total_pressure"
        },
        doc="First input parameter to be used in the shockwave computation."
    )

    def __init__(self, **params):
        params.pop("_theme", "")
        params.setdefault("page_title", "Normal Shock")
        params.setdefault("page_description",
            "Change in properties caused by a shock wave perpendicular"
            " to a 1D flow.")
        params.setdefault("sections", [
            NormalShockSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "mu":
            self.input_value = "2"
        elif self.input_parameter == "md":
            self.input_value = "0.5773502691896257"
        elif self.input_parameter == "pressure":
            self.input_value = "4.5"
        elif self.input_parameter ==  "density":
            self.input_value = "2.666666666666667"
        elif self.input_parameter == "temperature":
            self.input_value = "1.6874999999999998"
        elif self.input_parameter == "total_pressure":
            self.input_value = "0.7208738614847455"

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

