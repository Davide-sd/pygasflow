import panel as pn
import param
from pygasflow.interactive.diagrams import ObliqueShockDiagram
from pygasflow.interactive.pages.base import (
    ShockPage,
    ShockSection,
    _parse_input_string,
    stylesheet,
)
from pygasflow.shockwave import sonic_point_oblique_shock
from pygasflow.solvers import shockwave_solver


class ObliqueShockSection(ShockSection):
    """Section for displaying results about oblique shocks.
    """

    def __init__(self, **params):
        params.setdefault("solver", shockwave_solver)
        params.setdefault("title", "Oblique Shock Wave Section")
        params.setdefault("sonic_point_func", sonic_point_oblique_shock)
        params.setdefault("diagrams", [ObliqueShockDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="oblique_shock",
                columns_map={
                    "gamma": "gamma",
                    "mu": "Upstream Mach",
                    "mnu": "Upstream normal Mach",
                    "md": "Downstream Mach",
                    "mnd": "Downstream normal Mach",
                    "beta": "β [deg]",
                    "theta": "θ [deg]",
                    "Solution": "Solution",
                    "pr": "P2/P1",
                    "dr": "rho2/rho1",
                    "tr": "T2/T1",
                    "tpr": "P02/P01"
                },
                float_formatters_exclusion=["Solution"],
            ),
        ])
        params.setdefault("wrap_in_card", False)
        super().__init__(**params)

    def run_solver(self, v1, v2, gamma, flag):
        return self.solver(
            self.input_parameter_1, v1,
            self.input_parameter_2, v2,
            gamma=gamma, flag=flag, to_dict=True
        )


class ObliqueShockPage(ShockPage):
    input_parameter_1 = param.Selector(
        label="Select parameter:",
        objects={
            "Upstream Mach number": "mu",
            "Upstream normal Mach number": "mnu",
            "Downstream normal Mach number": "mnd",
            "Pressure Ratio P2/P1": "pressure",
            "Density Ratio rho2/rho1": "density",
            "Temperature Ratio T2/T1": "temperature",
            "Total Pressure Ratio P02/P01": "total_pressure",
            "Shock wave angle, β, [degrees]": "beta",
            "Deflection angle, θ, [degrees]": "theta"
        },
        doc="First input parameter to be used in the shockwave computation."
    )

    input_value_1 = param.String(
        default="2",
        label="Parameter values:",
        doc="Comma separated list of values."
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

    def __init__(self, **params):
        params.setdefault("page_title", "Oblique Shock")
        params.setdefault("page_description",
            "Change in properties of a 1D flow caused by an"
            " oblique shock wave. The streamlines experience some"
            " deflection angle, θ, at the shock.")
        params.setdefault("sections", [
            ObliqueShockSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("input_parameter_1", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter_1 in ["mu", "mnu"]:
            self.input_value_1 = "2"
        elif self.input_parameter_1 == "mnd":
            self.input_value_1 = "0.5773502691896257"
        elif self.input_parameter_1 == "beta":
            self.input_value_1 = "90"
        elif self.input_parameter_1 == "theta":
            self.input_value_1 = "0"
        elif self.input_parameter_1 == "pressure":
            self.input_value_1 = "4.5"
        elif self.input_parameter_1 ==  "density":
            self.input_value_1 = "2.666666666666667"
        elif self.input_parameter_1 == "temperature":
            self.input_value_1 = "1.6874999999999998"
        elif self.input_parameter_1 == "total_pressure":
            self.input_value_1 = "0.7208738614847455"

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
            self.param.input_parameter_1,
            self.param.input_value_1,
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
        "input_parameter_1", "input_value_1",
        "input_parameter_2", "input_value_2",
        "input_flag", "gamma",
        watch=True, on_init=True
    )
    def compute(self):
        value_1 = _parse_input_string(self.input_value_1)
        value_2 = _parse_input_string(self.input_value_2)
        gamma = _parse_input_string(self.gamma)

        # update all parameters simoultaneously
        new_params = dict(
            input_parameter_1=self.input_parameter_1,
            input_value_1=value_1,
            input_parameter_2=self.input_parameter_2,
            input_value_2=value_2,
            gamma=gamma,
            input_flag=self.input_flag
        )
        for s in self.sections:
            s.param.update(**new_params)
