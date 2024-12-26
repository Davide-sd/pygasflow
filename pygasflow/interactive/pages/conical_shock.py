import panel as pn
import param
from pygasflow.interactive.diagrams import ConicalShockDiagram
from pygasflow.interactive.pages.base import (
    ShockPage,
    ShockSection,
    _parse_input_string,
    stylesheet,
)
from pygasflow.shockwave import sonic_point_conical_shock
from pygasflow.solvers import conical_shockwave_solver


class ConicalShockSection(ShockSection):
    """Section for displaying results about conical shocks.
    """

    def __init__(self, **params):
        params.setdefault("solver", conical_shockwave_solver)
        params.setdefault("title", "Conical Shock Wave Section")
        params.setdefault("sonic_point_func", sonic_point_conical_shock)
        params.setdefault("diagrams", [ConicalShockDiagram])
        params.setdefault("tabulators", [
            dict(
                filename="conical_shock",
                columns_map={
                    "gamma": "gamma",
                    "mu": "Upstream Mach",
                    "mc": "M_c",
                    "theta_c": "θ_c [deg]",
                    "beta": "β [deg]",
                    "delta": "δ [deg]",
                    "Solution": "Solution",
                    "pr": "P2/P1",
                    "dr": "rho2/rho1",
                    "tr": "T2/T1",
                    "tpr": "P02/P01",
                    "pc_pu": "P_c/P1",
                    "rhoc_rhou": "rho_c/rho_1",
                    "Tc_Tu": "T_c/T1"
                },
                float_formatters_exclusion=["Solution"]
            ),
        ])
        params.setdefault("wrap_in_card", False)
        params.setdefault("input_parameter_2", "theta_c")
        super().__init__(**params)

    def run_solver(self, v1, v2, gamma, flag):
        res = self.solver(
            v1,
            self.input_parameter_2, v2,
            gamma=gamma, flag=flag, to_dict=True
        )
        return res


class ConicalShockPage(ShockPage):
    input_value_1 = param.String(
        default="2",
        label="Upstream Mach Number:",
        doc="Comma separated list of values."
    )

    input_parameter_2 = param.Selector(
        label="Select parameter:",
        objects={
            "Mach number at the cone's surface": "mc",
            "Half cone angle, θ_c, [degrees]": "theta_c",
            "Shock wave angle, β, [degrees]": "beta"
        },
        default="theta_c",
        doc="Second input parameter to be used in the shockwave computation."
    )

    input_value_2 = param.String(
        default="15",
        label="Parameter values:",
        doc="Comma separated list of values."
    )

    def __init__(self, **params):
        params.setdefault("page_title", "Conical Shock")
        params.setdefault("page_description",
            "Change in properties of an axisymmetric supersonic flow"
            " over a sharp cone at zero angle of attack to the free stream.")
        params.setdefault("sections", [
            ConicalShockSection(theme=params.get("theme", "default"))
        ])
        super().__init__(**params)

    @param.depends("input_parameter_2", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter_2 == "mc":
            self.input_value_2 = "1.7068679592453362"
        elif self.input_parameter_2 == "theta_c":
            self.input_value_2 = "15"
        elif self.input_parameter_2 == "beta":
            self.input_value_2 = "33.91469752764406"

    def _create_sidebar_controls(self):
        self.controls = pn.Column(
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
        "input_value_1",
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
            input_value_1=value_1,
            input_parameter_2=self.input_parameter_2,
            input_value_2=value_2,
            gamma=gamma,
            input_flag=self.input_flag
        )
        for s in self.sections:
            s.param.update(**new_params)
