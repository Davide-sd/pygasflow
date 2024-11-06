import param
import panel as pn
from pygasflow.interactive.diagrams import NormalShockDiagram
from pygasflow.interactive.pages.base import Common, stylesheet


class NormalShockPage(Common, pn.viewable.Viewer):
    input_parameter = param.Selector(
        label="Select parameter:",
        objects={
            "Upstream Mach number": "m1",
            "Downstream Mach number": "m2",
            "Pressure Ratio P2/P1": "pressure",
            "Density Ratio rho2/rho1": "density",
            "Temperature Ratio T2/T1": "temperature",
            "Total Pressure Ratio P02/P01": "total_pressure"
        },
        doc="Fisrt input parameter to be used in the shockwave computation."
    )

    _columns = {
        "gamma": "gamma",
        "m1": "Upstream Mach",
        "m2": "Downstream Mach",
        "pr": "P2/P1",
        "dr": "rho2/rho1",
        "tr": "T2/T1",
        "tpr": "P02/P01"
    }

    _float_formatters_exclusion = ["Solution"]

    _internal_map = {
        "m1": "m1",
        "m2": "m2",
        "pressure": "pr",
        "density": "dr",
        "temperature": "tr",
        "total_pressure": "tpr",
    }

    def __init__(self, **params):
        params.setdefault("_filename", "normal_shockwave")
        params.setdefault("_diagram", NormalShockDiagram)
        params.setdefault("page_title", "Normal Shock")
        super().__init__(**params)

    @param.depends("input_parameter", watch=True, on_init=True)
    def _validate_input_value(self):
        # set appropriate default values so that no errors are raised
        # when user changes `input_parameter`
        if self.input_parameter == "m1":
            self.input_value = "2"
        if self.input_parameter == "m2":
            self.input_value = "0.5773502691896257"
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
            self.param.gamma,
            pn.layout.Divider(stylesheets=[stylesheet]),
            self.param.num_decimal_places
        )
