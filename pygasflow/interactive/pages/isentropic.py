import param
import panel as pn
from pygasflow.solvers import isentropic_solver
from pygasflow.interactive.diagrams import IsentropicDiagram
from pygasflow.interactive.pages.base import Common


class IsentropicPage(Common, pn.viewable.Viewer):
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

    def __init__(self, **params):
        params.setdefault("_filename", "isentropic")
        params.setdefault("_solver", isentropic_solver)
        params.setdefault("_diagram", IsentropicDiagram)
        params.setdefault("page_title", "Isentropic")
        super().__init__(**params)

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
