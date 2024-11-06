import param
import panel as pn
from pygasflow.solvers import fanno_solver
from pygasflow.interactive.diagrams import FannoDiagram
from pygasflow.interactive.pages.base import Common


class FannoPage(Common, pn.viewable.Viewer):
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

    def __init__(self, **params):
        params.setdefault("_filename", "fanno")
        params.setdefault("_solver", fanno_solver)
        params.setdefault("_diagram", FannoDiagram)
        params.setdefault("page_title", "Fanno")
        params.setdefault("page_description", "1D flow with friction.")
        super().__init__(**params)

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
