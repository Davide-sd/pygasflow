import param
import panel as pn
from pygasflow.solvers import rayleigh_solver
from pygasflow.interactive.diagrams import RayleighDiagram
from pygasflow.interactive.pages.base import Common


class RayleighPage(Common, pn.viewable.Viewer):
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

    def __init__(self, **params):
        params.setdefault("_filename", "rayleigh")
        params.setdefault("_solver", rayleigh_solver)
        params.setdefault("_diagram", RayleighDiagram)
        params.setdefault("page_title", "Rayleigh")
        super().__init__(**params)

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
