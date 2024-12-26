import param
from pygasflow._version import __version__
from pygasflow.solvers import (
    isentropic_solver,
    fanno_solver,
    rayleigh_solver,
    shockwave_solver,
    normal_shockwave_solver,
    oblique_shockwave_solver,
    conical_shockwave_solver,
    gas_solver,
    ideal_gas_solver,
    sonic_condition,
    De_Laval_Solver
)

ise = isentropic_solver
fan = fanno_solver
ray = rayleigh_solver
ss = shockwave_solver
css = conical_shockwave_solver

from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_Min_Length_Nozzle,
    CD_TOP_Nozzle,
    Rao_Parabola_Angles
)

import pygasflow.atd
from pygasflow.common import pressure_coefficient, sound_speed

class defaults(param.Parameterized):
    """Default options for the module.
    """
    solver_to_dict = param.Boolean(False, doc="""
        Global setting to control what kind of output to expect from
        a solver. If False, it returns a tuple of results.
        If True, it returns a dictionary of results.""")

    print_number_formatter = param.String("{:>15.8f}",
        doc="Formatter to be used with number in printing functions.")


__all__ = [
    "ise",
    "fan",
    "ray",
    "ss",
    "css",
    "isentropic_solver",
    "fanno_solver",
    "rayleigh_solver",
    "shockwave_solver",
    "oblique_shockwave_solver",
    "normal_shockwave_solver",
    "conical_shockwave_solver",
    "De_Laval_Solver",
    "CD_Conical_Nozzle",
    "CD_Min_Length_Nozzle",
    "CD_TOP_Nozzle",
    "Rao_Parabola_Angles",
    "pressure_coefficient",
    "sound_speed",
    "gas_solver",
    "ideal_gas_solver",
    "sonic_condition",
]
