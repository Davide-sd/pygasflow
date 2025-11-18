import param
from pygasflow._version import __version__
from pygasflow.solvers import (
    isentropic_solver,
    isentropic_compression,
    fanno_solver,
    rayleigh_solver,
    specific_heat_solver,
    shockwave_solver,
    normal_shockwave_solver,
    oblique_shockwave_solver,
    conical_shockwave_solver,
    shock_compression,
    gas_solver,
    ideal_gas_solver,
    sonic_condition,
    De_Laval_Solver
)
import warnings

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
from pygasflow.common import (
    pressure_coefficient,
    sound_speed,
    speed_of_sound,
)
from pygasflow.utils.common import canonicalize_pint_dimensions


class _defaults(param.Parameterized):
    """Default options for the module.
    """
    solver_to_dict = param.Boolean(False, doc="""
        Global setting to control what kind of output to expect from
        a solver. If False, it returns a tuple of results.
        If True, it returns a dictionary of results.""")

    print_number_formatter = param.String("{:>15.8f}",
        doc="Formatter to be used with number in printing functions.")

    pint_ureg = param.Parameter(default=None, doc="""
        Pint's ``UnitRegistry`` instance to be used by this module.""")

    @param.depends("pint_ureg", watch=True, on_init=True)
    def _validate_pint_ureg(self):
        try:
            import pint
            if (
                (self.pint_ureg is not None)
                and (not isinstance(self.pint_ureg, pint.UnitRegistry))
            ):
                raise TypeError(
                    "`pint_ureg` must be an instance of `pint.UnitRegistry`."
                    f" Instead, pint_ureg='{self.pint_ureg}' was received."
                )
        except ImportError:
            warnings.warn(
                "`pint` is not installed, so the default value for"
                " `pint_ureg` will be set to `None`.",
                stacklevel=0
            )
            with param.discard_events(self):
                self.pint_ureg = None


defaults = _defaults()


__all__ = [
    "ise",
    "fan",
    "ray",
    "ss",
    "css",
    "isentropic_solver",
    "isentropic_compression",
    "fanno_solver",
    "rayleigh_solver",
    "specific_heat_solver",
    "shockwave_solver",
    "oblique_shockwave_solver",
    "normal_shockwave_solver",
    "conical_shockwave_solver",
    "shock_compression",
    "De_Laval_Solver",
    "CD_Conical_Nozzle",
    "CD_Min_Length_Nozzle",
    "CD_TOP_Nozzle",
    "Rao_Parabola_Angles",
    "pressure_coefficient",
    "sound_speed",
    "speed_of_sound",
    "gas_solver",
    "ideal_gas_solver",
    "sonic_condition",
    "canonicalize_pint_dimensions",
]
