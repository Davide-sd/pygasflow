import pygasflow
from pygasflow.solvers import (
    oblique_shockwave_solver,
    normal_shockwave_solver,
    conical_shockwave_solver,
    isentropic_solver,
    fanno_solver,
    rayleigh_solver,
    gas_solver,
    ideal_gas_solver
)
import pytest


def do_test(res, solver_to_dict, func_to_dict):
    if func_to_dict is True:
        assert isinstance(res, dict)
    elif func_to_dict is False:
        assert isinstance(res, (tuple, list))
    else: # func_to_dict=None
        if solver_to_dict:
            assert isinstance(res, dict)
        else:
            assert isinstance(res, (tuple, list))


@pytest.mark.parametrize("solver_to_dict, func_to_dict", [
    (True, None),
    (True, False),
    (True, True),
    # NOTE: it's very important to have solver_to_dict=False as the last
    # tests, because of they way pytest works. It loads the module only
    # once, so to get all other tests to pass it must have solver_to_dict=False
    (False, None),
    (False, False),
    (False, True),
])
def test_shockwave_solver_to_dict(solver_to_dict, func_to_dict):
    pygasflow.defaults.solver_to_dict = solver_to_dict
    do_test(
        oblique_shockwave_solver("mu", 5, "theta", 15, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        normal_shockwave_solver("mu", 5, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        conical_shockwave_solver(5, "theta_c", 15, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        isentropic_solver("m", 5, gamma=1.4, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        fanno_solver("m", 5, gamma=1.4, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        rayleigh_solver("m", 5, gamma=1.4, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        gas_solver("gamma", 1.4, "r", 287.05, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )
    do_test(
        ideal_gas_solver("p", R=287.05, T=288, rho=1.2259, to_dict=func_to_dict),
        solver_to_dict,
        func_to_dict
    )

