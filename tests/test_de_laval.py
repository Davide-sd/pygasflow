import numpy as np
from pygasflow.nozzles import CD_Conical_Nozzle, CD_TOP_Nozzle
from pygasflow.solvers import De_Laval_Solver
import pytest


# compare the results against
# https://onlineflowcalculator.com/pages/CFLOW/calculator.html
# section "Nozzle Flow"
expected_flow_conditions = {
    "Chocked": np.array([
        9.37162502e-01, 3.05903834e-01, 2.82708980e+02, 9.49579906e+04,
        1.17033452e+00, 2.88000000e+02, 1.01325000e+05, 3.37034817e+02,
        1.03100243e+02, 2.00000000e+00, 2.50000000e-01]),
    "Shock at Exit": np.array([
        5.13400728e-01, 5.47431655e-01, 2.71714446e+02, 5.20203288e+04,
        6.67080840e-01, 2.88000000e+02, 6.37752617e+04, 3.30416199e+02,
        1.80880286e+02, 1.25882579e+00, 3.97195548e-01]),
    "Supercritical": np.array([
        9.39326457e-02, 2.19719812e+00, 1.46524924e+02, 9.51772533e+03,
        2.26328772e-01, 2.88000000e+02, 1.01325000e+05, 2.42639062e+02,
        5.33126091e+02, 2.00000000e+00, 2.50000000e-01])
}

expected_flow_states = {
    "Throat": np.array([
        1.00000000e+00, 2.40000000e+02, 5.35281521e+04, 7.77121837e-01,
        2.88000000e+02, 1.01325000e+05, 3.10535022e+02, 3.10535022e+02,
        1.00000000e+00, 2.50000000e-01]),
    "Upstream SW": np.array([
        2.04715730e+00, 1.56677514e+02, 1.20327878e+04, 2.67594812e-01,
        2.88000000e+02, 1.01325000e+05, 2.50904414e+02, 5.13640803e+02,
        2.00000000e+00, 2.50000000e-01]),
    "Downstream SW": np.array([
        5.69519843e-01, 2.70455404e+02, 5.68267877e+04, 1.04761128e+00,
        2.88000000e+02, 7.08095759e+04, 3.29649786e+02, 1.87742094e+02,
        1.39767236e+00, 3.57737632e-01]),
    "Exit": np.array([
        4.71861372e-01, 2.75721929e+02, 4.24857455e+04, 7.68271386e-01,
        2.88000000e+02, 7.08095759e+04, 3.32843914e+02, 1.57056186e+02,
        1.39767236e+00, 3.57737632e-01])
}

At = 0.25
Ae = 0.5
Rt = np.sqrt(At / np.pi)
Re = np.sqrt(Ae / np.pi)


class Test_De_Laval_Solver:

    def create_solver(self):
        n = CD_Conical_Nozzle(Rt=Rt, Re=Re)
        P0 = 101325.0
        Pb = 60795.0
        T0 = 288
        gamma = 1.4
        R = 287
        solver = De_Laval_Solver(
            geometry=n, P0=P0, Pb_P0_ratio=Pb/P0, T0=T0,
            gamma=gamma, R=R
        )
        return solver

    @pytest.fixture(scope="class")
    def solver(self):
        return self.create_solver()

    def test_limit_pressure_ratios(self, solver):
        assert np.allclose(
            solver.limit_pressure_ratios,
            [9.37162502e-01, 5.13400728e-01, 9.39326457e-02])

        assert np.allclose(
            [np.round(1 / l, 2) for l in solver.limit_pressure_ratios],
            [1.07, 1.95, 10.65])

    def test_flow_conditions(self, solver):
        assert np.allclose(
            solver.flow_conditions["Chocked"].values,
            expected_flow_conditions["Chocked"])

        assert np.allclose(
            solver.flow_conditions["Shock at Exit"].values,
            expected_flow_conditions["Shock at Exit"])

        assert np.allclose(
            solver.flow_conditions["Supercritical"].values,
            expected_flow_conditions["Supercritical"])

    def test_current_flow_condition(self, solver):
        assert solver.current_flow_condition == "Shock in Nozzle"

    def test_flow_states(self, solver):
        assert np.allclose(
            solver.flow_states["Throat"].values,
            expected_flow_states["Throat"])

        assert np.allclose(
            solver.flow_states["Upstream SW"].values,
            expected_flow_states["Upstream SW"])

        assert np.allclose(
            solver.flow_states["Downstream SW"].values,
            expected_flow_states["Downstream SW"])

        assert np.allclose(
            solver.flow_states["Exit"].values,
            expected_flow_states["Exit"])

    def test_mass_flow_rate(self, solver):
        assert np.isclose(solver.mass_flow_rate, 60.33088673111959)

    def test_update_back_to_reservoir_pressure_ratio(self, solver):
        assert np.isclose(solver.Pb_P0_ratio, 0.6)

        with pytest.warns(
            UserWarning,
            match="The provided area ratio is located beyond the exit section."
        ):
            solver.Pb_P0_ratio = 0.08

        assert solver.current_flow_condition == "Underexpanded Flow"
        assert np.allclose(
            solver.flow_conditions["Chocked"].values,
            expected_flow_conditions["Chocked"])
        assert np.allclose(
            solver.flow_conditions["Shock at Exit"].values,
            expected_flow_conditions["Shock at Exit"])
        assert np.allclose(
            solver.flow_conditions["Supercritical"].values,
            expected_flow_conditions["Supercritical"])
        assert "Upstream SW" not in solver.flow_states
        assert "Downstream SW" not in solver.flow_states
        assert np.allclose(
            solver.flow_states["Throat"].values,
            expected_flow_states["Throat"])
        assert not np.allclose(
            solver.flow_states["Exit"].values,
            expected_flow_states["Exit"])

    def test_update_gamma(self):
        solver = self.create_solver()
        assert np.isclose(solver.Pb_P0_ratio, 0.6)

        solver.gamma = 1.2
        assert solver.current_flow_condition == "Shock in Nozzle"
        assert not np.allclose(
            solver.flow_conditions["Chocked"].values,
            expected_flow_conditions["Chocked"])
        assert not np.allclose(
            solver.flow_conditions["Shock at Exit"].values,
            expected_flow_conditions["Shock at Exit"])
        assert not np.allclose(
            solver.flow_conditions["Supercritical"].values,
            expected_flow_conditions["Supercritical"])
        assert not np.allclose(
            solver.flow_states["Throat"].values,
            expected_flow_states["Throat"])
        assert not np.allclose(
            solver.flow_states["Upstream SW"].values,
            expected_flow_states["Upstream SW"])
        assert not np.allclose(
            solver.flow_states["Downstream SW"].values,
            expected_flow_states["Downstream SW"])
        assert not np.allclose(
            solver.flow_states["Exit"].values,
            expected_flow_states["Exit"])

    def test_update_total_pressure(self):
        solver = self.create_solver()
        assert np.isclose(solver.Pb_P0_ratio, 0.6)

        solver.P0 = 80000
        assert np.isclose(solver.mass_flow_rate, 47.633564653240235)
        assert solver.current_flow_condition == "Shock in Nozzle"

    def test_update_nozzle(self):
        solver = self.create_solver()
        solver.geometry.throat_radius = Rt * 1.2
        assert solver.current_flow_condition == "Overexpanded Flow"
        assert not np.allclose(
            solver.flow_conditions["Chocked"].values,
            expected_flow_conditions["Chocked"])
        assert not np.allclose(
            solver.flow_conditions["Shock at Exit"].values,
            expected_flow_conditions["Shock at Exit"])
        assert not np.allclose(
            solver.flow_conditions["Supercritical"].values,
            expected_flow_conditions["Supercritical"])
        assert not np.allclose(
            solver.flow_states["Throat"].values,
            expected_flow_states["Throat"])
        assert "Upstream SW" not in solver.flow_states
        assert "Downstream SW" not in solver.flow_states
        assert not np.allclose(
            solver.flow_states["Exit"].values,
            expected_flow_states["Exit"])

    def test_change_nozzle(self):
        # verify that it's possible to change the nozzle. Solver will
        # automatically compute new results
        solver = self.create_solver()
        assert isinstance(solver.geometry, CD_Conical_Nozzle)
        solver.geometry = CD_TOP_Nozzle(Ri=2*Rt, Rt=Rt, Re=3*Rt)
        assert isinstance(solver.geometry, CD_TOP_Nozzle)
        assert solver.current_flow_condition == "Shock in Nozzle"
        assert not np.allclose(
            solver.flow_conditions["Chocked"].values,
            expected_flow_conditions["Chocked"])
        assert not np.allclose(
            solver.flow_conditions["Shock at Exit"].values,
            expected_flow_conditions["Shock at Exit"])
        assert not np.allclose(
            solver.flow_conditions["Supercritical"].values,
            expected_flow_conditions["Supercritical"])
        assert np.allclose(
            solver.flow_states["Throat"].values,
            expected_flow_states["Throat"])
        assert not np.allclose(
            solver.flow_states["Upstream SW"].values,
            expected_flow_states["Upstream SW"])
        assert not np.allclose(
            solver.flow_states["Downstream SW"].values,
            expected_flow_states["Downstream SW"])
        assert not np.allclose(
            solver.flow_states["Exit"].values,
            expected_flow_states["Exit"])

