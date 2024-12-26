import numpy as np
import pandas as pd
import param
from numbers import Number
from scipy.optimize import bisect

import pygasflow.shockwave as shockwave
from pygasflow.isentropic import (
    temperature_ratio,
    pressure_ratio,
    density_ratio,
    m_from_critical_area_ratio,
    m_from_critical_area_ratio_and_pressure_ratio,
    critical_area_ratio,
)
from pygasflow.generic import sound_speed
from pygasflow.nozzles.nozzle_geometry import Nozzle_Geometry
from pygasflow.nozzles import CD_Conical_Nozzle


def _create_section(gamma, R, M, T0, P0, rho0, tr, pr, dr, area_ratio, As):
    T = tr * T0
    a = sound_speed(gamma, R, T)
    return [
        M,
        T,
        pr * P0,
        dr * rho0,
        T0,
        P0,
        a,
        a * M,
        area_ratio,
        As
    ]


def nozzle_mass_flow_rate(gamma, R, T0, P0, As):
    """Compute the mass flow rate through the nozzle.

    Parameters
    ----------
    gamma : float
    R : float
    T0 : float
    P0 : float
    As : float
        Critical area (throat area).

    Return
    ------
    m_dot : float
    """
    a = P0 * As / np.sqrt(T0)
    e = (gamma + 1) / (gamma - 1)
    b = np.sqrt(gamma / R * (2 / (gamma + 1))**e)
    m_dot = a * b
    return m_dot


class De_Laval_Solver(param.Parameterized):
    """
    Solve a De Laval Nozzle (Convergent-Divergent nozzle), starting from
    stagnation conditions and the nozzle type.

    Notes
    -----

    This is a reactive component. Once a ``De_Laval_Solver`` solver has been
    created, by updating one of its parameters everything else will be
    automatically recomputed.

    Examples
    --------

    Visualize all the quantities along the length of a conical nozzle using air
    and a back-to-stagnation pressure ratio of 0.1.

    >>> from pygasflow.nozzles import CD_Conical_Nozzle
    >>> from pygasflow.solvers import De_Laval_Solver
    >>> Ri, Rt, Re = 0.4, 0.2, 0.8  # Inlet Radius, Throat Radius, Exit Radius
    >>> nozzle = CD_Conical_Nozzle(Ri, Re, Rt)
    >>> solver = De_Laval_Solver(
    ...     gamma=1.4, R=287.05, T0=298, P0=101325,
    ...     nozzle=nozzle, Pb_P0_ratio=0.1)
    >>> solver.mass_flow_rate
    np.float64(29.809853965846262)
    >>> solver.current_flow_condition
    'Shock in Nozzle'

    Show ratios at current Pb/P0 value, at known states:

    >>> solver.flow_states
                              Throat    Upstream SW  Downstream SW          Exit
    Mach number             1.000000       4.286437       0.427968      0.357162
    Temperature           248.333333      63.747285     287.469587    290.586275
    Pressure            53528.152140     458.749502    9757.205237   1106.651037
    Density                 0.750913       0.025070       1.082635      0.121474
    Total Temperature     298.000000     298.000000     298.000000    298.000000
    Total Pressure     101325.000000  101325.000000   11066.510374  11066.510374
    Speed of sound        315.907766     160.056619     339.890281    341.727825
    Flow velocity         315.907766     686.072659     145.462314    122.052320
    A / A*                  1.000000      16.000000       1.747487      1.747487
    A*                      0.125664       0.125664       1.150577      1.150577

    Show the possible flow conditions, depending on the back-to-stagnation
    pressure ration:

    >>> solver.flow_condition_summary
                                                             Condition
    No flow                                                Pb / P0 = 1
    Subsonic                              Pb / P0 > 0.9990833631058978
    Chocked                               Pb / P0 = 0.9990833631058978
    Shock in Nozzle  0.08373983107077117 < Pb / P0 < 0.999083363105...
    Shock at Exit                        Pb / P0 = 0.08373983107077117
    Overexpanded     0.003635619912775599 < Pb / P0 < 0.08373983107...
    Supercritical                       Pb / P0 = 0.003635619912775599
    Underexpanded                       Pb / P0 < 0.003635619912775599

    Show ratios at known flow conditions:

    >>> solver.flow_conditions
                             Chocked  Shock at Exit  Supercritical
    Pb / P0                 0.999083       0.083740       0.003636
    Mach number             0.036197       0.424347       4.459324
    Temperature           297.921929     287.640869      59.874057
    Pressure           101232.121767    8484.938383     368.379188
    Density                 1.183745       0.102764       0.021434
    Total Temperature     298.000000     298.000000     298.000000
    Total Pressure     101325.000000    9603.477943  101325.000000
    Speed of sound        346.014285     339.991524     155.117978
    Flow velocity          12.524826     144.274458     691.721305
    A / A*                 16.000000       1.516463      16.000000
    A*                      0.125664       1.325861       0.125664

    Get the location of the shockwave along the divergent (loc, radius):

    >>> solver.nozzle.shockwave_location
    (np.float64(2.038730013513229), np.float64(0.7427484426649532))

    Visualize the results at the current Pb/P0 value:

    >>> solver.plot(interactive=False)              #doctest: +SKIP

    Visualize an interactive application:

    .. panel-screenshot::

        from pygasflow.nozzles import CD_Conical_Nozzle
        from pygasflow.solvers import De_Laval_Solver
        Ri, Rt, Re = 0.4, 0.2, 0.8  # Inlet Radius, Throat Radius, Exit Radius
        nozzle = CD_Conical_Nozzle(Ri, Re, Rt)
        solver = De_Laval_Solver(
            gamma=1.4, R=287.05, T0=298, P0=101325,
            nozzle=nozzle, Pb_P0_ratio=0.25)
        solver.plot()

    """

    # editable parameters
    R = param.Number(287.05, bounds=(0, 5000),
        step=0.05,
        label="Specific gas constant, R, [J / (Kg K)]",
        doc="Specific gas constant, R, [J / (Kg K)]")
    gamma = param.Number(1.4, bounds=(1, 2),
        inclusive_bounds=(False, True),
        step=0.01,
        label="Ratio of specific heats, γ",
        doc="Ratio of specific heats, γ")
    T0 = param.Number(303.15, bounds=(0, 4000), step=0.05,
        doc="Total temperature, T0 [K]",
        label="Total temperature, T0 [K]")
    P0 = param.Number(8*101325, bounds=(0, 300*101325),
        label="Total pressure, P0 [Pa]",
        doc="Total pressure, P0 [Pa]")
    nozzle = param.ClassSelector(
        class_=Nozzle_Geometry,
        doc="The convergent-divergent nozzle type.")
    Pb_P0_ratio = param.Number(
        default=0.1, bounds=(0, 1), step=0.001,
        label="Pb / P0",
        doc="Back to Stagnation pressure ratio, Pb / P0.")

    # read-only parameters
    rho0 = param.Number(constant=True, doc="Upstream density.")
    critical_temperature = param.Number(constant=True)
    critical_pressure = param.Number(constant=True)
    critical_density = param.Number(constant=True)
    critical_velocity = param.Number(constant=True)
    limit_pressure_ratios = param.List(
        [0, 0, 0], item_type=Number, bounds=(3, 3), constant=True, doc="""
        List of 3 pressure ratios, [r1, r2, r3], where:

        * r1: Exit pressure ratio corresponding to fully subsonic flow in the
          divergent (when M=1 in A*).
        * r2: Exit pressure ratio corresponding to a normal shock wave at the
          exit section of the divergent.
        * r3: Exit pressure ratio corresponding to fully supersonic flow in
          the divergent and Pe = Pb (pressure at back). Design condition.""")
    current_flow_condition = param.String(
        default="",
        constant=True,
        doc="""
            Store the current flow condition (fully subsonic, fully supersonic,
            shock at exit, chocked, etc.)""")
    flow_conditions = param.DataFrame(
        constant=True,
        doc="""
            Store the results of the flow at the exit section for well known
            conditions, like choked, shock at exit and supercritical.""")
    flow_condition_summary = param.DataFrame(
        constant=True,
        doc="""
            A table summarizing the different flow conditions the nozzle can
            have based on the nozzle pressure ratio (back-to-stagnation
            pressure ratio).""")
    flow_results = param.List(item_type=np.ndarray, constant=True,
        doc="Store the results of the flow analysis.")
    flow_states = param.DataFrame(
        constant=True,
        doc="""
            Store the state of the flow at interesting nozzle locations,
            like in the throat, exit section, upstream and downstream of a
            shockwave.""")
    flow_states_labels = param.List([
            "Mach number", "Temperature", "Pressure", "Density",
            "Total Temperature", "Total Pressure",
            "Speed of sound", "Flow velocity", "A / A*", "A*"
        ],
        constant=True,
        doc="Labels for each one of the values of a `flow_states` entry.")
    mass_flow_rate = param.Number(0, constant=True,
        doc="Mass flow rate through the nozzle.")

    # for interactive applications only
    error_log = param.String("", doc="""
        Visualize on the interactive application any error that raises
        from the computation.""")
    is_interactive_app = param.Boolean(False, doc="""
        If True, exceptions are going to be intercepted and shown on
        error_log, otherwise fall back to the standard behaviour.""")
    _num_decimal_places = param.Integer(doc="""
        Set the number of decimal places to be shown in the
        `flow_condition_summary` dataframe. While other dataframes can
        be formatted from the Tabulators inside the interactive
        applications, this dataframe cannot because it contains
        string-formatted numbers. They must be formatted inside this class.""")

    def __init__(self, **params):
        params.setdefault(
            "nozzle",
            CD_Conical_Nozzle(Rj=0, R0=0,
                is_interactive_app=params.get("is_interactive_app", False))
        )
        super().__init__(**params)

    def _set_upstream_density(self):
        from pygasflow.solvers import ideal_gas_solver
        rho0 = ideal_gas_solver(
            "rho", p=self.P0, T=self.T0, R=self.R, to_dict=True)["rho"]
        with param.edit_constant(self):
            self.rho0 = rho0
            self.input_state = {
                "P0": self.P0,
                "T0": self.T0,
                "rho": rho0,
            }

    @param.depends(
        "T0", "P0", "R", "gamma", "Pb_P0_ratio",
        "nozzle",
        "nozzle.inlet_radius",
        "nozzle.outlet_radius",
        "nozzle.throat_radius",
        "nozzle.junction_radius_j",
        "nozzle.junction_radius_0",
        "nozzle.theta_c",
        "nozzle.theta_N",
        "nozzle.theta_e",
        "nozzle.fractional_length",
        "nozzle.N",
        watch=True, on_init=True
    )
    def update(self):
        try:
            self._update_logic()
            self.error_log = ""
        except ValueError as err:
            self.error_log = f"ValueError: {err}"
            if not self.is_interactive_app:
                raise ValueError(f"{err}")

    def _update_logic(self):
        self._set_upstream_density()
        T0, P0, rho0 = self.T0, self.P0, self.rho0
        gamma, R = self.gamma, self.R
        Ae = self.nozzle.outlet_area
        At = self.nozzle.throat_area

        # STEP 1: compute limit pressure ratios that will help determine
        # the kind of flow in the nozzle.

        ct = temperature_ratio(1, gamma) * T0
        cp = pressure_ratio(1, gamma) * P0
        cr = density_ratio(1, gamma) * rho0
        cv = sound_speed(ct, R, gamma)

        Ae_As_ratio = Ae / At
        M2_sub = m_from_critical_area_ratio(Ae_As_ratio, "sub", gamma)
        M2_sup = m_from_critical_area_ratio(Ae_As_ratio, "super", gamma)

        # exit pressure ratio corresponding to fully subsonic flow in the
        # divergent (when M=1 in A*)
        r1 = pressure_ratio(M2_sub, gamma)
        # exit pressure ratio corresponding to fully supersonic flow in the
        # divergent and Pe = Pb (pressure at back). Design condition.
        r3 = pressure_ratio(M2_sup, gamma)

        # isentropic pressure at the exit section of the divergent
        p_exit = r3 * P0

        # pressure downstream of the normal shock wave at the exit section of
        # the divergent
        p2 = shockwave.pressure_ratio(M2_sup, gamma) * p_exit
        # exit pressure ratio corresponding to a normal shock wave at the
        # exit section of the divergent
        r2 = p2 / P0

        states = {
            "Throat": _create_section(
                gamma, R, 1, T0, P0, rho0, ct / T0, cp / P0, cr / rho0, 1, At)
        }

        # STEP 2: analyze the flow in the nozzle, compute the important
        # quantities and find if and where a shockwave is present

        Lc = self.nozzle.length_convergent

        # copy the arrays: we do not want to modify the original nozzle in
        # case of shock wave at the exit plane
        area_ratios = np.copy(self.nozzle.area_ratio_array)
        L = np.copy(self.nozzle.length_array) + Lc

        M = np.zeros_like(area_ratios)
        P_ratios = np.zeros_like(area_ratios)
        rho_ratios = np.zeros_like(area_ratios)
        T_ratios = np.zeros_like(area_ratios)
        Asw_As_ratio = None
        Pe_P0_ratio = self.Pb_P0_ratio

        if np.isclose(Pe_P0_ratio, 1):  # no flow
            P_ratios += 1
            T_ratios += 1
            rho_ratios += 1
            # force the nozzle to set to None the location of the SW
            self.nozzle.location_divergent_from_area_ratio(Ae / At + 1)

        # fully subsonic flow
        elif Pe_P0_ratio >= r1:
            M = m_from_critical_area_ratio(area_ratios, "sub", self.gamma)
            P_ratios = pressure_ratio(M, self.gamma)
            rho_ratios = density_ratio(M, self.gamma)
            T_ratios = temperature_ratio(M, self.gamma)
            # force the nozzle to set to None the location of the SW
            self.nozzle.location_divergent_from_area_ratio(Ae / At + 1)

        # fully supersonic flow in the divergent
        elif Pe_P0_ratio < r2:
            M[L < Lc] = m_from_critical_area_ratio(
                area_ratios[L < Lc], "sub", self.gamma)
            M[L == Lc] = 1
            M[L > Lc] = m_from_critical_area_ratio(
                area_ratios[L > Lc], "super", self.gamma)
            P_ratios = pressure_ratio(M, self.gamma)
            rho_ratios = density_ratio(M, self.gamma)
            T_ratios = temperature_ratio(M, self.gamma)
            # force the nozzle to set to None the location of the SW
            self.nozzle.location_divergent_from_area_ratio(Ae / At + 1)

        # shock wave at the exit plane
        elif np.isclose(Pe_P0_ratio, r2):
            # upstream area ratio where the shockwave happens
            Ae_At_ratio = Ae / At
            self.nozzle.location_divergent_from_area_ratio(Ae_At_ratio)
            # Supersonic Mach number at the exit section just upstream of
            # the shock wave
            Meup_sw = m_from_critical_area_ratio(
                Ae_At_ratio, "super", self.gamma)
            # Subsonic Mach number at the exit section just downstream of
            # the shock wave
            Medw_sw = shockwave.mach_downstream(Meup_sw, self.gamma)
            # downstream of the shock wave there is a new isentropic critical
            # area ratio
            Ae_A2s_ratio = critical_area_ratio(Medw_sw, self.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.total_pressure_ratio(Meup_sw)

            M[L < Lc] = m_from_critical_area_ratio(
                area_ratios[L < Lc], "sub", self.gamma)
            M[L == Lc] = 1
            M[L > Lc] = m_from_critical_area_ratio(
                area_ratios[L > Lc], "super", self.gamma)

            # append the last subsonic point at the exit
            M = np.append(M, m_from_critical_area_ratio(
                Ae_A2s_ratio, "sub", self.gamma))
            P_ratios = pressure_ratio(M, self.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been
            # returned P/P02. Need to compute the ratio P/P0.
            P_ratios[-1] *= P02_P0_ratio

            rho_ratios = density_ratio(M, self.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the
            # shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[-1] *= P02_P0_ratio

            T_ratios = temperature_ratio(M, self.gamma)
            L = np.append(L, L[-1])
            area_ratios = np.append(area_ratios, Ae_At_ratio)

            states["Upstream SW"] = _create_section(
                gamma, R, M[-2], T0, P0, rho0, T_ratios[-2], P_ratios[-2],
                rho_ratios[-2], Ae_At_ratio, At)
            sec = _create_section(
                gamma, R, M[-1], T0, P02_P0_ratio * P0, rho0,
                T_ratios[-1], P_ratios[-1],
                rho_ratios[-1], Ae_A2s_ratio, At)
            states["Downstream SW"] = sec
            states["Exit"] = sec

        # shock into the divergent
        else:
            Ae_At_ratio = Ae / At
            # area ratio of the shock wave
            Asw_As_ratio = find_shockwave_area_ratio(
                Ae_At_ratio, Pe_P0_ratio, self.R, self.gamma)
            self.nozzle.location_divergent_from_area_ratio(Asw_As_ratio)
            # Mach number at the exit section given the exit pressure ratio
            # Pe_P0_ratio
            Me = m_from_critical_area_ratio_and_pressure_ratio(
                Ae_At_ratio, Pe_P0_ratio, self.gamma)
            # downstream of the shock wave there is a new isentropic critical
            # area ratio
            Ae_A2s_ratio = critical_area_ratio(Me, self.gamma)
            # critical area downstream of the shock wave
            A2s = Ae / Ae_A2s_ratio
            # Mach number just upstream of the shock wave
            Mup_sw = m_from_critical_area_ratio(
                Asw_As_ratio, "super", self.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.total_pressure_ratio(Mup_sw)

            # NOTE: in order to display a good jump while keeping the number
            # of discretization points relatively low, here I insert two points
            # at the location of the shockwave.
            sw_loc_in_div = self.nozzle.shockwave_location[0]
            if sw_loc_in_div not in L:
                idx = np.where(L < Lc + sw_loc_in_div)[0][-1]
                L = np.concatenate(
                    [L[:idx+1], [Lc + sw_loc_in_div] * 2, L[idx+1:]])

                # slightly bigger area ratio, to allow selective indexing below
                Asw_As_ratio_plus_eps = (Asw_As_ratio
                    + abs(area_ratios[idx+1] - Asw_As_ratio) / 1000)
                area_ratios = np.concatenate(
                    [area_ratios[:idx+1],
                    [Asw_As_ratio, Asw_As_ratio_plus_eps],
                    area_ratios[idx+1:]])
                M = np.zeros_like(L)

            # find indeces before and after the shock wave in the divergent
            idx_before_sw = np.bitwise_and(L > Lc, area_ratios <= Asw_As_ratio)
            idx_after_sw = np.bitwise_and(L > Lc, area_ratios > Asw_As_ratio)

            # adjust the area ratios to use the new A2s downstream of the
            # shock wave
            area_ratios[idx_after_sw] = area_ratios[idx_after_sw] * At / A2s

            # mach number in the convergent
            M[L < Lc] = m_from_critical_area_ratio(
                area_ratios[L < Lc], "sub", self.gamma)
            M[L == Lc] = 1
            # supersonic mach number
            M[idx_before_sw] = m_from_critical_area_ratio(
                area_ratios[idx_before_sw], "super", self.gamma)
            # subsonic mach number
            M[idx_after_sw] = m_from_critical_area_ratio(
                area_ratios[idx_after_sw], "sub", self.gamma)

            P_ratios = pressure_ratio(M, self.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been
            # returned P/P02. Need to compute the ratio P/P0.
            P_ratios[idx_after_sw] *= P02_P0_ratio

            rho_ratios = density_ratio(M, self.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the
            # shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[idx_after_sw] *= P02_P0_ratio

            T_ratios = temperature_ratio(M, self.gamma)

            states["Upstream SW"] = _create_section(
                gamma, R, Mup_sw, T0, P0, rho0,
                temperature_ratio(Mup_sw, gamma),
                pressure_ratio(Mup_sw, gamma),
                density_ratio(Mup_sw, gamma),
                Ae_At_ratio, At)
            Mdo_sw = shockwave.mach_downstream(Mup_sw, gamma)
            A2s = (1 / Ae_A2s_ratio) * Ae
            states["Downstream SW"] = _create_section(
                gamma, R, Mdo_sw, T0, P02_P0_ratio * P0, rho0,
                temperature_ratio(Mdo_sw, gamma),
                pressure_ratio(Mdo_sw, gamma),
                density_ratio(Mdo_sw, gamma),
                Ae_A2s_ratio, A2s)
            states["Exit"] = _create_section(
                gamma, R, M[-1], T0, P02_P0_ratio * P0, rho0,
                T_ratios[-1], P_ratios[-1], rho_ratios[-1],
                Ae_A2s_ratio, A2s)

        if "Exit" not in states:
            states["Exit"] = _create_section(
                gamma, R, M[-1], T0, P0, rho0,
                T_ratios[-1], P_ratios[-1], rho_ratios[-1],
                Ae / At, At)

        states = pd.DataFrame(
            data=states,
            index=self.flow_states_labels
        )
        L -= Lc
        with param.edit_constant(self):
            self.param.update(dict(
                critical_temperature=ct,
                critical_pressure=cp,
                critical_density=cr,
                critical_velocity=cv,
                limit_pressure_ratios=[r1, r2, r3],
                flow_states=states,
                flow_results=[L, M,  P_ratios, rho_ratios, T_ratios],
                mass_flow_rate=nozzle_mass_flow_rate(
                    self.gamma, self.R, self.T0, self.P0,
                    self.nozzle.throat_area)
            ))
        self._update_flow_conditions()
        self._update_flow_condition_summary()

    def _update_flow_conditions(self):
        # TODO: it should be possible to perform the task of this method inside
        # _update_logic, thus reducing code repetition
        P0, T0, rho0, gamma, R = self.P0, self.T0, self.rho0, self.gamma, self.R
        Ae = self.nozzle.outlet_area
        At = self.nozzle.throat_area
        r1, r2, r3 = self.limit_pressure_ratios
        flow_conditions = {}

        # chocked flow
        M = m_from_critical_area_ratio(Ae / At, "sub", gamma)
        pr = pressure_ratio(M, gamma)
        dr = density_ratio(M, gamma)
        tr = temperature_ratio(M, gamma)
        flow_conditions["Chocked"] = [r1] + _create_section(
            gamma, R, M, T0, P0, rho0,
            tr, pr, dr, Ae / At, At)

        # shock wave at the exit plane
        Meup_sw = m_from_critical_area_ratio(Ae / At, "super", gamma)
        Medw_sw = shockwave.mach_downstream(Meup_sw, gamma)
        Ae_A2s_ratio = critical_area_ratio(Medw_sw, gamma)
        A2s = Ae / Ae_A2s_ratio
        P02_P0_ratio = shockwave.total_pressure_ratio(Meup_sw)
        pr = pressure_ratio(Medw_sw, gamma)
        dr = density_ratio(Medw_sw, gamma) * P02_P0_ratio
        tr = temperature_ratio(Medw_sw, gamma)
        flow_conditions["Shock at Exit"] = [r2] + _create_section(
            gamma, R, Medw_sw, T0, P02_P0_ratio * P0, rho0,
            tr, pr, dr, Ae_A2s_ratio, A2s)

        # supercritical
        M = m_from_critical_area_ratio(Ae / At, "super", gamma)
        pr = pressure_ratio(M, gamma)
        dr = density_ratio(M, gamma)
        tr = temperature_ratio(M, gamma)
        flow_conditions["Supercritical"] = [r3] + _create_section(
            gamma, R, M, T0, P0, rho0,
            tr, pr, dr, Ae / At, At)

        labels = ["Pb / P0"] + self.flow_states_labels
        with param.edit_constant(self):
            self.flow_conditions = pd.DataFrame(
                data=flow_conditions,
                index=labels
            )

    @param.depends("_num_decimal_places", watch=True, on_init=True)
    def _update_flow_condition_summary(self):
        def to_string(n):
            if self._num_decimal_places:
                return str(round(n, self._num_decimal_places))
            return str(n)

        r1s, r2s, r3s = [to_string(n) for n in self.limit_pressure_ratios]
        data = {
            "No flow": "Pb / P0 = 1",
            "Subsonic": f"Pb / P0 > {r1s}",
            "Chocked": f"Pb / P0 = {r1s}",
            "Shock in Nozzle": f"{r2s} < Pb / P0 < {r1s}",
            "Shock at Exit": f"Pb / P0 = {r2s}",
            "Overexpanded": f"{r3s} < Pb / P0 < {r2s}",
            "Supercritical": f"Pb / P0 = {r3s}",
            "Underexpanded": f"Pb / P0 < {r3s}",
        }
        df = pd.DataFrame(
            data={"Condition": list(data.values())},
            index=list(data.keys())
        )
        with param.edit_constant(self):
            self.flow_condition_summary = df

    @param.depends(
        "limit_pressure_ratios", "Pb_P0_ratio", watch=True, on_init=True
    )
    def _update_flow_condition(self):
        r1, r2, r3 = self.limit_pressure_ratios

        if self.Pb_P0_ratio is None:
            fc = "Undefined"
        if np.isclose(self.Pb_P0_ratio, 1):
            fc = "No flow"
        if self.Pb_P0_ratio < r3:
            fc = "Underexpanded Flow"
        elif np.isclose(self.Pb_P0_ratio, r3):
            fc = "Design Condition"
        elif self.Pb_P0_ratio < r2:
            fc = "Overexpanded Flow"
        elif self.Pb_P0_ratio >= r2 and self.Pb_P0_ratio < r1:
            fc = "Shock in Nozzle"
        elif np.isclose(self.Pb_P0_ratio, r1):
            fc = "Chocked"
        else:
            fc = "Subsonic Flow"

        with param.edit_constant(self):
            self.current_flow_condition = fc

    @property
    def critical_area(self):
        """Returns the critical area"""
        return self.nozzle.throat_area

    @property
    def inlet_area(self):
        """Returns the inlet area"""
        return self.nozzle.inlet_area

    @property
    def outlet_area(self):
        """Returns the outlet area"""
        return self.nozzle.outlet_area

    def __str__(self):
        s = "De Laval nozzle characteristics:\n"
        s += "Geometry: " + self.nozzle.__str__()
        s += "Critical Quantities:\n"
        s += "\tT*\t{}\n".format(self.critical_temperature)
        s += "\tP*\t{}\n".format(self.critical_pressure)
        s += "\trho*\t{}\n".format(self.critical_density)
        s += "\tu*\t{}\n".format(self.critical_velocity)
        s += f"Mass flow rate: {self.mass_flow_rate}\n"
        s += "Important Pressure Ratios:\n"
        s += f"\tr1\t{self.limit_pressure_ratios[0]}\n"
        s += f"\tr2\t{self.limit_pressure_ratios[1]}\n"
        s += f"\tr3\t{self.limit_pressure_ratios[2]}\n"
        s += "Flow Conditions:\n"
        s += f"\tSubsonic:\tPb/P0 > {self.limit_pressure_ratios[0]}\n"
        s += f"\tChocked:\tPb/P0 = {self.limit_pressure_ratios[0]}\n"
        s += f"\tInternal Shock:\t{self.limit_pressure_ratios[0]} < Pb/P0 < {self.limit_pressure_ratios[1]}\n"
        s += f"\tShock at Exit:\tPb/P0 = {self.limit_pressure_ratios[1]}\n"
        s += f"\tOverexpanded:\t{self.limit_pressure_ratios[1]} < Pb/P0 < {self.limit_pressure_ratios[2]}\n"
        s += f"\tSupercritical:\tPb/P0 = {self.limit_pressure_ratios[2]}\n"
        s += f"\tunderexpanded:\tPb/P0 > {self.limit_pressure_ratios[2]}\n"
        s += "Nozzle Pressure Ratio, Pb/P0: \t{}\n".format(self.Pb_P0_ratio)
        s += "Current Flow Condition: \t{}\n".format(self.current_flow_condition)
        for state, values in self.flow_states.items():
            s += f"State: {state}\n"
            s += "\n".join([f"\t{l}: {v}" for l, v in zip(
                self.flow_states_labels, values)]) + "\n"
        return s

    def plot(self, interactive=True, show_nozzle=True, **params):
        """Visualize the results.

        Parameters
        ----------
        interactive : bool
            If True, returns an interactive application in the form of a
            servable object, which will be automatically rendered inside a
            Jupyter Notebook. If any other interpreter is used, then
            ``solver.plot(interactive=True).show()`` might be requires in
            order to visualize the application on a browser.
            If False, a Bokeh figure will be shown on the screen.
        show_nozzle : bool
        **params :
            Keyword arguments sent to ``DeLavalDiagram``.

        Returns
        -------
        app :
            The application or figure to be shown.
        """
        from pygasflow.interactive.diagrams import DeLavalDiagram
        d = DeLavalDiagram(
            solver=self,
            show_nozzle=show_nozzle,
            **params
        )
        if interactive:
            return d.servable()

        from bokeh.plotting import show
        if show_nozzle:
            from bokeh.layouts import column
            c = column(d.nozzle_diagram.figure, d.figure)
            show(c)
            return c
        show(d.figure)
        return d.figure


def find_shockwave_area_ratio(Ae_At_ratio, Pe_P0_ratio, R, gamma):
    """ Iterative procedure to find the critical area ratio where the shock wave happens.

    Parameters
    ----------
    Ae_At_ratio : float
        Area ratio between the exit section of the divergent and the
        throat section.
    Pe_P0_ratio : float
        Pressure ratio between the back (= exit) pressure to the
        reservoir pressure.
    R : float
        Specific Gas Constant.
    gamma : float
        Specific Heats ratio.

    Returns
    -------
    Asw/At : float
        The critical area ratio Asw/At

    Examples
    --------

    >>> from pygasflow.solvers import find_shockwave_area_ratio
    >>> find_shockwave_area_ratio(20, 0.1, 287, 1.4)
    14.247517873372033

    """
    # Pe: pressure at the exit section = back pressure
    # P01: total pressure = P0
    #
    # Back to Reservoir pressure ratio:
    # (Pe/P0) = (Pe/P02) * (P02/P2) * (P2/P1) * (P1/P01)
    #
    # A1* = At: critical area upstream of the shock wave, coincide with the throat area
    # A2*: critical area downstream of the shock wave
    # (Ae/A2*) = (Ae/A1*) * (A1*/Asw) * (Asw/A2*)
    def func(Asw_At_ratio):
        """
        Args:
        Asw_At_ratio : float
            Estimate of the critical area ratio between the Shock Wave area
            and the throat area.

        Return:
            The zero function "estimated ratio Pe/P0" - "Pe/P0".
        """
        # Mach number just upstream of the shock wave
        Mup_sw = m_from_critical_area_ratio(Asw_At_ratio, "super", gamma)

        P1_P01_ratio = pressure_ratio(Mup_sw, gamma)
        # pressure ratio across the shock wave
        P2_P1_ratio = shockwave.pressure_ratio(Mup_sw, gamma)
        # Mach number just downstream of the shock wave
        Mdown_sw = shockwave.mach_downstream(Mup_sw, gamma)
        P02_P2_ratio = 1 / pressure_ratio(Mdown_sw)

        # critical area ratio just downstream of the shock wave
        Asw_A2s_ratio = critical_area_ratio(Mdown_sw, gamma)
        # critical area ratio at the exit (downstream of the shock wave)
        Ae_A2s_ratio = Ae_At_ratio / Asw_At_ratio * Asw_A2s_ratio
        # Mach number at the exit section
        Me = m_from_critical_area_ratio(Ae_A2s_ratio, "sub", gamma)
        Pe_P02_ratio = pressure_ratio(Me, gamma)
        estimated_Pe_P0_ratio = Pe_P02_ratio * P02_P2_ratio * P2_P1_ratio * P1_P01_ratio

        return estimated_Pe_P0_ratio - Pe_P0_ratio

    return bisect(func, 1, Ae_At_ratio)
