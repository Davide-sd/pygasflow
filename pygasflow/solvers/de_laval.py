import numpy as np
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pygasflow.shockwave as shockwave
from pygasflow.isentropic import (
    temperature_ratio,
    pressure_ratio,
    density_ratio,
    temperature_ratio,
    m_from_critical_area_ratio,
    m_from_critical_area_ratio_and_pressure_ratio,
    critical_area_ratio,
)
from pygasflow.generic import sound_speed
from pygasflow.utils.common import Flow_State, Ideal_Gas

from pygasflow.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle
)

class De_Laval_Solver(object):
    """
    Solve a De Laval Nozzle (Convergent-Divergent nozzle), starting from
    stagnation conditions and the geometry type.

    Examples
    --------

    Visualize all the quantities along the length of a TOP nozzle using air
    and a back-to-stagnation pressure ratio of 0.12.

    .. plot::
       :context: reset
       :format: python
       :include-source: True

       from pygasflow.nozzles import CD_TOP_Nozzle
       from pygasflow.solvers import De_Laval_Solver
       from pygasflow.utils import Ideal_Gas, Flow_State
       # Set air as the gas to use in the nozzle
       gas = Ideal_Gas(287, 1.4)
       # stagnation condition
       upstream_state = Flow_State(p0=8 * 101325, t0 = 303.15)
       Ri = 0.4            # Inlet Radius
       Rt = 0.2            # Throat Radius
       Re = 1.2            # Exit (outlet) Radius
       theta_c = 40        # half cone angle of the divergent
       theta_N = 15        # half cone angle of the convergent
       Rbt = 0.75 * Rt     # Junction radius between the convergent and divergent
       Rbc = 1.5 * Rt      # Junction radius between the "combustion chamber" and convergent
       K = 0.8             # Fractional Length of the TOP nozzle
       geom_type = "axisymmetric"  # geometry type
       geom_top = CD_TOP_Nozzle(Ri, Re, Rt, Rbc, theta_c, K, geom_type, 1000)
       nozzle_top = De_Laval_Solver(gas, geom_top, upstream_state)
       nozzle_top.plot(0.12)

    Compute the critical area ratio where the shockwave is located:

    >>> from pygasflow.nozzles import CD_TOP_Nozzle
    >>> from pygasflow.solvers import De_Laval_Solver
    >>> from pygasflow.utils import Ideal_Gas, Flow_State
    >>> gas = Ideal_Gas(287, 1.4) # Set air as the gas to use in the nozzle
    >>> upstream_state = Flow_State(p0=8 * 101325, t0 = 303.15) # stagnation condition
    >>> Ri, Rt, Re = 0.4, 0.2, 1.2
    >>> theta_c, theta_N = 40, 15
    >>> Rbt, Rbc = 0.75 * Rt, 1.5 * Rt
    >>> geom_top = CD_TOP_Nozzle(Ri, Re, Rt, Rbc, theta_c, 0.8, "axisymmetric")
    >>> nozzle_top = De_Laval_Solver(gas, geom_top, upstream_state)
    >>> results = nozzle_top.compute(0.12)
    >>> print(results[-1])
    12.329635184167444

    """

    def __init__(self, gas, geometry, input_state, Pb_P0_ratio=None):
        """
        Parameters
        ----------
        gas : Ideal_Gas
            Gas used in the nozzle.
        geometry : Nozzle_Geometry
            Nozzle geometry (lengths, areas, ...)
        input_state : Flow_State
            Represents the stagnation flow state.
        Pb_P0_ratio : float, optional
            Back to Stagnation pressure ratio. Default to None.
        """
        self._gas = gas
        self._geometry = geometry
        self._input_state = input_state

        R = gas.R
        gamma = gas.gamma
        T0 = input_state.total_temperature
        P0 = input_state.total_pressure
        rho0 = gas.solve(p=P0, t=T0)

        Ae_As_ratio = geometry.outlet_area / geometry.critical_area

        self._critical_temperature = temperature_ratio(1, gamma) * T0
        self._critical_pressure = pressure_ratio(1, gamma) * P0
        self._critical_density = density_ratio(1, gamma) * rho0
        self._critical_velocity = sound_speed(self._critical_temperature, R, gamma)

        M2_sub = m_from_critical_area_ratio(Ae_As_ratio, "sub", gamma)
        M2_sup = m_from_critical_area_ratio(Ae_As_ratio, "super", gamma)

        # exit pressure ratio corresponding to fully subsonic flow in the divergent (when M=1 in A*)
        r1 = pressure_ratio(M2_sub, gamma)
        # exit pressure ratio corresponding to fully supersonic flow in the divergent
        # and Pe = Pb (pressure at back). Design condition.
        r3 = pressure_ratio(M2_sup, gamma)

        # isentropic pressure at the exit section of the divergent
        p_exit = r3 * P0

        # pressure downstream of the normal shock wave at the exit section of the divergent
        p2 = shockwave.pressure_ratio(M2_sup, gamma) * p_exit
        # exit pressure ratio corresponding to a normal shock wave at the exit section of the divergent
        r2 = p2 / P0

        self._r1 = r1
        self._r2 = r2
        self._r3 = r3

        self._flow_condition = self.flow_condition(self._gas, self._input_state, Pb_P0_ratio)
        self._output_state = None

        self._Pb_P0_ratio = Pb_P0_ratio
        if self._Pb_P0_ratio:
            # compute output state
            _, _, M, pr, rhor, tr, _, _ = self.compute(self._Pb_P0_ratio)

            self._output_state = Flow_State(
                m = M[-1],
                p = pr[-1] * P0,
                rho = rhor[-1] * rho0,
                t = tr[-1] * T0,
                p0 = self._Pb_P0_ratio * P0,
                t0 = T0
            )

    def flow_condition(self, gas, upstream_state, Pb_P0_ratio):
        """
        Return the flow condition inside the nozzle given the Back to Stagnation pressure ratio.

        Parameters
        ----------
        Pb_P0_ratio : float
            Back to Stagnation pressure ratio.

        Returns
        -------
        Flow_Condition : str
        """
        if Pb_P0_ratio == None:
            return "Undefined"
        if Pb_P0_ratio == 1:
            return "No flow"
        if Pb_P0_ratio < self._r3:
            return "Underexpanded Flow"
        elif Pb_P0_ratio == self._r3:
            return "Design Condition"
        elif Pb_P0_ratio < self._r2:
            return "Overexpanded Flow"
        elif Pb_P0_ratio >= self._r2 and Pb_P0_ratio < self._r1:
            return "Shock in Nozzle"
        else:
            return "Subsonic Flow"

    def compute(self, Pe_P0_ratio):
        """
        Compute the flow quantities along the nozzle geometry.

        Parameters
        ----------
        Pe_P0_ratio : float
            Back to Stagnation pressure ratio. The pressure at the
            exit plane coincide with the back pressure.

        Returns
        -------
        L : np.ndarray
            Lengths along the stream flow.
        area_ratios : np.ndarray
            Area ratios along the stream flow.
        M : np.ndarray
            Mach numbers.
        P_ratios : np.ndarray
            Pressure ratios.
        rho_ratios : np.ndarray
            Density ratios.
        T_ratios : np.ndarray
            Temperature ratios.
        flow_condition : string
            The flow condition given the input pressure ratio.
        Asw_At_ratio : float
            Area ratio of the shock wave location if present, otherwise
            return None.
        """
        self._flow_condition = self.flow_condition(self._gas, self._input_state, Pe_P0_ratio)

        Ae = self._geometry.outlet_area
        At = self._geometry.critical_area
        Lc = self._geometry.length_convergent
        # copy the arrays: we do not want to modify the original geometry in case of
        # shock wave at the exit plane
        area_ratios = np.copy(self._geometry.area_ratio_array)
        L = np.copy(self._geometry.length_array) + Lc

        M = np.zeros_like(area_ratios)
        P_ratios = np.zeros_like(area_ratios)
        rho_ratios = np.zeros_like(area_ratios)
        T_ratios = np.zeros_like(area_ratios)
        flow_condition = self.flow_condition(self._gas, self._input_state, Pe_P0_ratio)
        Asw_At_ratio = None

        if Pe_P0_ratio > 1:
            raise ValueError("The back to reservoir pressure ratio must be Pe/P0 <= 1.")

        elif Pe_P0_ratio == 1:  # no flow
            P_ratios += 1
            T_ratios += 1
            rho_ratios += 1

        elif Pe_P0_ratio >= self._r1:  # fully subsonic flow
            M = m_from_critical_area_ratio(area_ratios, "sub", self._gas.gamma)
            P_ratios = pressure_ratio(M, self._gas.gamma)
            rho_ratios = density_ratio(M, self._gas.gamma)
            T_ratios = temperature_ratio(M, self._gas.gamma)

        elif Pe_P0_ratio < self._r2: # fully supersonic flow in the divergent
            M[L < Lc] = m_from_critical_area_ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            M[L > Lc] = m_from_critical_area_ratio(area_ratios[L > Lc], "super", self._gas.gamma)
            P_ratios = pressure_ratio(M, self._gas.gamma)
            rho_ratios = density_ratio(M, self._gas.gamma)
            T_ratios = temperature_ratio(M, self._gas.gamma)

        elif Pe_P0_ratio == self._r2:    # shock wave at the exit plane
            Ae_At_ratio = Ae / At
            Asw_At_ratio = Ae_At_ratio
            # Supersonic Mach number at the exit section just upstream of the shock wave
            Meup_sw = m_from_critical_area_ratio(Ae_At_ratio, "super", self._gas.gamma)
            # Subsonic Mach number at the exit section just downstream of the shock wave
            Medw_sw = shockwave.mach_downstream(Meup_sw, self._gas.gamma)
            # downstream of the shock wave there is a new isentropic critical area ratio
            Ae_A2s_ratio = critical_area_ratio(Medw_sw, self._gas.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.total_pressure_ratio(Meup_sw)

            M[L < Lc] = m_from_critical_area_ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            M[L > Lc] = m_from_critical_area_ratio(area_ratios[L > Lc], "super", self._gas.gamma)

            # append the last subsonic point at the exit
            M = np.append(M, m_from_critical_area_ratio(Ae_A2s_ratio, "sub", self._gas.gamma))
            P_ratios = pressure_ratio(M, self._gas.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been returned P/P02.
            # Need to compute the ratio P/P0.
            P_ratios[-1] *= P02_P0_ratio

            rho_ratios = density_ratio(M, self._gas.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[-1] *= P02_P0_ratio

            T_ratios = temperature_ratio(M, self._gas.gamma)
            L = np.append(L, L[-1])
            area_ratios = np.append(area_ratios, Ae_At_ratio)
        else:   # shock into the divergent
            Ae_At_ratio = Ae / At
            # area ratio of the shock wave
            Asw_At_ratio = find_shockwave_area_ratio(Ae_At_ratio, Pe_P0_ratio, self._gas.R, self._gas.gamma)
            # Mach number at the exit section given the exit pressure ratio Pe_P0_ratio
            Me = m_from_critical_area_ratio_and_pressure_ratio(Ae_At_ratio, Pe_P0_ratio, self._gas.gamma)
            # downstream of the shock wave there is a new isentropic critical area ratio
            Ae_A2s_ratio = critical_area_ratio(Me, self._gas.gamma)
            # critical area downstream of the shock wave
            A2s = Ae / Ae_A2s_ratio
            # Mach number just upstream of the shock wave
            Mup_sw = m_from_critical_area_ratio(Asw_At_ratio, "super", self._gas.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.total_pressure_ratio(Mup_sw)

            # find indeces before and after the shock wave in the divergent
            idx_before_sw = np.bitwise_and(L > Lc, area_ratios <= Asw_At_ratio)
            idx_after_sw = np.bitwise_and(L > Lc, area_ratios > Asw_At_ratio)

            # adjust the area ratios to use the new A2s downstream of the shock wave
            area_ratios[idx_after_sw] = area_ratios[idx_after_sw] * At / A2s

            # mach number in the convergent
            M[L < Lc] = m_from_critical_area_ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            # supersonic mach number
            M[idx_before_sw] = m_from_critical_area_ratio(area_ratios[idx_before_sw], "super", self._gas.gamma)
            # subsonic mach number
            M[idx_after_sw] = m_from_critical_area_ratio(area_ratios[idx_after_sw], "sub", self._gas.gamma)

            P_ratios = pressure_ratio(M, self._gas.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been returned P/P02.
            # Need to compute the ratio P/P0.
            P_ratios[idx_after_sw] *= P02_P0_ratio

            rho_ratios = density_ratio(M, self._gas.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[idx_after_sw] *= P02_P0_ratio

            T_ratios = temperature_ratio(M, self._gas.gamma)
        L -= Lc
        return L, area_ratios, M, P_ratios, rho_ratios, T_ratios, flow_condition, Asw_At_ratio

    @property
    def critical_temperature(self):
        """Returns the critical temperature"""
        return self._critical_temperature

    @property
    def critical_pressure(self):
        """Returns the critical pressure"""
        return self._critical_pressure

    @property
    def critical_density(self):
        """Returns the critical density"""
        return self._critical_density

    @property
    def critical_velocity(self):
        """Returns the critical velocity"""
        return self._critical_velocity

    @property
    def critical_area(self):
        """Returns the critical area"""
        return self._geometry.critical_area

    @property
    def inlet_area(self):
        """Returns the inlet area"""
        return self._geometry.inlet_area

    @property
    def outlet_area(self):
        """Returns the outlet area"""
        return self._geometry.outlet_area

    @property
    def limit_pressure_ratios(self):
        """
        Returns
        -------
        r1 : float
            Exit pressure ratio corresponding to fully subsonic flow in the
            divergent (when M=1 in A*).
        r2 : float
            Exit pressure ratio corresponding to a normal shock wave at the
            exit section of the divergent.
        r3 : float
            Exit pressure ratio corresponding to fully supersonic flow in the
            divergent and Pe = Pb (pressure at back). Design condition.

        """
        return [self._r1, self._r2, self._r3]

    def __str__(self):
        s = "De Laval nozzle characteristics:\n"
        s += "Geometry: " + self._geometry.__str__()
        s += "Critical Quantities:\n"
        s += "\tT*\t{}\n".format(self._critical_temperature)
        s += "\tP*\t{}\n".format(self._critical_pressure)
        s += "\trho*\t{}\n".format(self._critical_density)
        s += "\tu*\t{}\n".format(self._critical_velocity)
        s += "Important Pressure Ratios:\n"
        s += "\tr1\t{}\n".format(self._r1)
        s += "\tr2\t{}\n".format(self._r2)
        s += "\tr3\t{}\n".format(self._r3)
        s += "Flow Condition: \t{}\n".format(self._flow_condition)
        s += "Input state\t{}\n".format(self._input_state)
        if self._output_state != None:
            s += "Output state\t{}\n".format(self._output_state)
        return s

    def plot(self, Pb_P0_ratio=None, show=True, size=None, title=None):
        """Visualize the nozzle geometry, the Mach number and the ratios along
        the length of the nozzle.

        Parameters
        ----------
        Pb_P0_ratio : float, optional
            Back to Stagnation pressure ratio. Default to None.
        size : (width, height) or None, optional
            Set the figure size, in inches.
        title : str or None
            Title of the overall plot. Default to None, in which case a
            default title will be added, showing the type of the geometry and
            the flow condition.

        """
        if (Pb_P0_ratio is None) and (self._Pb_P0_ratio is None):
            raise ValueError("Pb_P0_ratio must be provided.")
        elif (Pb_P0_ratio is None):
            Pb_P0_ratio = self._Pb_P0_ratio

        if size is None:
            size = (8, 10)

        L, A, M, P, rho, T, flow_condition, Asw_At_ratio = self.compute(Pb_P0_ratio)

        if title is None:
            title = self._geometry._title + ": " + flow_condition

        fig, ax = plt.subplots(nrows=4, sharex=True)
        fig.set_size_inches(size)
        radius_nozzle, radius_container = self._geometry.get_points(False)
        ar_nozzle, ar_container = self._geometry.get_points(True)

        # nozzle geometry
        ax[0].add_patch(patches.Polygon(radius_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))
        ax[0].add_patch(patches.Polygon(radius_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))
        ax[0].set_ylim(0, max(radius_container[:, 1]))
        ax[0].set_ylabel("r [m]")
        ax[0].set_title(title)

        ax[1].add_patch(patches.Polygon(ar_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))
        ax[1].add_patch(patches.Polygon(ar_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))
        ax[1].set_ylim(0, max(ar_container[:, 1]))
        ax[1].set_ylabel("$A/A^{*}$")

        # draw the shock wave if present in the nozzle
        if Asw_At_ratio:
            # get shock wave location in the divergent
            x = self._geometry.location_divergent_from_area_ratio(Asw_At_ratio)
            rsw = np.sqrt((Asw_At_ratio * self._geometry.critical_area) / np.pi)
            ax[0].plot([x, x], [0, rsw], 'r')
            ax[1].plot([x, x], [0, Asw_At_ratio], 'r')
            ax[0].text(x, rsw + 0.5 * (max(radius_container[:, 1]) - max(radius_nozzle[:, -1])),
                "SW",
                color="r",
                ha='center',
                va="center",
                bbox=dict(boxstyle="round", fc="white", lw=0, alpha=0.85),
            )
            ax[1].text(x, Asw_At_ratio + 0.5 * (max(ar_container[:, 1]) - max(ar_nozzle[:, -1])),
                "SW",
                color="r",
                ha='center',
                va="center",
                bbox=dict(boxstyle="round", fc="white", lw=0, alpha=0.85),
            )

        # mach number
        ax[2].plot(L, M)
        ax[2].set_ylim(0)
        ax[2].grid()
        ax[2].set_ylabel("M")

        # ratios
        ax[3].plot(L, P, label="$P/P_{0}$")
        ax[3].plot(L, rho, label=r"$\rho/\rho_{0}$")
        ax[3].plot(L, T, label="$T/T_{0}$")
        ax[3].set_xlim(min(ar_container[:, 0]), max(ar_container[:, 0]))
        ax[3].set_ylim(0, 1)
        ax[3].legend(loc="lower left")
        ax[3].grid()
        ax[3].set_xlabel("L [m]")
        ax[3].set_ylabel("ratios")

        plt.tight_layout()
        plt.show()


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
