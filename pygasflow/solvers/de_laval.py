import numpy as np
from scipy.optimize import bisect

import pygasflow.shockwave as shockwave
from pygasflow.isentropic import (
    Temperature_Ratio,
    Pressure_Ratio,
    Density_Ratio,
    Temperature_Ratio,
    M_From_Critical_Area_Ratio,
    M_From_Critical_Area_Ratio_And_Pressure_Ratio,
    Critical_Area_Ratio,
)
from pygasflow.generic import Sound_Speed
from pygasflow.utils.common import Flow_State, Ideal_Gas

from pygasflow.utils.nozzles import (
    CD_Conical_Nozzle,
    CD_TOP_Nozzle
)

class De_Laval_Solver(object):
    """ 
    Solve a De Laval Nozzle (Convergent-Divergent nozzle), starting from
    stagnation conditions.
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
            Pb_P0_ratio : float
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

        Ae_As_ratio = geometry.Outlet_area / geometry.Critical_area
        
        self._critical_temperature = Temperature_Ratio(1, gamma) * T0
        self._critical_pressure = Pressure_Ratio(1, gamma) * P0
        self._critical_density = Density_Ratio(1, gamma) * rho0
        self._critical_velocity = Sound_Speed(self._critical_temperature, R, gamma)

        M2_sub = M_From_Critical_Area_Ratio(Ae_As_ratio, "sub", gamma)
        M2_sup = M_From_Critical_Area_Ratio(Ae_As_ratio, "sup", gamma)

        # exit pressure ratio corresponding to fully subsonic flow in the divergent (when M=1 in A*)
        r1 = Pressure_Ratio(M2_sub, gamma)
        # exit pressure ratio corresponding to fully supersonic flow in the divergent 
        # and Pe = Pb (pressure at back). Design condition.
        r3 = Pressure_Ratio(M2_sup, gamma)

        # isentropic pressure at the exit section of the divergent
        p_exit = r3 * P0

        # pressure downstream of the normal shock wave at the exit section of the divergent
        p2 = shockwave.Pressure_Ratio(M2_sup, gamma) * p_exit
        # exit pressure ratio corresponding to a normal shock wave at the exit section of the divergent
        r2 = p2 / P0

        self._r1 = r1
        self._r2 = r2
        self._r3 = r3

        self._flow_condition = self.Flow_Condition(Pb_P0_ratio)
        self._output_state = None

        if Pb_P0_ratio:
            # compute output state
            _, _, M, pr, rhor, tr, _, _ = self.Compute(Pb_P0_ratio)

            self._output_state = Flow_State(
                m = M[-1],
                p = pr[-1] * P0,
                rho = rhor[-1] * rho0,
                t = tr[-1] * T0,
                p0 = Pb_P0_ratio * P0,
                t0 = T0
            )
    
    def Flow_Condition(self, Pb_P0_ratio):
        """
        Return the flow condition inside the nozzle given the Back to Stagnation pressure ratio.

        Parameters
        ----------
            Pb_P0_ratio : float
                Back to Stagnation pressure ratio.
        
        Returns
        -------
            Flow Condition [string]
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

    def Compute(self, Pe_P0_ratio):
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
                Area ratio of the shock wave location, if present.
        """
        Ae = self._geometry.Outlet_area
        At = self._geometry.Critical_area
        Lc = self._geometry.Length_Convergent
        # copy the arrays: we do not want to modify the original geometry in case of 
        # shock wave at the exit plane
        area_ratios = np.copy(self._geometry.Area_ratio_array)
        L = np.copy(self._geometry.Length_array) + Lc

        M = np.zeros_like(area_ratios)
        P_ratios = np.zeros_like(area_ratios)
        rho_ratios = np.zeros_like(area_ratios)
        T_ratios = np.zeros_like(area_ratios)
        flow_condition = self.Flow_Condition(Pe_P0_ratio)
        Asw_At_ratio = None

        if Pe_P0_ratio > 1:
            raise ValueError("The back to reservoir pressure ratio must be Pe/P0 <= 1.")

        elif Pe_P0_ratio == 1:  # no flow
            P_ratios += 1
            T_ratios += 1
            rho_ratios += 1

        elif Pe_P0_ratio >= self._r1:  # fully subsonic flow
            M = M_From_Critical_Area_Ratio(area_ratios, "sub", self._gas.gamma)
            P_ratios = Pressure_Ratio(M, self._gas.gamma)
            rho_ratios = Density_Ratio(M, self._gas.gamma)
            T_ratios = Temperature_Ratio(M, self._gas.gamma)

        elif Pe_P0_ratio < self._r2: # fully supersonic flow in the divergent
            M[L < Lc] = M_From_Critical_Area_Ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            M[L > Lc] = M_From_Critical_Area_Ratio(area_ratios[L > Lc], "sup", self._gas.gamma)
            P_ratios = Pressure_Ratio(M, self._gas.gamma)
            rho_ratios = Density_Ratio(M, self._gas.gamma)
            T_ratios = Temperature_Ratio(M, self._gas.gamma)
            
        elif Pe_P0_ratio == self._r2:    # shock wave at the exit plane
            Ae_At_ratio = Ae / At
            Asw_At_ratio = Ae_At_ratio
            # Supersonic Mach number at the exit section just upstream of the shock wave
            Meup_sw = M_From_Critical_Area_Ratio(Ae_At_ratio, "sup", self._gas.gamma)
            # Subsonic Mach number at the exit section just downstream of the shock wave
            Medw_sw = shockwave.Mach_Downstream(Meup_sw, self._gas.gamma)
            # downstream of the shock wave there is a new isentropic critical area ratio
            Ae_A2s_ratio = Critical_Area_Ratio(Medw_sw, self._gas.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.Total_Pressure_Ratio(Meup_sw)

            M[L < Lc] = M_From_Critical_Area_Ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            M[L > Lc] = M_From_Critical_Area_Ratio(area_ratios[L > Lc], "sup", self._gas.gamma)
            
            # append the last subsonic point at the exit
            M = np.append(M, M_From_Critical_Area_Ratio(Ae_A2s_ratio, "sub", self._gas.gamma))
            P_ratios = Pressure_Ratio(M, self._gas.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been returned P/P02.
            # Need to compute the ratio P/P0.
            P_ratios[-1] *= P02_P0_ratio

            rho_ratios = Density_Ratio(M, self._gas.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[-1] *= P02_P0_ratio

            T_ratios = Temperature_Ratio(M, self._gas.gamma)
            L = np.append(L, L[-1])
            area_ratios = np.append(area_ratios, Ae_At_ratio)   
        else:   # shock into the divergent
            Ae_At_ratio = Ae / At
            # area ratio of the shock wave
            Asw_At_ratio = Find_Shockwave_Area_Ratio(Ae_At_ratio, Pe_P0_ratio, self._gas.R, self._gas.gamma)
            # Mach number at the exit section given the exit pressure ratio Pe_P0_ratio
            Me = M_From_Critical_Area_Ratio_And_Pressure_Ratio(Ae_At_ratio, Pe_P0_ratio, self._gas.gamma)
            # downstream of the shock wave there is a new isentropic critical area ratio
            Ae_A2s_ratio = Critical_Area_Ratio(Me, self._gas.gamma)
            # critical area downstream of the shock wave
            A2s = Ae / Ae_A2s_ratio            
            # Mach number just upstream of the shock wave
            Mup_sw = M_From_Critical_Area_Ratio(Asw_At_ratio, "sup", self._gas.gamma)
            # total pressure ratio across the shock wave
            P02_P0_ratio = shockwave.Total_Pressure_Ratio(Mup_sw)

            # find indeces before and after the shock wave in the divergent
            idx_before_sw = np.bitwise_and(L > Lc, area_ratios <= Asw_At_ratio)
            idx_after_sw = np.bitwise_and(L > Lc, area_ratios > Asw_At_ratio)

            # adjust the area ratios to use the new A2s downstream of the shock wave
            area_ratios[idx_after_sw] = area_ratios[idx_after_sw] * At / A2s

            # mach number in the convergent
            M[L < Lc] = M_From_Critical_Area_Ratio(area_ratios[L < Lc], "sub", self._gas.gamma)
            M[L == Lc] = 1
            # supersonic mach number
            M[idx_before_sw] = M_From_Critical_Area_Ratio(area_ratios[idx_before_sw], "sup", self._gas.gamma)
            # subsonic mach number
            M[idx_after_sw] = M_From_Critical_Area_Ratio(area_ratios[idx_after_sw], "sub", self._gas.gamma)

            P_ratios = Pressure_Ratio(M, self._gas.gamma)
            # For idx_after_sw (downstream of the shock wave), I've been returned P/P02.
            # Need to compute the ratio P/P0.
            P_ratios[idx_after_sw] *= P02_P0_ratio

            rho_ratios = Density_Ratio(M, self._gas.gamma)
            # P02 = rho02 * R * T02
            # P01 = rho01 * R * T01
            # Taking the ratios, and knowing that T01 = T02 across the shock wave, leads to:
            # P02 / P01 = rho02 / rho01
            # Need to compute the ratio rho/rho0 after the shock wave
            rho_ratios[idx_after_sw] *= P02_P0_ratio

            T_ratios = Temperature_Ratio(M, self._gas.gamma)
        L -= Lc
        return L, area_ratios, M, P_ratios, rho_ratios, T_ratios, flow_condition, Asw_At_ratio

    @property
    def Critical_Temperature(self):
        return self._critical_temperature
    
    @property
    def Critical_Pressure(self):
        return self._critical_pressure
    
    @property
    def Critical_Density(self):
        return self._critical_density
    
    @property
    def Critical_Velocity(self):
        return self._critical_velocity
    
    @property
    def Critical_Area(self):
        return self._geometry.Critical_area
    
    @property
    def Inlet_Area(self):
        return self._geometry.Inlet_area
    
    @property
    def Outlet_Area(self):
        return self._geometry.Outlet_area
    
    @property
    def Limit_Pressure_Ratios(self):
        return [self._r1, self._r2, self._r3]
    
    def __str__(self):
        s = "De Laval nozzle characteristics:\n"
        s += "Geometry: " + self._geometry.__str__()
        s += "Critical Quantities:\n"
        s += "T*\t{}\n".format(self._critical_temperature)
        s += "P*\t{}\n".format(self._critical_pressure)
        s += "rho*\t{}\n".format(self._critical_density)
        s += "u*\t{}\n".format(self._critical_velocity)
        s += "Important Pressure Ratios:\n"
        s += "r1\t{}\n".format(self._r1)
        s += "r2\t{}\n".format(self._r2)
        s += "r3\t{}\n".format(self._r3)
        s += "Flow Condition: \t{}\n".format(self._flow_condition)
        s += "Input state\t{}\n".format(self._input_state)
        if self._output_state != None:
            s += "Output state\t{}\n".format(self._output_state)
        return s


def Find_Shockwave_Area_Ratio(Ae_At_ratio, Pe_P0_ratio, R, gamma):
    """ Iterative procedure to find the critical area ratio where the shock wave happens.

    Args:
        Ae_At_ratio:    Area ratio between the exit section of the divergent and the throat section.
        Pe_P0_ratio:    Pressure ratio between the back (= exit) pressure to the reservoir pressure.
        R:              Specific Gas Constant.
        gamma:          Specific Heats ratio.
    
    Return:
        The critical area ratio Asw/At
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
            Asw_At_ratio:   Estimate of the critical area ratio between the Shock Wave area and the throat area.
        
        Return:
            The zero function "estimated ratio Pe/P0" - "Pe/P0".
        """
        # Mach number just upstream of the shock wave
        Mup_sw = M_From_Critical_Area_Ratio(Asw_At_ratio, "sup", gamma)

        P1_P01_ratio = Pressure_Ratio(Mup_sw, gamma)
        # pressure ratio across the shock wave
        P2_P1_ratio = shockwave.Pressure_Ratio(Mup_sw, gamma)
        # Mach number just downstream of the shock wave
        Mdown_sw = shockwave.Mach_Downstream(Mup_sw, gamma)
        P02_P2_ratio = 1 / Pressure_Ratio(Mdown_sw)

        # critical area ratio just downstream of the shock wave
        Asw_A2s_ratio = Critical_Area_Ratio(Mdown_sw, gamma)
        # critical area ratio at the exit (downstream of the shock wave)
        Ae_A2s_ratio = Ae_At_ratio / Asw_At_ratio * Asw_A2s_ratio
        # Mach number at the exit section
        Me = M_From_Critical_Area_Ratio(Ae_A2s_ratio, "sub", gamma)
        Pe_P02_ratio = Pressure_Ratio(Me, gamma)
        estimated_Pe_P0_ratio = Pe_P02_ratio * P02_P2_ratio * P2_P1_ratio * P1_P01_ratio

        return estimated_Pe_P0_ratio - Pe_P0_ratio

    return bisect(func, 1, Ae_At_ratio)