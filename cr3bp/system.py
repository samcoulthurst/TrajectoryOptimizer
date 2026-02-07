"""
CR3BP System Class

This module contains the CR3BPSystem class which encapsulates system parameters
and provides convenient methods for unit conversions, frame conversions, and
solving the equations of motion.
"""

import numpy as np
from .dynamics import cr3bp_eom, solve_CR3BP
from .conversions import (
    rotating_to_inertial, inertial_to_rotating,
    convert_trajectory_to_inertial, convert_trajectory_to_rotating,
    compute_characteristic_scales,
    state_to_normalized, state_to_SI,
    trajectory_to_normalized, trajectory_to_SI,
    position_to_normalized, position_to_SI,
    velocity_to_normalized, velocity_to_SI,
    time_to_normalized, time_to_SI
)


class CR3BPSystem:
    """
    Class to handle CR3BP system parameters, unit conversions, and 
    provide convenient access to dynamics.
    
    This class stores the physical parameters of a CR3BP system and provides
    methods for converting between SI and normalized units, as well as 
    convenient wrappers for solving the equations of motion.
    """
    
    def __init__(self, m1, m2, r1, r2, d):
        """
        Initialize CR3BP system parameters.
        
        Parameters
        ----------
        m1 : float
            Mass of primary 1 (larger body) in kg
        m2 : float
            Mass of primary 2 (smaller body) in kg
        d : float
            Distance between primaries in meters
        """
        # Properties of the system
        self.m1 = m1  # kg
        self.m2 = m2  # kg
        self.m_total = m1 + m2  # kg
        self.d = d  # meters
        self.r1 = r1
        self.r2 = r2
        
        # Gravitational constant
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        
        # Compute characteristic scales
        scales = compute_characteristic_scales(m1, m2, d, self.G)
        self.mu = scales['mu']
        self.l_star = scales['l_star']
        self.t_star = scales['t_star']
        self.v_star = scales['v_star']
        self.a_star = scales['a_star']
        
    def __repr__(self):
        return (f"CR3BPSystem(m1={self.m1:.3e} kg, m2={self.m2:.3e} kg, "
                f"d={self.d:.3e} m, mu={self.mu:.6f})")
    
    def info(self):
        """Print system information and characteristic scales."""
        print(f"CR3BP System Information:")
        print(f"  Primary 1 mass: {self.m1:.3e} kg")
        print(f"  Primary 2 mass: {self.m2:.3e} kg")
        print(f"  Primary 1 radius: {self.r1:.3e} m")
        print(f"  Primary 2 radius: {self.r2:.3e} m")
        print(f"  Total mass: {self.m_total:.3e} kg")
        print(f"  Distance: {self.d:.3e} m ({self.d/1e3:.1f} km)")
        print(f"  Mass parameter μ: {self.mu:.6f}")
        print(f"\nCharacteristic scales:")
        print(f"  Length (l*): {self.l_star:.3e} m ({self.l_star/1e3:.1f} km)")
        print(f"  Time (t*): {self.t_star:.3e} s ({self.t_star/86400:.3f} days)")
        print(f"  Velocity (v*): {self.v_star:.3e} m/s ({self.v_star/1e3:.3f} km/s)")
        print(f"  Acceleration (a*): {self.a_star:.3e} m/s^2")
        print(f"  Period: {2*np.pi*self.t_star/86400:.3f} days")
    
    # ========================================================================
    # UNIT CONVERSIONS
    # ========================================================================
    
    def position_to_normalized(self, r_SI):
        """Convert position from SI (m) to normalized units."""
        return position_to_normalized(r_SI, self.l_star)
    
    def position_to_SI(self, r_norm):
        """Convert position from normalized units to SI (m)."""
        return position_to_SI(r_norm, self.l_star)
    
    def velocity_to_normalized(self, v_SI):
        """Convert velocity from SI (m/s) to normalized units."""
        return velocity_to_normalized(v_SI, self.v_star)
    
    def velocity_to_SI(self, v_norm):
        """Convert velocity from normalized units to SI (m/s)."""
        return velocity_to_SI(v_norm, self.v_star)
    
    def time_to_normalized(self, t_SI):
        """Convert time from SI (s) to normalized units."""
        return time_to_normalized(t_SI, self.t_star)
    
    def time_to_SI(self, t_norm):
        """Convert time from normalized units to SI (s)."""
        return time_to_SI(t_norm, self.t_star)
    
    def state_to_normalized(self, state_SI):
        """Convert state vector from SI to normalized units."""
        return state_to_normalized(state_SI, self.l_star, self.v_star)
    
    def state_to_SI(self, state_norm):
        """Convert state vector from normalized to SI units."""
        return state_to_SI(state_norm, self.l_star, self.v_star)
    
    def trajectory_to_normalized(self, states_SI):
        """Convert trajectory from SI to normalized units."""
        return trajectory_to_normalized(states_SI, self.l_star, self.v_star)
    
    def trajectory_to_SI(self, states_norm):
        """Convert trajectory from normalized to SI units."""
        return trajectory_to_SI(states_norm, self.l_star, self.v_star)
    
    def delta_v_to_normalized(self, dv_SI):
        """Convert delta-v from SI (m/s) to normalized units."""
        return self.velocity_to_normalized(dv_SI)
    
    def delta_v_to_SI(self, dv_norm):
        """Convert delta-v from normalized units to SI (m/s)."""
        return self.velocity_to_SI(dv_norm)
    
    # ========================================================================
    # FRAME CONVERSIONS
    # ========================================================================
    
    def rotating_to_inertial(self, state_rot, t):
        """Convert state from rotating to inertial frame."""
        return rotating_to_inertial(state_rot, t, self.mu)
    
    def inertial_to_rotating(self, state_iner, t):
        """Convert state from inertial to rotating frame."""
        return inertial_to_rotating(state_iner, t, self.mu)
    
    def trajectory_rotating_to_inertial(self, states_rot, times):
        """Convert trajectory from rotating to inertial frame."""
        return convert_trajectory_to_inertial(states_rot, times, self.mu)
    
    def trajectory_inertial_to_rotating(self, states_iner, times):
        """Convert trajectory from inertial to rotating frame."""
        return convert_trajectory_to_rotating(states_iner, times, self.mu)
    
    # ========================================================================
    # DYNAMICS
    # ========================================================================
    
    def eom(self, t, state):
        """See dynamics.py"""
        return cr3bp_eom(t, state, self.mu)
    
    def solve(self, state0, t_span, t_eval=None, rtol=1e-12, atol=1e-12,
              dense_output=False, events=None):
        """See dynamics.py"""
        return solve_CR3BP(state0, t_span, self.mu, t_eval=t_eval,
                          rtol=rtol, atol=atol, dense_output=dense_output,
                          events=events)


# ============================================================================
# PRE-CONFIGURED SYSTEMS
# ============================================================================

def create_earth_moon_system():
    """Create Earth-Moon CR3BP system."""
    m_earth = 5.972e24  # kg
    m_moon = 7.342e22   # kg
    d_em = 384400e3     # m (average distance)
    r_earth = 6371e3    # m
    r_moon = 1737e3     # m
    return CR3BPSystem(m_earth, m_moon, r_earth, r_moon, d_em)