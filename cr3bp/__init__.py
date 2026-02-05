"""
CR3BP - Circular Restricted Three-Body Problem Package

"""

from .dynamics import cr3bp_eom_with_stm, solve_CR3BP_with_STM, circular_orbit_state_earth
from .conversions import (
    rotating_to_inertial,
    inertial_to_rotating,
    convert_trajectory_to_inertial,
    convert_trajectory_to_rotating,
    state_to_normalized,
    state_to_SI,
    trajectory_to_normalized,
    trajectory_to_SI
)
from .plotting import plot_trajectory
from .system import CR3BPSystem, create_earth_moon_system, create_sun_earth_system
from .leo_lmo_optimizer import optimize_leo_to_lmo
from .grid_search import grid_search_initial_guess, optimize_with_grid_search

__version__ = '0.1.0'

__all__ = [
    # Dynamics
    'cr3bp_eom_with_stm',
    'solve_CR3BP_with_STM',
    'circular_orbit_state_earth',

    # Frame conversions
    'rotating_to_inertial',
    'inertial_to_rotating',
    'convert_trajectory_to_inertial',
    'convert_trajectory_to_rotating',

    # Unit conversions
    'state_to_normalized',
    'state_to_SI',
    'trajectory_to_normalized',
    'trajectory_to_SI',

    # System
    'CR3BPSystem',
    'create_earth_moon_system',
    'create_sun_earth_system',

    # Plotting
    'plot_trajectory',

    # Optimization
    'optimize_leo_to_lmo',
    'grid_search_initial_guess',
    'optimize_with_grid_search',
]