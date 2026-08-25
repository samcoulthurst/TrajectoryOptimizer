"""
CR3BP - Circular Restricted Three-Body Problem Package

"""

from .dynamics import cr3bp_eom, solve_CR3BP, decision_to_state0, circular_orbit_state_earth, ciruclar_orbit_trajectory_moon
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
from .visualization import animate_trajectory
from .system import CR3BPSystem, create_earth_moon_system
from .objective import constraint, objective_func, evaluate
from .grid_search import grid_search_method


__version__ = '0.1.0'

__all__ = [
    # Dynamics
    'cr3bp_eom',
    'solve_CR3BP',
    'decision_to_state0',
    'circular_orbit_state_earth',
    'ciruclar_orbit_trajectory_moon',
    
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

    # Plotting
    'plot_trajectory',

    # Animation
    'animate_trajectory',

    # Objective
    'constraint',
    'objective_func',
    'evaluate',

    # Grid Search
    'grid_search_method',
]