"""
Optimization subpackage for CR3BP trajectory design.

This module provides tools for optimizing transfer trajectories
in the Circular Restricted Three-Body Problem using single shooting
and nonlinear programming (IPOPT).

Classes
-------
SingleShootingProblem
    Single shooting formulation for bi-impulsive transfers
LEOtoLMOProblem
    cyipopt-compatible NLP interface

Functions
---------
circular_orbit_state_earth
    Compute state on circular Earth orbit
circular_orbit_state_moon
    Compute state on circular Moon orbit
compute_lmo_insertion_dv
    Compute delta-V for LMO insertion
create_nlp_problem
    Factory function for cyipopt problem
solve_transfer
    Convenience function to solve transfer problem
"""

from .orbits import (
    circular_orbit_state_earth,
    circular_orbit_state_moon,
    compute_lmo_insertion_dv,
    L_STAR_KM,
    R_EARTH_KM,
    R_MOON_KM
)
from .shooting import SingleShootingProblem
from .nlp_interface import (
    LEOtoLMOProblem,
    create_nlp_problem,
    solve_transfer
)

__all__ = [
    # Orbit utilities
    'circular_orbit_state_earth',
    'circular_orbit_state_moon',
    'compute_lmo_insertion_dv',
    'L_STAR_KM',
    'R_EARTH_KM',
    'R_MOON_KM',
    # Shooting
    'SingleShootingProblem',
    # NLP interface
    'LEOtoLMOProblem',
    'create_nlp_problem',
    'solve_transfer',
]
