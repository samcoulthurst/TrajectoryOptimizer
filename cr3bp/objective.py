"""
CR3BP Objective

This module contains the objective and constraint functions for CR3BP trajectories.

"""
import numpy as np
from .dynamics import decision_to_state0, v_llo_at_rocket_pos

def evaluate(cr3bp, x0, leo_alt_m, lmo_alt_m):
    """
    Function which evaluates both the constraint and the objective for a given function:
    constraint - distance from idealized 100km llo
    objective - total delta_v required to make the transition
    
    Parameters
    ----------
    cr3bp : Class

    x0 : float
        Decision variables - (theta, delta_v, delta_v_angle, tof)

    Return
    ------
    distance_rocket_lmo : float
        Distance from rocket to lmo radius in normalized units
    Total Delta_v: float
        Total delta v needed
    """

    ### Run the integration:
    state0 = decision_to_state0(cr3bp, x0, leo_alt_m)

    tof = x0[3]
    t_span = (0,tof)
    t_eval = np.linspace(0, tof, 1000)
    states, _, _ = cr3bp.solve(state0, t_span, t_eval)

    state_f = states[:,-1]

    ### Constraint #######################################
    final_pos = states[:,-1][0:3]
    
    #print(f'Final Rocket Pos {cr3bp.l_star * final_pos}')
    moon_pos = np.array([1-cr3bp.mu, 0, 0])
    #print(f'Final Moon Pos {cr3bp.l_star * moon_pos}')

    distance_moon_rocket = np.linalg.norm(final_pos-moon_pos)
    distance_rocket_lmo = distance_moon_rocket - (lmo_alt_m/cr3bp.l_star)


    ### Objective ##################################
    delta_v1 = x0[1]
    state_f_inertial = cr3bp.rotating_to_inertial(state_f,tof)

    moon_f = np.array([1-cr3bp.mu,0,0,0,0,0])
    moon_f_inertial = cr3bp.rotating_to_inertial(moon_f,tof)

    v_rocket = state_f_inertial[3:]
    v_llo = v_llo_at_rocket_pos(cr3bp, state_f_inertial, moon_f_inertial)

    delta_v2 = np.linalg.norm(v_llo - v_rocket)

    return distance_rocket_lmo, delta_v1+delta_v2

def constraint(cr3bp, x0, leo_alt_m, lmo_alt_m):
    """
    Constraint function which determines the rocket is near the moon.
    
    Parameters
    ----------
    cr3bp : Class

    x0 : float
        Decision variables - (theta, delta_v, delta_v_angle, tof)

    Return
    ------
    distance_rocket_lmo : float
        Distance from rocket to lmo radius in normalized units
    """

    state0 = decision_to_state0(cr3bp, x0, leo_alt_m)

    tof = x0[3]
    t_span = (0,tof)
    t_eval = np.linspace(0, tof, 1000)
    states, _, _ = cr3bp.solve(state0, t_span, t_eval)

    final_pos = states[:,-1][0:3]
    
    print(f'Final Rocket Pos {cr3bp.l_star * final_pos}')
    moon_pos = np.array([1-cr3bp.mu, 0, 0])
    print(f'Final Moon Pos {cr3bp.l_star * moon_pos}')

    distance_moon_rocket = np.linalg.norm(final_pos-moon_pos)
    distance_rocket_lmo = distance_moon_rocket - (lmo_alt_m/cr3bp.l_star)
    return distance_rocket_lmo


def objective_func(cr3bp, x0, leo_alt_m):
    """
    Objective function which determines the total delta v needed to go from leo to lmo.
    
    Parameters
    ----------
    cr3bp : Class

    x0 : float
        Decision variables - (theta, delta_v, delta_v_angle, tof)

    Return
    ------
    Total Delta_v: float
        Total delta v needed
    """

    delta_v1 = x0[1]
    
    state0 = decision_to_state0(cr3bp, x0, leo_alt_m)

    tof = x0[3]
    t_span = (0,tof)
    t_eval = np.linspace(0, tof, 1000)
    states, _, _ = cr3bp.solve(state0, t_span, t_eval)

    state_f = states[:,-1]
    state_f_inertial = cr3bp.rotating_to_inertial(state_f,tof)

    moon_f = np.array([1-cr3bp.mu,0,0,0,0,0])
    moon_f_inertial = cr3bp.rotating_to_inertial(moon_f,tof)

    v_rocket = state_f_inertial[3:]
    v_llo = v_llo_at_rocket_pos(cr3bp, state_f_inertial, moon_f_inertial)

    delta_v2 = np.linalg.norm(v_llo - v_rocket)

    return delta_v1 + delta_v2





