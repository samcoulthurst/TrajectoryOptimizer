"""
grid_search.py

This module contains the grid search functions to optimize the trajectories.

"""

import numpy as np
from .objective import evaluate

def grid_search_method(cr3bp, dec_var_ranges, tol, leo_alt_m, lmo_alt_m):
    """
    Function which performs the grid search algorithm to minimise the objective
    function whilst satisfying the constraint function.
    
    Parameters
    ----------
    cr3bp : Class

    dec_var_ranges : array 4,2
        Ranges of the decision variables - [[theta_min,theta_max],
                                            [delta_v_min,delta_v_max],
                                            [delta_v_angle_min, delta_v_angle_max],
                                            [tof_min, tof_max]]
    tol : float
        Tolerance for constraint satisfaction in normalized units

    Return
    ------
    optimal_delta_v : float
        Optimal delta v found in the grid search
    """

    # Create grid of decision variables
    num_theta = 5
    num_delta_v = 5
    num_delta_v_angle = 5
    num_tof = 5
    print(f"Performing grid search with {num_theta} x {num_delta_v} x {num_delta_v_angle} x {num_tof} = {num_theta*num_delta_v*num_delta_v_angle*num_tof} grid points...")
    print(f"Estimated time is {num_theta*num_delta_v*num_delta_v_angle*num_tof / 2 / 60:.2f} minutes (assuming 1s per evaluation)")
    theta_range = np.linspace(dec_var_ranges[0][0], dec_var_ranges[0][1], num_theta)
    delta_v_range = np.linspace(dec_var_ranges[1][0], dec_var_ranges[1][1], num_delta_v)
    delta_v_angle_range = np.linspace(dec_var_ranges[2][0], dec_var_ranges[2][1], num_delta_v_angle)
    tof_range = np.linspace(dec_var_ranges[3][0], dec_var_ranges[3][1], num_tof)

    optimal_delta_v = None
    min_delta_v = np.inf

    # Iterate over the grid
    for itheta, theta in enumerate(theta_range):
        for idelta_v, delta_v in enumerate(delta_v_range):
            for idelta_v_angle, delta_v_angle in enumerate(delta_v_angle_range):
                for itof, tof in enumerate(tof_range):
                    print(f"Iteration: {itheta*num_delta_v+num_delta_v_angle*num_tof + idelta_v*num_delta_v_angle*num_tof + idelta_v_angle*num_tof + itof + 1}/{num_theta*num_delta_v*num_delta_v_angle*num_tof}")
                    x0 = [theta, delta_v, delta_v_angle, tof]
                    distance_rocket_lmo, total_delta_v = evaluate(cr3bp, x0, leo_alt_m, lmo_alt_m)
                    print(f"Evaluating grid point: Theta={theta:.2f} rad, Delta_v={delta_v:.2f}, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s => Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km, Total Delta_v={total_delta_v*cr3bp.v_star*1e-3:.2f} km/s")
                    # Check if constraint is satisfied and if objective is improved
                    if distance_rocket_lmo <= tol and total_delta_v < min_delta_v:
                        min_delta_v = total_delta_v
                        optimal_delta_v = delta_v
                        optimal_theta = theta
                        optimal_delta_v_angle = delta_v_angle
                        optimal_tof = tof
                        print(f"New optimal found: Delta_v={optimal_delta_v*cr3bp.v_star*1e-3:.2f} km/s, Theta={optimal_theta:.2f} rad, Delta_v_angle={optimal_delta_v_angle:.2f} rad, TOF={optimal_tof:.2f} s, Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km")
    
    if optimal_delta_v is None:
        print("No feasible solution found within the specified ranges and tolerance.")
        return None
    else:
        return np.array([optimal_theta, optimal_delta_v, optimal_delta_v_angle, optimal_tof])

