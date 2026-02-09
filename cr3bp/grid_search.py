"""
grid_search.py

This module contains the grid search functions to optimize the trajectories.

"""

import numpy as np
import pandas as pd
from .objective import evaluate

def grid_search_method(cr3bp, dec_var_ranges, tol, leo_alt_m, lmo_alt_m, print_intermediates=True):
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
    results_df : pd.DataFrame
        Table of all feasible solutions with decision variables and objective values
    """

    # Create grid of decision variables
    num_theta = 20
    num_delta_v = 20
    num_delta_v_angle = 10
    num_tof = 20
    print(f"Performing grid search with {num_theta} x {num_delta_v} x {num_delta_v_angle} x {num_tof} = {num_theta*num_delta_v*num_delta_v_angle*num_tof} grid points...")
    theta_range = np.linspace(dec_var_ranges[0][0], dec_var_ranges[0][1], num_theta)
    delta_v_range = np.linspace(dec_var_ranges[1][0], dec_var_ranges[1][1], num_delta_v)
    delta_v_angle_range = np.linspace(dec_var_ranges[2][0], dec_var_ranges[2][1], num_delta_v_angle)
    tof_range = np.linspace(dec_var_ranges[3][0], dec_var_ranges[3][1], num_tof)

    feasible_solutions = []
    min_delta_v = np.inf
    min_constraint = np.inf
    min_constraint_x0 = None
    iteration = 0
    total_iterations = num_theta * num_delta_v * num_delta_v_angle * num_tof

    for theta in theta_range:
        for delta_v in delta_v_range:
            for delta_v_angle in delta_v_angle_range:
                for tof in tof_range:
                    
                    iteration += 1
                    x0 = [theta, delta_v, delta_v_angle, tof]
                    distance_rocket_lmo, total_delta_v = evaluate(cr3bp, x0, leo_alt_m, lmo_alt_m)
                    

                    if distance_rocket_lmo < min_constraint:
                        min_constraint = distance_rocket_lmo
                        min_constraint_x0 = [theta, delta_v, delta_v_angle, tof]

                    if distance_rocket_lmo <= tol:
                        print(f"Iteration: {iteration+1}/{total_iterations}") if print_intermediates else None
                        feasible_solutions.append({
                            'theta': theta,
                            'delta_v': delta_v,
                            'delta_v_angle': delta_v_angle,
                            'tof': tof,
                            'total_delta_v': total_delta_v,
                            'distance_to_lmo': distance_rocket_lmo
                        })
                        print(f"Grid point satisfies constraint: Theta={theta:.2f} rad, Delta_v={delta_v:.2f}, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s => Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km, Total Delta_v={total_delta_v*cr3bp.v_star*1e-3:.2f} km/s") if print_intermediates else None

                        if total_delta_v < min_delta_v:
                            min_delta_v = total_delta_v
                            print(f"New optimal found: Delta_v={delta_v*cr3bp.v_star*1e-3:.2f} km/s, Theta={theta:.2f} rad, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s, Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km")

    results_df = pd.DataFrame(feasible_solutions)
    if not results_df.empty:
        results_df = results_df.sort_values('total_delta_v').reset_index(drop=True)
        print(f"\nFound {len(results_df)} feasible solutions.")
    else:
        print(f"\nNo feasible solutions found. Closest constraint value: {min_constraint*cr3bp.l_star*1e-3:.2f} km at {min_constraint_x0}")

    return results_df

