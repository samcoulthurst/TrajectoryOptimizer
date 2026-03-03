"""
grid_search.py

This module contains the grid search functions to optimize the trajectories.

"""

import numpy as np
import pandas as pd
from .objective import evaluate

def grid_search_method(cr3bp, grid_size, dec_var_ranges, tol, leo_alt_m, lmo_alt_m, print_intermediates=True):
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
    num_theta = grid_size[0]
    num_delta_v = grid_size[1]
    num_delta_v_angle = grid_size[2]
    num_tof = grid_size[3]
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
                    try:
                        distance_rocket_lmo, total_delta_v = evaluate(cr3bp, x0, leo_alt_m, lmo_alt_m)
                    except: # Catch any integration errors or other issues in evaluation
                        print(f"Error evaluating grid point: Theta={theta:.2f} rad, Delta_v={delta_v:.2f}, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s. Skipping.")
                        #feasible_solutions.append({
                        #    'theta': theta,
                        #    'delta_v': delta_v,
                        #    'delta_v_angle': delta_v_angle,
                        #    'tof': tof,
                        #    'total_delta_v': np.nan,
                        #    'distance_to_lmo': np.nan
                        #})
                        continue

                    if abs(distance_rocket_lmo) < min_constraint:
                        min_constraint = distance_rocket_lmo
                        min_constraint_x0 = [theta, delta_v, delta_v_angle, tof]

                    if abs(distance_rocket_lmo) <= tol:
                        print(f"Iteration: {iteration+1}/{total_iterations}") if print_intermediates else None
                        feasible_solutions.append({
                            'theta': theta,
                            'delta_v': delta_v,
                            'delta_v_angle': delta_v_angle,
                            'tof': tof,
                            'total_delta_v': total_delta_v,
                            'distance_to_lmo': distance_rocket_lmo
                        })
                        np.save("grid_search_results_backup.npy", feasible_solutions) # Backup results after each feasible solution found
                        print(f"Grid point satisfies constraint: Theta={theta:.2f} rad, Delta_v={delta_v:.2f}, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s => Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km, Total Delta_v={total_delta_v*cr3bp.v_star*1e-3:.2f} km/s") if print_intermediates else None
                        
                        if total_delta_v < min_delta_v:
                            min_delta_v = total_delta_v
                            print(f"New optimal found: Delta_v={delta_v*cr3bp.v_star*1e-3:.2f} km/s, Theta={theta:.2f} rad, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s, Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km")
                    #else:
                        #feasible_solutions.append({
                        #    'theta': theta,
                        #    'delta_v': delta_v,
                        #    'delta_v_angle': delta_v_angle,
                        #    'tof': tof,
                        #    'total_delta_v': np.nan,
                        #    'distance_to_lmo': np.nan
                        #})
                    
                        

    results_df = pd.DataFrame(feasible_solutions)
    if not results_df.empty:
        results_df = results_df.sort_values('total_delta_v').reset_index(drop=True)
        print(f"\nFound {len(results_df)} feasible solutions.")
    else:
        print(f"\nNo feasible solutions found. Closest constraint value: {min_constraint*cr3bp.l_star*1e-3:.2f} km at {min_constraint_x0}")

    return results_df

def shrink_ranges(optimals, dec_var_ranges, shrink_factor=0.5):
    new_ranges = []
    for i, optimal in enumerate(optimals):
        current_range = dec_var_ranges[i][1] - dec_var_ranges[i][0]
        lower_bound = max(optimal - (current_range * shrink_factor / 2), dec_var_ranges[i][0])
        upper_bound = min(optimal + (current_range * shrink_factor / 2), dec_var_ranges[i][1])
        new_ranges.append([lower_bound, upper_bound])
    return new_ranges

def sweep_1d(cr3bp, var, delta, grid_size, optimal_params, leo_alt_m, lmo_alt_m, tol=3e-6):
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    idx = var_names.index(var)
    
    ranges = [[opt, opt] for opt in optimal_params]
    ranges[idx] = [optimal_params[idx] - delta, optimal_params[idx] + delta]
    
    grid = [1, 1, 1, 1]
    grid[idx] = grid_size
    
    return grid_search_method(cr3bp, tuple(grid), ranges, tol, leo_alt_m, lmo_alt_m)

def sweep_2d(cr3bp, var1, delta1, var2, delta2, grid_size, optimal_params, leo_alt_m, lmo_alt_m, tol=3e-6):
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    idx1 = var_names.index(var1)
    idx2 = var_names.index(var2)
    
    ranges = [[opt, opt] for opt in optimal_params]
    ranges[idx1] = [optimal_params[idx1] - delta1, optimal_params[idx1] + delta1]
    ranges[idx2] = [optimal_params[idx2] - delta2, optimal_params[idx2] + delta2]
    
    grid = [1, 1, 1, 1]
    grid[idx1] = grid_size
    grid[idx2] = grid_size
    
    return grid_search_method(cr3bp, tuple(grid), ranges, tol, leo_alt_m, lmo_alt_m)