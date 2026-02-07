"""
grid_search.py

This module contains the grid search functions to optimize the trajectories.

"""

import numpy as np
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
    optimal_delta_v : float
        Optimal delta v found in the grid search
    """

    # Create grid of decision variables
    num_theta = 20
    num_delta_v = 20
    num_delta_v_angle = 10
    num_tof = 20
    print(f"Performing grid search with {num_theta} x {num_delta_v} x {num_delta_v_angle} x {num_tof} = {num_theta*num_delta_v*num_delta_v_angle*num_tof} grid points...")
    #print(f"Estimated time is {num_theta*num_delta_v*num_delta_v_angle*num_tof / 2 / 60:.2f} minutes (assuming 1s per evaluation)")
    theta_range = np.linspace(dec_var_ranges[0][0], dec_var_ranges[0][1], num_theta)
    delta_v_range = np.linspace(dec_var_ranges[1][0], dec_var_ranges[1][1], num_delta_v)
    delta_v_angle_range = np.linspace(dec_var_ranges[2][0], dec_var_ranges[2][1], num_delta_v_angle)
    tof_range = np.linspace(dec_var_ranges[3][0], dec_var_ranges[3][1], num_tof)

    optimal_delta_v = None
    min_delta_v = np.inf
    min_constraint = np.inf
    min_constraint_x0 = None
    iteration = 0
    total_iterations = num_theta * num_delta_v * num_delta_v_angle * num_tof
    # Iterate over the grid
    for itheta, theta in enumerate(theta_range):
        for idelta_v, delta_v in enumerate(delta_v_range):
            for idelta_v_angle, delta_v_angle in enumerate(delta_v_angle_range):
                for itof, tof in enumerate(tof_range):
                    print(f"Iteration: {iteration+1}/{total_iterations}") if print_intermediates else None
                    iteration += 1
                    x0 = [theta, delta_v, delta_v_angle, tof]
                    distance_rocket_lmo, total_delta_v = evaluate(cr3bp, x0, leo_alt_m, lmo_alt_m)
                    print(f"Evaluating grid point: Theta={theta:.2f} rad, Delta_v={delta_v:.2f}, Delta_v_angle={delta_v_angle:.2f} rad, TOF={tof:.2f} s => Distance to LMO={distance_rocket_lmo*cr3bp.l_star*1e-3:.2f} km, Total Delta_v={total_delta_v*cr3bp.v_star*1e-3:.2f} km/s") if print_intermediates else None
                    # Track minimum constraint value
                    if distance_rocket_lmo < min_constraint:
                        min_constraint = distance_rocket_lmo
                        min_constraint_x0 = [theta, delta_v, delta_v_angle, tof]

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
        print(f"Minimum constraint value: {min_constraint*cr3bp.l_star*1e-3:.2f} km (tolerance: {tol*cr3bp.l_star*1e-3:.2f} km)")
        print(f"Best constraint point: Theta={min_constraint_x0[0]:.2f} rad, Delta_v={min_constraint_x0[1]:.2f}, Delta_v_angle={min_constraint_x0[2]:.2f} rad, TOF={min_constraint_x0[3]:.2f} s")
        return np.array(min_constraint_x0), min_constraint
    else:
        return np.array([optimal_theta, optimal_delta_v, optimal_delta_v_angle, optimal_tof]),

