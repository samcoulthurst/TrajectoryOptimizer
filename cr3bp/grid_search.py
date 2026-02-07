"""
Grid Search for LEO to LMO Transfer Initial Guess

Performs a coarse grid search over departure angle, time of flight, and
delta-v magnitude to find a good initial guess for the optimizer.

Uses LEOtoLMOProblem to evaluate objective (total delta-v) and constraint
(LMO altitude arrival) at each grid point.
"""

import numpy as np
from .leo_lmo_optimizer import (
    LEOtoLMOProblem,
    leo_state_rotating,
    L_STAR_M, MU_EM
)

def grid_search_initial_guess(
    problem=None,
    constraint_tol=0.01,
    leo_alt_m=463e3,
    lmo_alt_m=100e3,
    n_theta=32,
    n_tof=32,
    n_dv_mag=32,
    theta_range=(1.4*np.pi, 1.75*np.pi),
    tof_range=(1.0, 3.0),
    dv_mag_range=(3.3, 4.2),
    verbose=True
):
    """
    Grid search to find a good initial guess for the LEO to LMO transfer.

    Minimizes total delta-v (objective) among points where the constraint
    violation is within tolerance. Falls back to the least-infeasible point
    if no feasible points are found.

    Parameters
    ----------
    problem : LEOtoLMOProblem, optional
        Problem instance. If None, one is created from leo_alt_m/lmo_alt_m.
    constraint_tol : float
        Maximum allowable constraint violation in normalized units
        (default: 0.01, ~3,844 km). Points within this tolerance are
        considered feasible.
    leo_alt_m : float
        LEO altitude in meters (default: 463 km)
    lmo_alt_m : float
        LMO altitude in meters (default: 100 km)
    n_theta : int
        Number of departure angles to try
    n_tof : int
        Number of time-of-flight values to try
    n_dv_mag : int
        Number of delta-v magnitudes to try
    theta_range : tuple
        Range of departure angles (radians)
    tof_range : tuple
        Range of TOF values (normalized time)
    dv_mag_range : tuple
        Range of delta-v magnitudes (normalized)
    verbose : bool
        Print progress and results

    Returns
    -------
    x0 : ndarray, shape (5,)
        Best initial guess [theta, dv1_x, dv1_y, dv1_z, T]
    best_obj : float
        Total delta-v at best point (normalized, |dv1| + |dv2|)
    best_con : float
        Constraint violation at best point (normalized, |r_arrival - r_LMO|)
    """
    mu = MU_EM

    if problem is None:
        problem = LEOtoLMOProblem(
            mu=mu,
            leo_alt_m=leo_alt_m,
            lmo_alt_m=lmo_alt_m,
            print_iter=False
        )

    # Create grid
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta, endpoint=False)
    tofs = np.linspace(tof_range[0], tof_range[1], n_tof)
    dv_mags = np.linspace(dv_mag_range[0], dv_mag_range[1], n_dv_mag)

    total_evals = n_theta * n_tof * n_dv_mag
    if verbose:
        print(f"Grid search: {n_theta} angles x {n_tof} TOFs x {n_dv_mag} dv_mags = {total_evals} evaluations")
        print(f"  Theta range: [{np.rad2deg(theta_range[0]):.0f}, {np.rad2deg(theta_range[1]):.0f}] deg")
        print(f"  TOF range: [{tof_range[0]:.2f}, {tof_range[1]:.2f}] normalized")
        print(f"  DV magnitude range: [{dv_mag_range[0]:.2f}, {dv_mag_range[1]:.2f}] normalized")
        print(f"  Constraint tolerance: {constraint_tol:.4f} normalized ({constraint_tol * L_STAR_M / 1000:.0f} km)")

    best_x0 = None
    best_obj = np.inf
    best_con = np.inf
    found_feasible = False
    eval_count = 0

    for theta in thetas:
        # Get initial state and velocity direction for prograde burn
        state0_base = leo_state_rotating(theta, leo_alt_m, mu)
        v_dir = state0_base[3:6] / np.linalg.norm(state0_base[3:6])

        for dv_mag in dv_mags:
            # Compute delta-v vector (prograde)
            dv1 = dv_mag * v_dir

            for tof in tofs:
                eval_count += 1
                x = np.array([theta, dv1[0], dv1[1], dv1[2], tof])

                try:
                    obj, con = problem.evaluate(x)
                    con_viol = abs(con)

                    if con_viol < constraint_tol:
                        # Feasible point: pick lowest objective
                        if not found_feasible or obj < best_obj:
                            best_x0 = x
                            best_obj = obj
                            best_con = con_viol
                            found_feasible = True
                    else:
                        # Infeasible: track least-infeasible as fallback
                        if not found_feasible and con_viol < best_con:
                            best_x0 = x
                            best_obj = obj
                            best_con = con_viol

                    if verbose and eval_count % 500 == 0:
                        status = "FEASIBLE" if found_feasible else "infeasible"
                        print(f"  [{eval_count}/{total_evals}] Best: obj={best_obj:.4f}, "
                              f"con={best_con * L_STAR_M / 1000:.1f} km [{status}]")

                except Exception:
                    # Integration failed (trajectory crashed, etc.)
                    pass

    if verbose:
        if best_x0 is not None:
            con_km = best_con * L_STAR_M / 1000
            status = "FEASIBLE" if found_feasible else "INFEASIBLE (fallback)"
            print(f"\nGrid search complete! [{status}]")
            print(f"  Best theta: {np.rad2deg(best_x0[0]):.2f} deg")
            print(f"  Best dv1 magnitude: {np.linalg.norm(best_x0[1:4]):.4f} normalized")
            print(f"  Best TOF: {best_x0[4]:.4f} normalized")
            print(f"  Objective (total dv): {best_obj:.6f} normalized")
            print(f"  Constraint violation: {con_km:.2f} km")
        else:
            print("\nGrid search failed to find any valid trajectory!")

    return best_x0, best_obj, best_con



def optimize_with_grid_search(
    problem=None,
    constraint_tol=0.01,
    leo_alt_m=463e3,
    lmo_alt_m=100e3,
    n_theta=6,
    n_tof=8,
    n_dv_mag=4,
    print_iter=True,
    max_iter=200
):
    """
    Run grid search followed by IPOPT optimization.

    Parameters
    ----------
    problem : LEOtoLMOProblem, optional
        Problem instance. If None, one is created from leo_alt_m/lmo_alt_m.
    constraint_tol : float
        Maximum allowable constraint violation for grid search (normalized)
    leo_alt_m : float
        LEO altitude in meters
    lmo_alt_m : float
        LMO altitude in meters
    n_theta, n_tof, n_dv_mag : int
        Grid search resolution
    print_iter : bool
        Print IPOPT iteration statistics
    max_iter : int
        Maximum IPOPT iterations

    Returns
    -------
    x0 : ndarray, shape (5,)
        Best initial guess from grid search
    best_obj : float
        Total delta-v at best point (normalized)
    best_con : float
        Constraint violation at best point (normalized)
    """
    mu = MU_EM

    if problem is None:
        problem = LEOtoLMOProblem(
            mu=mu,
            leo_alt_m=leo_alt_m,
            lmo_alt_m=lmo_alt_m,
            print_iter=print_iter
        )

    print("=" * 60)
    print("PHASE 1: Grid Search for Initial Guess")
    print("=" * 60)

    x0, best_obj, best_con = grid_search_initial_guess(
        problem=problem,
        constraint_tol=constraint_tol,
        leo_alt_m=leo_alt_m,
        lmo_alt_m=lmo_alt_m,
        n_theta=n_theta,
        n_tof=n_tof,
        n_dv_mag=n_dv_mag,
        verbose=True
    )

    if x0 is None:
        print("\nGrid search failed! Using default initial guess.")
        x0 = None

    print("\n" + "=" * 60)
    print("PHASE 2: IPOPT Optimization")
    print("=" * 60)

    #result = optimize_leo_to_lmo(
    #    leo_alt_m=leo_alt_m,
    #    lmo_alt_m=lmo_alt_m,
    #    x0=x0,
    #    print_iter=print_iter,
    #    max_iter=max_iter
    #)

    return x0, best_obj, best_con


if __name__ == "__main__":
    result = optimize_with_grid_search()
