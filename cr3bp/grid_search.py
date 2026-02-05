"""
Grid Search for LEO to LMO Transfer Initial Guess

Performs a coarse grid search over departure angle, time of flight, and
delta-v magnitude to find a good initial guess for the optimizer.
"""

import numpy as np
from .dynamics import solve_CR3BP_with_STM
from .leo_lmo_optimizer import (
    leo_state_rotating,
    L_STAR_M, R_MOON_M, MU_EM
)

def grid_search_initial_guess(
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

    Parameters
    ----------
    leo_alt_m : float
        LEO altitude in meters (default: 463 km)
    lmo_alt_m : float
        LMO altitude in meters (default: 100 km)
    n_theta : int
        Number of departure angles to try (default: 6)
    n_tof : int
        Number of time-of-flight values to try (default: 8)
    n_dv_mag : int
        Number of delta-v magnitudes to try (default: 4)
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
    miss_distance : float
        Miss distance from target LMO altitude (normalized)
    """
    mu = MU_EM
    r_LMO = (R_MOON_M + lmo_alt_m) / L_STAR_M

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

    best_x0 = None
    best_miss = np.inf
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

                # Apply delta-v and propagate
                state0 = state0_base.copy()
                state0[3:6] += dv1

                try:
                    states, times, stm, sol = solve_CR3BP_with_STM(
                        state0, (0, tof), mu, rtol=1e-10, atol=1e-10
                    )
                    state_f = states[:, -1]

                    # Compute miss distance from LMO altitude
                    r_rel = state_f[:3] - np.array([1 - mu, 0, 0])
                    r_arrival = np.linalg.norm(r_rel)
                    miss = abs(r_arrival - r_LMO)

                    if miss < best_miss:
                        best_miss = miss
                        best_x0 = np.array([theta, dv1[0], dv1[1], dv1[2], tof])

                        if verbose and eval_count % 50 == 0:
                            miss_km = miss * L_STAR_M / 1000
                            print(f"  [{eval_count}/{total_evals}] New best: miss = {miss_km:.1f} km")

                except Exception:
                    # Integration failed (trajectory crashed, etc.)
                    pass

    if verbose:
        if best_x0 is not None:
            miss_km = best_miss * L_STAR_M / 1000
            print(f"\nGrid search complete!")
            print(f"  Best theta: {np.rad2deg(best_x0[0]):.2f} deg")
            print(f"  Best dv1 magnitude: {np.linalg.norm(best_x0[1:4]):.4f} normalized")
            print(f"  Best TOF: {best_x0[4]:.4f} normalized")
            print(f"  Miss distance: {miss_km:.2f} km")
        else:
            print("\nGrid search failed to find any valid trajectory!")

    return best_x0, best_miss



def optimize_with_grid_search(
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
    result : dict
        Optimization results (same as optimize_leo_to_lmo)
    """
    # Import here to avoid circular import
    #from .leo_lmo_optimizer import optimize_leo_to_lmo

    print("=" * 60)
    print("PHASE 1: Grid Search for Initial Guess")
    print("=" * 60)

    x0, miss = grid_search_initial_guess(
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

    return x0, miss


if __name__ == "__main__":
    result = optimize_with_grid_search()
