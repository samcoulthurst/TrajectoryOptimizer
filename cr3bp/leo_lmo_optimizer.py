"""
LEO to LMO Bi-Impulsive Transfer Optimizer

Minimizes total delta-v for a bi-impulsive transfer from Low Earth Orbit
to Low Moon Orbit using single shooting and IPOPT.

Decision variables: [theta, dv1_x, dv1_y, dv1_z, T]
- theta: departure angle on LEO (radians)
- dv1: departure impulse in rotating frame (normalized)
- T: time of flight (normalized)

Constraint: arrive at LMO altitude above Moon surface
Objective: minimize |dv1| + |dv2|
"""

import numpy as np
import cyipopt
from .dynamics import solve_CR3BP_with_STM, circular_orbit_state_earth
from .conversions import inertial_to_rotating, compute_characteristic_scales


# ============================================================================
# CONSTANTS
# ============================================================================
L_STAR_M = 384400e3      # Earth-Moon distance (m)
R_EARTH_M = 6371e3       # Earth radius (m)
R_MOON_M = 1737e3        # Moon radius (m)
MU_EM = 0.01215058       # Earth-Moon mass parameter


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def leo_state_rotating(theta, altitude_m, mu=MU_EM):
    """
    Compute state on circular LEO in ROTATING frame.

    Parameters
    ----------
    theta : float
        True anomaly / angle on orbit (radians)
    altitude_m : float
        Altitude above Earth surface (meters)
    mu : float
        CR3BP mass parameter

    Returns
    -------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in rotating frame, normalized units
    """
    # Get state in inertial frame using existing function
    state_iner = circular_orbit_state_earth(theta, altitude_m, inclination=0.0, mu=mu)

    # Convert to rotating frame at t=0
    # At t=0, frames are aligned, so position is unchanged
    # Velocity transforms as: v_rot = v_iner + omega x r = v_iner + [y, -x, 0]
    state_rot = inertial_to_rotating(state_iner, t=0.0, mu=mu)

    return state_rot


def leo_state_derivative_theta(theta, altitude_m, mu=MU_EM):
    """
    Compute derivative of LEO state with respect to theta (in rotating frame).

    This is needed for the analytical Jacobian using the STM.

    Parameters
    ----------
    theta : float
        True anomaly / angle on orbit (radians)
    altitude_m : float
        Altitude above Earth surface (meters)
    mu : float
        CR3BP mass parameter

    Returns
    -------
    dstate_dtheta : ndarray, shape (6,)
        Derivative of state vector with respect to theta
    """
    # Normalized orbital radius
    r_earth = R_EARTH_M / L_STAR_M
    r_orbit = r_earth + altitude_m / L_STAR_M

    # Circular velocity magnitude
    v_circ = np.sqrt((1 - mu) / r_orbit)

    # Position derivatives (in orbital plane, Earth-centered)
    # x_orb = r_orbit * cos(theta), y_orb = r_orbit * sin(theta)
    dx_dtheta = -r_orbit * np.sin(theta)
    dy_dtheta = r_orbit * np.cos(theta)
    dz_dtheta = 0.0

    # Velocity derivatives in inertial frame
    # vx_iner = -v_circ * sin(theta), vy_iner = v_circ * cos(theta)
    dvx_iner_dtheta = -v_circ * np.cos(theta)
    dvy_iner_dtheta = -v_circ * np.sin(theta)
    dvz_iner_dtheta = 0.0

    # Convert to rotating frame at t=0:
    # v_rot = v_iner + [y, -x, 0]
    # So: dvx_rot/dtheta = dvx_iner/dtheta + dy/dtheta
    #     dvy_rot/dtheta = dvy_iner/dtheta - dx/dtheta
    dvx_dtheta = dvx_iner_dtheta + dy_dtheta
    dvy_dtheta = dvy_iner_dtheta - dx_dtheta
    dvz_dtheta = dvz_iner_dtheta

    return np.array([dx_dtheta, dy_dtheta, dz_dtheta,
                     dvx_dtheta, dvy_dtheta, dvz_dtheta])


def compute_lmo_insertion_dv(state_f, lmo_altitude_m, mu=MU_EM):
    """
    Compute delta-v required for circular LMO insertion.

    Parameters
    ----------
    state_f : ndarray, shape (6,)
        Arrival state in rotating frame [x, y, z, vx, vy, vz]
    lmo_altitude_m : float
        Target LMO altitude above Moon surface (meters)
    mu : float
        CR3BP mass parameter

    Returns
    -------
    dv2 : ndarray, shape (3,)
        Delta-v vector in rotating frame (normalized)
    dv2_mag : float
        Delta-v magnitude (normalized)
    """
    x, y, z, vx, vy, vz = state_f

    # Position relative to Moon (Moon at [1-mu, 0, 0])
    r_rel = np.array([x - (1 - mu), y, z])
    r_mag = np.linalg.norm(r_rel)
    r_hat = r_rel / r_mag

    # Target LMO radius (normalized)
    r_LMO = (R_MOON_M + lmo_altitude_m) / L_STAR_M

    # Circular velocity at LMO (Moon-centered two-body)
    v_circ = np.sqrt(mu / r_mag)

    # Velocity direction: prograde = perpendicular to position, in orbital plane
    # v_dir = cross(z_hat, r_hat) for prograde
    z_hat = np.array([0.0, 0.0, 1.0])
    v_dir = np.cross(z_hat, r_hat)
    v_dir_mag = np.linalg.norm(v_dir)

    if v_dir_mag < 1e-10:
        # Edge case: polar arrival (r_hat parallel to z)
        v_dir = np.array([0.0, 1.0, 0.0])
    else:
        v_dir = v_dir / v_dir_mag

    # Required velocity in Moon-centered inertial frame
    v_circ_iner = v_circ * v_dir

    # Convert to rotating frame: v_rot = v_iner + [y, -x, 0]
    # (using the full position, not relative position, since rotating frame
    # angular velocity is about barycenter)
    v_required = v_circ_iner.copy()
    v_required[0] += y
    v_required[1] -= x

    # Delta-v = required - actual
    v_arrival = np.array([vx, vy, vz])
    dv2 = v_required - v_arrival
    dv2_mag = np.linalg.norm(dv2)

    return dv2, dv2_mag


# ============================================================================
# NLP PROBLEM CLASS
# ============================================================================

class LEOtoLMOProblem:
    """
    cyipopt-compatible NLP problem for LEO to LMO bi-impulsive transfer.

    Decision variables: x = [theta, dv1_x, dv1_y, dv1_z, T]
    Objective: minimize |dv1| + |dv2|
    Constraint: arrive at LMO altitude
    """

    def __init__(self, mu, leo_alt_m, lmo_alt_m, print_iter=True):
        """
        Initialize the optimization problem.

        Parameters
        ----------
        mu : float
            CR3BP mass parameter
        leo_alt_m : float
            LEO altitude in meters
        lmo_alt_m : float
            LMO altitude in meters
        print_iter : bool
            Whether to print iteration statistics
        """
        self.mu = mu
        self.leo_alt_m = leo_alt_m
        self.lmo_alt_m = lmo_alt_m
        self.print_iter = print_iter

        # Target LMO radius (normalized)
        self.r_LMO = (R_MOON_M + lmo_alt_m) / L_STAR_M

        # Iteration counter
        self.iter_count = 0

    def _propagate(self, x, return_stm=False):
        """
        Propagate trajectory from LEO with given decision variables.

        Parameters
        ----------
        x : ndarray
            Decision variables [theta, dv1_x, dv1_y, dv1_z, T]
        return_stm : bool
            If True, also return the State Transition Matrix

        Returns
        -------
        state_f : ndarray, shape (6,)
            Final state in rotating frame
        stm : ndarray, shape (6, 6), optional
            State Transition Matrix (only if return_stm=True)
        """
        theta, dv1_x, dv1_y, dv1_z, T = x

        # Get initial state on LEO in rotating frame
        state0 = leo_state_rotating(theta, self.leo_alt_m, self.mu)

        # Apply departure impulse
        state0[3] += dv1_x
        state0[4] += dv1_y
        state0[5] += dv1_z

        # Propagate CR3BP
        states, times, stm, sol = solve_CR3BP_with_STM(
            state0, (0, T), self.mu, rtol=1e-12, atol=1e-12
        )

        if return_stm:
            return states[:, -1], stm
        return states[:, -1]  # Final state

    def objective(self, x):
        """Compute total delta-v = |dv1| + |dv2|."""
        theta, dv1_x, dv1_y, dv1_z, T = x

        # First burn magnitude
        dv1 = np.array([dv1_x, dv1_y, dv1_z])
        dv1_mag = np.linalg.norm(dv1)

        # Propagate and compute second burn
        state_f = self._propagate(x)
        _, dv2_mag = compute_lmo_insertion_dv(state_f, self.lmo_alt_m, self.mu)

        return dv1_mag + dv2_mag

    def evaluate(self, x):
        """
        Compute objective and constraint in a single propagation.

        Parameters
        ----------
        x : ndarray
            Decision variables [theta, dv1_x, dv1_y, dv1_z, T]

        Returns
        -------
        obj : float
            Total delta-v = |dv1| + |dv2| (normalized)
        con : float
            Constraint value: |r_arrival - r_moon| - r_LMO (should be 0)
        """
        state_f = self._propagate(x)

        # Objective: total delta-v
        dv1_mag = np.linalg.norm(x[1:4])
        _, dv2_mag = compute_lmo_insertion_dv(state_f, self.lmo_alt_m, self.mu)
        obj = dv1_mag + dv2_mag

        # Constraint: arrive at LMO altitude
        r_rel = state_f[:3] - np.array([1 - self.mu, 0, 0])
        con = np.linalg.norm(r_rel) - self.r_LMO

        return obj, con

    def gradient(self, x):
        """Compute objective gradient via central finite differences."""
        eps = 1e-7
        n = len(x)
        grad = np.zeros(n)

        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * eps)

        return grad

    def constraints(self, x):
        """
        Compute constraint: arrive at LMO altitude.

        Constraint: |r_final - r_moon| - r_LMO = 0
        """
        state_f = self._propagate(x)

        # Position relative to Moon
        r_rel = state_f[:3] - np.array([1 - self.mu, 0, 0])
        r_mag = np.linalg.norm(r_rel)

        # Constraint: should equal zero at solution
        return np.array([r_mag - self.r_LMO])

    def jacobian(self, x):
        """
        Compute constraint Jacobian analytically using the STM.

        The constraint is: c = |r_f - r_moon| - r_LMO
        The Jacobian is: ∂c/∂x = r_hat @ ∂r_f/∂x

        where r_hat is the unit vector from Moon to final position.

        For each decision variable:
        - ∂r_f/∂theta = STM[:3,:] @ ∂state_0/∂theta
        - ∂r_f/∂dv1 = STM[:3, 3:6] (position sensitivity to initial velocity)
        - ∂r_f/∂T = v_f (final velocity, from EoM: dr/dt = v)
        """
        theta = x[0]

        # Propagate and get STM
        state_f, stm = self._propagate(x, return_stm=True)

        # Position relative to Moon and unit vector
        r_moon = np.array([1 - self.mu, 0, 0])
        r_rel = state_f[:3] - r_moon
        r_mag = np.linalg.norm(r_rel)
        r_hat = r_rel / r_mag

        # Extract STM blocks
        # STM maps δstate_0 -> δstate_f
        # Upper-left 3x6: ∂r_f/∂state_0
        drf_dstate0 = stm[:3, :]

        # Compute ∂state_0/∂theta
        dstate0_dtheta = leo_state_derivative_theta(theta, self.leo_alt_m, self.mu)

        # Jacobian components:

        # 1. ∂c/∂theta = r_hat @ (∂r_f/∂state_0 @ ∂state_0/∂theta)
        drf_dtheta = drf_dstate0 @ dstate0_dtheta
        dc_dtheta = r_hat @ drf_dtheta

        # 2. ∂c/∂dv1 = r_hat @ STM[:3, 3:6]
        # dv1 adds directly to initial velocity, so ∂state_0/∂dv1 = [0,0,0,1,0,0], etc.
        drf_ddv1 = stm[:3, 3:6]  # 3x3 matrix
        dc_ddv1 = r_hat @ drf_ddv1  # 1x3 vector

        # 3. ∂c/∂T = r_hat @ v_f
        # From EoM: ∂state_f/∂T = f(state_f) where f is the dynamics
        # For position: ∂r_f/∂T = v_f
        v_f = state_f[3:6]
        dc_dT = r_hat @ v_f

        # Assemble Jacobian: [∂c/∂theta, ∂c/∂dv1_x, ∂c/∂dv1_y, ∂c/∂dv1_z, ∂c/∂T]
        jac = np.array([dc_dtheta, dc_ddv1[0], dc_ddv1[1], dc_ddv1[2], dc_dT])

        return jac

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Callback for iteration statistics."""
        if self.print_iter:
            if iter_count == 0:
                print(f"\n{'Iter':>5} {'Objective':>12} {'Constr Viol':>12} {'Step':>10}")
                print("-" * 45)
            print(f"{iter_count:>5} {obj_value:>12.6f} {inf_pr:>12.2e} {d_norm:>10.2e}")

        self.iter_count = iter_count
        return True


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def optimize_leo_to_lmo(leo_alt_m=463e3, lmo_alt_m=100e3, x0=None,
                        print_iter=True, max_iter=200):
    """
    Optimize bi-impulsive LEO to LMO transfer.

    Parameters
    ----------
    leo_alt_m : float
        LEO altitude in meters (default: 463 km)
    lmo_alt_m : float
        LMO altitude in meters (default: 100 km)
    x0 : ndarray, optional
        Initial guess [theta, dv1_x, dv1_y, dv1_z, T]
        If None, uses default initial guess
    print_iter : bool
        Print iteration statistics (default: True)
    max_iter : int
        Maximum IPOPT iterations (default: 200)

    Returns
    -------
    result : dict
        Optimization results including:
        - x: optimal decision variables
        - theta_rad, theta_deg: departure angle
        - dv1, dv1_mag_norm, dv1_mag_km_s: first burn
        - dv2, dv2_mag_norm, dv2_mag_km_s: second burn
        - tof_norm, tof_days: time of flight
        - total_dv_km_s: total delta-v
        - state_f: final state
        - converged: convergence status
        - info: IPOPT info dict
    """
    mu = MU_EM

    # Create problem instance
    problem = LEOtoLMOProblem(mu, leo_alt_m, lmo_alt_m, print_iter)

    # Variable bounds
    # x = [theta, dv1_x, dv1_y, dv1_z, T]
    lb = np.array([0.0, -1.0, 0.0, -0.5, 0.5])
    ub = np.array([2*np.pi, 1.0, 5.0, 0.5, 6.0])

    # Constraint bounds (equality constraint: cl = cu = 0)
    cl = np.array([0.0])
    cu = np.array([0.0])

    # Default initial guess
    if x0 is None:
        # Compute velocity direction at theta=pi for prograde TLI
        state0 = leo_state_rotating(np.pi, leo_alt_m, mu)
        v_dir = state0[3:6] / np.linalg.norm(state0[3:6])

        # Initial TLI impulse ~3.0 normalized (~3.1 km/s) in velocity direction
        dv1_init = 3.0 * v_dir

        x0 = np.array([
            np.pi,           # theta: depart from far side of Earth
            dv1_init[0],     # dv1_x
            dv1_init[1],     # dv1_y
            dv1_init[2],     # dv1_z
            1.2              # T: ~5 days
        ])

    # Create IPOPT problem
    nlp = cyipopt.Problem(
        n=5,
        m=1,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    # Set IPOPT options
    nlp.add_option('print_level', 5)
    nlp.add_option('tol', 1e-6)
    nlp.add_option('max_iter', max_iter)
    nlp.add_option('jacobian_approximation', 'exact')
    nlp.add_option('hessian_approximation', 'limited-memory')

    # Solve
    print("Starting optimization...")
    x_opt, info = nlp.solve(x0)

    # Extract results
    theta, dv1_x, dv1_y, dv1_z, T = x_opt
    dv1 = np.array([dv1_x, dv1_y, dv1_z])
    dv1_mag = np.linalg.norm(dv1)

    # Final state and second burn
    state_f = problem._propagate(x_opt)
    dv2, dv2_mag = compute_lmo_insertion_dv(state_f, lmo_alt_m, mu)

    # Get characteristic scales for unit conversion
    # Earth-Moon system parameters
    m_earth = 5.972e24  # kg
    m_moon = 7.342e22   # kg
    scales = compute_characteristic_scales(m_earth, m_moon, L_STAR_M)
    v_star = scales['v_star']  # m/s
    t_star = scales['t_star']  # s

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Converged: {info['status'] == 0}")
    print(f"IPOPT status: {info['status_msg']}")

    print(f"\nDeparture:")
    print(f"  Theta: {theta:.4f} rad ({np.rad2deg(theta):.2f} deg)")
    print(f"  DV1: [{dv1[0]:.6f}, {dv1[1]:.6f}, {dv1[2]:.6f}] (normalized)")
    print(f"  |DV1|: {dv1_mag:.6f} normalized = {dv1_mag * v_star / 1000:.4f} km/s")

    print(f"\nTransfer:")
    print(f"  TOF: {T:.4f} normalized = {T * t_star / 86400:.2f} days")

    print(f"\nArrival:")
    print(f"  DV2: [{dv2[0]:.6f}, {dv2[1]:.6f}, {dv2[2]:.6f}] (normalized)")
    print(f"  |DV2|: {dv2_mag:.6f} normalized = {dv2_mag * v_star / 1000:.4f} km/s")

    total_dv_norm = dv1_mag + dv2_mag
    total_dv_km_s = total_dv_norm * v_star / 1000
    print(f"\nTotal Delta-V: {total_dv_km_s:.4f} km/s")

    # Constraint satisfaction
    r_rel = state_f[:3] - np.array([1 - mu, 0, 0])
    r_arrival = np.linalg.norm(r_rel)
    r_target = (R_MOON_M + lmo_alt_m) / L_STAR_M
    constraint_error_km = abs(r_arrival - r_target) * L_STAR_M / 1000
    print(f"\nConstraint error: {constraint_error_km:.4f} km")

    return {
        'x': x_opt,
        'theta_rad': theta,
        'theta_deg': np.rad2deg(theta),
        'dv1': dv1,
        'dv1_mag_norm': dv1_mag,
        'dv1_mag_km_s': dv1_mag * v_star / 1000,
        'dv2': dv2,
        'dv2_mag_norm': dv2_mag,
        'dv2_mag_km_s': dv2_mag * v_star / 1000,
        'tof_norm': T,
        'tof_days': T * t_star / 86400,
        'total_dv_km_s': total_dv_km_s,
        'state_f': state_f,
        'converged': info['status'] == 0,
        'info': info
    }


if __name__ == "__main__":
    # Run optimization with default parameters
    result = optimize_leo_to_lmo()
