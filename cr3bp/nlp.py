import numpy as np
import cyipopt

from cr3bp.system import create_earth_moon_system
from cr3bp.objective import evaluate
from cr3bp import file_path

class EarthMoonTransferNLP:
    """
    NLP formulation for minimum delta-v Earth-Moon transfer (LEO to LMO).

    Minimize: total_delta_v = delta_v1 + delta_v2
    Subject to: distance_rocket_lmo = 0 (equality constraint)

    Decision variables:
        x[0] = theta         : angle around LEO orbit (radians)
        x[1] = delta_v       : magnitude of initial impulse (normalized)
        x[2] = delta_v_angle : angle of impulse relative to velocity (radians)
        x[3] = tof           : time of flight (normalized)

    Uses finite-difference gradients and L-BFGS Hessian approximation.

    Parameters
    ----------
    cr3bp : CR3BPSystem
        Earth-Moon CR3BP system instance
    leo_alt_m : float
        Low Earth Orbit altitude in meters
    lmo_alt_m : float
        Low Moon Orbit altitude in meters
    """

    # Finite-difference step size for each decision variable
    FD_STEP = np.array([1e-6, 1e-6, 1e-6, 1e-6])

    def __init__(self, cr3bp, leo_alt_m, lmo_alt_m):
        self.cr3bp = cr3bp
        self.leo_alt_m = leo_alt_m
        self.lmo_alt_m = lmo_alt_m

    def objective(self, x):
        """
        Objective function: total delta-v in normalized units.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Decision variables [theta, delta_v, delta_v_angle, tof]

        Returns
        -------
        float
            Total delta-v (delta_v1 + delta_v2) in normalized units
        """
        _, total_delta_v = evaluate(self.cr3bp, x, self.leo_alt_m, self.lmo_alt_m)
        return total_delta_v

    def gradient(self, x):
        """
        Gradient of the objective function via forward finite differences.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Decision variables [theta, delta_v, delta_v_angle, tof]

        Returns
        -------
        grad : ndarray, shape (4,)
            Gradient of total delta-v w.r.t. each decision variable
        """
        n = len(x)
        grad = np.zeros(n)
        f0 = self.objective(x)

        for i in range(n):
            x_fwd = x.copy()
            x_fwd[i] += self.FD_STEP[i]
            f_fwd = self.objective(x_fwd)
            grad[i] = (f_fwd - f0) / self.FD_STEP[i]

        return grad

    def constraints(self, x):
        """
        Constraint function: distance from rocket final position to LMO radius.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Decision variables [theta, delta_v, delta_v_angle, tof]

        Returns
        -------
        ndarray, shape (1,)
            Distance from rocket to LMO orbit (normalized). Equality constraint = 0.
        """
        try:
            distance_rocket_lmo, _ = evaluate(self.cr3bp, x, self.leo_alt_m, self.lmo_alt_m)
        except Exception:
            # If evaluation fails (e.g., out of bounds), return a large constraint value
            distance_rocket_lmo = 1e6  # Large value to indicate constraint violation
        return np.array([distance_rocket_lmo])

    def jacobian(self, x):
        """
        Jacobian of constraints via forward finite differences.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Decision variables [theta, delta_v, delta_v_angle, tof]

        Returns
        -------
        jac : ndarray, shape (4,)
            Jacobian of the constraint w.r.t. each decision variable (flattened)
        """
        n = len(x)
        jac = np.zeros(n)
        c0 = self.constraints(x)[0]

        for i in range(n):
            x_fwd = x.copy()
            x_fwd[i] += self.FD_STEP[i]
            c_fwd = self.constraints(x_fwd)[0]
            jac[i] = (c_fwd - c0) / self.FD_STEP[i]

        return jac


def solve_transfer_optimization():
    """Solve the minimum delta-v Earth-Moon transfer using NLP with cyipopt."""

    ### System Setup ###############################################
    em = create_earth_moon_system()
    leo_alt_m = 463e3   # 463 km LEO altitude
    lmo_alt_m = 100e3   # 100 km LMO altitude

    ### Initial Guess from Grid Search #############################
    x0 = np.load(f"{file_path}/optimals_iteration3.npy")
    print(f"Initial guess from grid search: {x0}")
    print(f"  theta          = {x0[0]:.6f} rad")
    print(f"  delta_v        = {x0[1]:.6f} (normalized) = {x0[1]*em.v_star*1e-3:.4f} km/s")
    print(f"  delta_v_angle  = {x0[2]:.6f} rad")
    print(f"  tof            = {x0[3]:.6f} (normalized) = {x0[3]*em.t_star/86400:.3f} days")

    ### Problem Setup ##############################################
    n_vars = 4   # theta, delta_v, delta_v_angle, tof
    n_cons = 1   # distance_rocket_lmo = 0

    # Variable bounds
    lb = np.array([3.5, 2.0, 0.0, 0.5])
    ub = np.array([4.5, 4.0, 0.2, 1.5])

    # Constraint bounds (equality: distance_rocket_lmo = 0)
    cl = np.array([0.0])
    cu = np.array([0.0])

    # Create problem instance
    problem = EarthMoonTransferNLP(em, leo_alt_m, lmo_alt_m)

    # Define the problem for cyipopt
    nlp = cyipopt.Problem(
        n=n_vars,
        m=n_cons,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    ### Solver Options #############################################
    nlp.add_option('print_level', 5)
    nlp.add_option('max_iter', 200)
    nlp.add_option('tol', 1e-4)
    nlp.add_option('acceptable_tol', 1e-3)
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('mu_strategy', 'adaptive')

    ### Solve ######################################################
    x_opt, info = nlp.solve(x0)

    ### Results ####################################################
    constraint_val, objective_val = evaluate(em, x_opt, leo_alt_m, lmo_alt_m)

    total_dv_SI = objective_val * em.v_star       # m/s
    total_dv_kms = total_dv_SI * 1e-3             # km/s
    constraint_km = constraint_val * em.l_star * 1e-3  # km
    tof_days = x_opt[3] * em.t_star / 86400       # days

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Solver status: {info['status_msg']}")
    print(f"\nOptimal decision variables:")
    print(f"  theta          = {x_opt[0]:.6f} rad")
    print(f"  delta_v        = {x_opt[1]:.6f} (normalized) = {x_opt[1]*em.v_star*1e-3:.4f} km/s")
    print(f"  delta_v_angle  = {x_opt[2]:.6f} rad")
    print(f"  tof            = {x_opt[3]:.6f} (normalized) = {tof_days:.3f} days")
    print(f"\nTotal delta-v:")
    print(f"  Normalized     = {objective_val:.6f}")
    print(f"  SI             = {total_dv_SI:.2f} m/s")
    print(f"  SI             = {total_dv_kms:.4f} km/s")
    print(f"\nConstraint satisfaction:")
    print(f"  distance_rocket_lmo (normalized) = {constraint_val:.6e}")
    print(f"  distance_rocket_lmo (km)         = {constraint_km:.4f}")
    print(f"\nTime of flight: {tof_days:.3f} days")
    print("=" * 60)

    return x_opt, info
