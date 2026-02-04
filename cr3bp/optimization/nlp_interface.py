"""
cyipopt interface for trajectory optimization problems.

Provides the NLP problem class compatible with cyipopt for solving
the LEO to LMO transfer optimization.
"""

import numpy as np


class LEOtoLMOProblem:
    """
    NLP problem formulation for LEO to LMO transfer optimization.

    Compatible with cyipopt.Problem interface. Implements:
    - objective(x): Total delta-V
    - gradient(x): Gradient of objective (numerical finite differences)
    - constraints(x): Altitude constraint at Moon
    - jacobian(x): Jacobian of constraints (analytical via STM)

    Decision variables: x = [theta, dv1_x, dv1_y, dv1_z, TOF]

    Uses State Transition Matrix (STM) for analytical constraint Jacobian.
    Uses numerical differentiation for objective gradient.
    IPOPT approximates the Hessian using L-BFGS.
    """

    def __init__(self, shooting_problem, print_diagnostics=True):
        """
        Initialize NLP problem.

        Parameters
        ----------
        shooting_problem : SingleShootingProblem
            The shooting problem instance
        print_diagnostics : bool
            Whether to print iteration diagnostics
        """
        self.shooting = shooting_problem
        self.print_diagnostics = print_diagnostics

        # Finite difference step size (for gradient only)
        self.fd_step = 1e-7

        # Cache for avoiding redundant evaluations
        self._cache_x = None
        self._cache_obj = None
        self._cache_con = None
        self._cache_state_f = None
        self._cache_stm = None
        self._cache_sol = None

        # Track iteration history for diagnostics
        self.iteration_history = []

    def _evaluate_and_cache(self, x, with_stm=True):
        """
        Evaluate shooting function and cache results.

        This avoids redundant trajectory propagations when IPOPT
        calls objective() and constraints() with the same x.

        Parameters
        ----------
        x : ndarray
            Decision variables
        with_stm : bool
            If True, also compute and cache the STM
        """
        need_recompute = self._cache_x is None or not np.allclose(x, self._cache_x, rtol=1e-14)
        need_stm = with_stm and self._cache_stm is None

        if need_recompute or need_stm:
            theta, dv1, tof = self.shooting.unpack_variables(x)

            # Initial state on LEO
            state_leo = self.shooting.get_initial_state(theta)

            # Apply departure impulse
            state_post_dv1 = state_leo.copy()
            state_post_dv1[3:6] += dv1

            # Propagate with or without STM
            if with_stm:
                state_f, stm, sol = self.shooting.propagate_with_stm(state_post_dv1, tof)
                self._cache_stm = stm
            else:
                state_f, sol = self.shooting.propagate(state_post_dv1, tof)
                self._cache_stm = None

            # Compute constraint and objective
            constraint = self.shooting.compute_arrival_constraint(state_f)
            dv2, dv2_mag = self.shooting.compute_dv2(state_f)
            dv1_mag = np.linalg.norm(dv1)
            objective = dv1_mag + dv2_mag

            self._cache_x = x.copy()
            self._cache_obj = objective
            self._cache_con = np.array([constraint])
            self._cache_state_f = state_f
            self._cache_sol = sol

        return self._cache_obj, self._cache_con

    def objective(self, x):
        """
        Compute objective function (total delta-V).

        Parameters
        ----------
        x : ndarray
            Decision variables

        Returns
        -------
        f : float
            Objective value (total delta-V in normalized units)
        """
        obj, _ = self._evaluate_and_cache(x)
        return obj

    def gradient(self, x):
        """
        Compute gradient of objective using forward finite differences.

        Parameters
        ----------
        x : ndarray
            Decision variables

        Returns
        -------
        grad : ndarray
            Gradient vector
        """
        n = len(x)
        grad = np.zeros(n)
        f0 = self.objective(x)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += self.fd_step
            # Clear cache to force re-evaluation
            self._cache_x = None
            f_plus = self.objective(x_plus)
            grad[i] = (f_plus - f0) / self.fd_step

        # Restore cache for original x
        self._cache_x = None
        self._evaluate_and_cache(x)

        return grad

    def constraints(self, x):
        """
        Compute constraint values.

        Parameters
        ----------
        x : ndarray
            Decision variables

        Returns
        -------
        c : ndarray
            Constraint values (altitude constraint)
        """
        _, con = self._evaluate_and_cache(x)
        return con

    def jacobian(self, x):
        """
        Compute Jacobian of constraints using analytical derivatives via STM.

        The constraint is: c = ||r_f - r_moon|| - r_LMO
        where r_f is the final position and r_moon = (1-μ, 0, 0).

        Decision variables: x = [theta, dv1_x, dv1_y, dv1_z, TOF]

        Using chain rule:
        - ∂c/∂theta = ∂c/∂state_f @ Φ @ ∂state_0/∂theta
        - ∂c/∂dv1 = ∂c/∂state_f @ Φ[:, 3:6]
        - ∂c/∂TOF = ∂c/∂state_f @ f(state_f)

        Parameters
        ----------
        x : ndarray
            Decision variables

        Returns
        -------
        jac : ndarray
            Flattened Jacobian matrix (m x n -> m*n)
        """
        # Ensure we have the cached values with STM
        self._evaluate_and_cache(x, with_stm=True)

        theta, dv1, tof = self.shooting.unpack_variables(x)
        state_f = self._cache_state_f
        stm = self._cache_stm
        mu = self.shooting.mu

        # Position relative to Moon
        r_moon = np.array([1 - mu, 0, 0])
        r_rel = state_f[:3] - r_moon
        r_rel_norm = np.linalg.norm(r_rel)

        # Gradient of constraint w.r.t. final state
        # c = ||r_rel|| - r_LMO
        # ∂c/∂r_f = r_rel / ||r_rel||
        # ∂c/∂v_f = 0
        dc_dstate_f = np.zeros(6)
        dc_dstate_f[:3] = r_rel / r_rel_norm

        # Initialize Jacobian
        jac = np.zeros(5)

        # ∂c/∂theta: chain through STM and initial state derivatives
        dstate0_dtheta = self.shooting.get_initial_state_derivatives(theta)
        # The impulse dv1 is added to velocity, so initial state derivative is the same
        dstate_f_dtheta = stm @ dstate0_dtheta
        jac[0] = dc_dstate_f @ dstate_f_dtheta

        # ∂c/∂dv1: dv1 directly affects initial velocity
        # ∂state_0/∂dv1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # So ∂state_f/∂dv1 = Φ[:, 3:6]
        dstate_f_ddv1 = stm[:, 3:6]  # shape (6, 3)
        jac[1:4] = dc_dstate_f @ dstate_f_ddv1

        # ∂c/∂TOF: derivative of final state w.r.t. time is the EoM
        # ∂state_f/∂TOF = f(state_f)
        f_state_f = self.shooting.system.eom(tof, state_f)
        jac[4] = dc_dstate_f @ f_state_f

        return jac

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """
        Callback called by IPOPT after each iteration.

        Parameters
        ----------
        alg_mod : int
            Algorithm mode (1=regular, 2=restoration phase)
        iter_count : int
            Current iteration number
        obj_value : float
            Current objective value
        inf_pr : float
            Primal infeasibility (constraint violation)
        inf_du : float
            Dual infeasibility (optimality condition)
        mu : float
            Barrier parameter
        d_norm : float
            Norm of primal step
        regularization_size : float
            Regularization size
        alpha_du : float
            Dual step size
        alpha_pr : float
            Primal step size
        ls_trials : int
            Number of line search trials

        Returns
        -------
        bool
            True to continue, False to terminate
        """
        # Store iteration data
        iter_data = {
            'iter': iter_count,
            'obj': obj_value,
            'inf_pr': inf_pr,
            'inf_du': inf_du,
            'mu': mu,
            'd_norm': d_norm,
            'alpha_pr': alpha_pr,
            'ls_trials': ls_trials,
            'mode': 'regular' if alg_mod == 1 else 'restoration'
        }
        self.iteration_history.append(iter_data)

        if self.print_diagnostics:
            # Print header every 20 iterations
            if iter_count % 20 == 0:
                print("\n" + "="*80)
                print(f"{'Iter':>5} {'Objective':>12} {'Constr Viol':>12} {'Dual Inf':>12} "
                      f"{'Step':>10} {'α_pr':>8} {'Mode':>10}")
                print("="*80)

            mode_str = "RESTORE" if alg_mod == 2 else "normal"
            print(f"{iter_count:>5} {obj_value:>12.6f} {inf_pr:>12.2e} {inf_du:>12.2e} "
                  f"{d_norm:>10.2e} {alpha_pr:>8.2e} {mode_str:>10}")

            # Warnings for potential issues
            if alg_mod == 2:
                print("  ⚠ RESTORATION PHASE - solver struggling with feasibility")
            if alpha_pr < 1e-8:
                print("  ⚠ Very small step size - may be stuck")
            if ls_trials > 10:
                print(f"  ⚠ Many line search trials ({ls_trials}) - difficult region")

        return True  # Continue optimization


def create_nlp_problem(shooting_problem, x0=None, print_diagnostics=True):
    """
    Create and configure cyipopt problem for LEO-LMO transfer.

    Parameters
    ----------
    shooting_problem : SingleShootingProblem
        The shooting problem instance
    x0 : ndarray, optional
        Initial guess [theta, dv1_x, dv1_y, dv1_z, TOF].
        If None, uses a default Hohmann-like guess.
    print_diagnostics : bool
        Whether to print iteration diagnostics

    Returns
    -------
    nlp : cyipopt.Problem
        Configured NLP problem ready to solve
    x0 : ndarray
        Initial guess (input or default)
    problem_obj : LEOtoLMOProblem
        Problem object (for accessing cached results after solve)
    """
    import cyipopt

    # Create problem object
    problem_obj = LEOtoLMOProblem(shooting_problem, print_diagnostics=print_diagnostics)

    # Problem dimensions
    n_vars = 5   # [theta, dv1_x, dv1_y, dv1_z, TOF]
    n_cons = 1   # Moon altitude constraint

    # Variable bounds
    # theta: [0, 2*pi] - full orbit coverage
    # dv1 components: TLI needs ~3.1 km/s ≈ 3.0 normalized, allow margin
    # TOF: [0.5, 20] normalized (~2 to 80 days for low-energy options)
    lb = np.array([0.0, -5.0, -5.0, -2.0, 0.5])
    ub = np.array([2 * np.pi, 5.0, 5.0, 2.0, 6.0])

    # Constraint bounds (equality constraint: cl = cu = 0)
    cl = np.array([0.0])
    cu = np.array([0.0])

    # Default initial guess (Hohmann-like transfer)
    # TLI from LEO needs ~3.1 km/s prograde ≈ 3.0 normalized units
    # Transfer time ~4-5 days ≈ 1.0-1.2 normalized
    if x0 is None:
        x0 = np.array([
            np.pi,   # theta: depart from far side of Earth (away from Moon)
            -1,     # dv1_x: small radial-out component
            4.8,    # dv1_y: prograde at θ=π means -y direction
            0.0,     # dv1_z: planar transfer
            1.2      # TOF: ~5 days in normalized time
        ])

    # Create cyipopt problem
    nlp = cyipopt.Problem(
        n=n_vars,
        m=n_cons,
        problem_obj=problem_obj,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    # Solver options
    nlp.add_option('print_level', 5)
    nlp.add_option('tol', 1e-6)
    nlp.add_option('max_iter', 500)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory')  # L-BFGS

    return nlp, x0, problem_obj


def solve_transfer(shooting_problem, x0=None, print_level=5, print_diagnostics=True):
    """
    Convenience function to solve the LEO-LMO transfer problem.

    Parameters
    ----------
    shooting_problem : SingleShootingProblem
        The shooting problem instance
    x0 : ndarray, optional
        Initial guess
    print_level : int
        IPOPT print level (0=silent, 5=default, 12=verbose)
    print_diagnostics : bool
        Whether to print custom iteration diagnostics

    Returns
    -------
    result : dict
        Optimization results containing:
        - 'x': Optimal decision variables
        - 'theta_rad': Departure angle (radians)
        - 'theta_deg': Departure angle (degrees)
        - 'dv1': Departure delta-V vector (normalized)
        - 'dv1_mag': Departure delta-V magnitude (normalized)
        - 'tof': Time of flight (normalized)
        - 'objective': Total delta-V (normalized)
        - 'constraint': Final constraint violation
        - 'state_f': Final state at Moon
        - 'converged': Whether optimization converged
        - 'info': Full solver info dict
        - 'iteration_history': List of iteration diagnostics
    """
    # Create and configure NLP
    nlp, x0, problem_obj = create_nlp_problem(shooting_problem, x0, print_diagnostics)
    nlp.add_option('print_level', print_level)

    # Solve
    x_opt, info = nlp.solve(x0)

    # Extract results
    theta, dv1, tof = shooting_problem.unpack_variables(x_opt)
    obj, con, state_f, sol = shooting_problem.evaluate(x_opt)

    result = {
        'x': x_opt,
        'theta_rad': theta,
        'theta_deg': np.rad2deg(theta),
        'dv1': dv1,
        'dv1_mag': np.linalg.norm(dv1),
        'tof': tof,
        'objective': obj,
        'constraint': con,
        'state_f': state_f,
        'converged': info['status'] == 0,
        'info': info,
        'iteration_history': problem_obj.iteration_history
    }

    # Print convergence summary
    if print_diagnostics and problem_obj.iteration_history:
        print("\n" + "="*80)
        print("CONVERGENCE SUMMARY")
        print("="*80)
        n_iters = len(problem_obj.iteration_history)
        final = problem_obj.iteration_history[-1]
        print(f"Total iterations: {n_iters}")
        print(f"Final objective:  {final['obj']:.6f}")
        print(f"Final constraint violation: {final['inf_pr']:.2e}")
        print(f"Final dual infeasibility:   {final['inf_du']:.2e}")

        # Check for issues
        restoration_iters = sum(1 for h in problem_obj.iteration_history if h['mode'] == 'restoration')
        if restoration_iters > 0:
            print(f"⚠ Restoration phase entered {restoration_iters} times")

        small_steps = sum(1 for h in problem_obj.iteration_history if h['alpha_pr'] < 1e-8)
        if small_steps > 5:
            print(f"⚠ Very small steps taken {small_steps} times - potential convergence issues")

        if info['status'] == 0:
            print("✓ Optimization converged successfully")
        else:
            print(f"✗ Optimization terminated with status {info['status']}")
            status_messages = {
                1: "Maximum iterations exceeded",
                2: "Infeasible problem detected",
                -1: "Solver error",
            }
            print(f"  Reason: {status_messages.get(info['status'], 'Unknown')}")

    return result
