"""
Single shooting implementation for trajectory optimization.

Provides the SingleShootingProblem class for bi-impulsive transfer
trajectory optimization in the CR3BP.
"""

import numpy as np
from .orbits import L_STAR_KM, R_EARTH_KM, R_MOON_KM, compute_lmo_insertion_dv


class SingleShootingProblem:
    """
    Single shooting formulation for bi-impulsive LEO to LMO transfer.

    Decision variables: [theta, dv1_x, dv1_y, dv1_z, TOF]

    Attributes
    ----------
    system : CR3BPSystem
        The CR3BP system object (provides mu, solve method, unit conversions)
    leo_altitude : float
        LEO altitude in km
    lmo_altitude : float
        LMO altitude in km
    leo_inclination : float
        LEO inclination in radians
    """

    def __init__(self, system, leo_altitude_km=463.0, lmo_altitude_km=100.0,
                 leo_inclination=0.0):
        """
        Initialize the shooting problem.

        Parameters
        ----------
        system : CR3BPSystem
            Earth-Moon CR3BP system instance
        leo_altitude_km : float
            LEO parking orbit altitude (km)
        lmo_altitude_km : float
            Target LMO altitude (km)
        leo_inclination : float
            LEO inclination (radians)
        """
        self.system = system
        self.mu = system.mu
        self.leo_altitude = leo_altitude_km
        self.lmo_altitude = lmo_altitude_km
        self.leo_inclination = leo_inclination

        # Precompute normalized radii
        self.l_star_km = system.l_star / 1000.0  # convert m to km
        self.R_Earth = R_EARTH_KM / self.l_star_km
        self.R_Moon = R_MOON_KM / self.l_star_km
        self.r_LEO = self.R_Earth + leo_altitude_km / self.l_star_km
        self.r_LMO = self.R_Moon + lmo_altitude_km / self.l_star_km

        # Integration tolerances
        self.rtol = 1e-12
        self.atol = 1e-12

    def unpack_variables(self, x):
        """
        Unpack decision variable vector.

        Parameters
        ----------
        x : ndarray, shape (5,)
            Decision variables [theta, dv1_x, dv1_y, dv1_z, TOF]

        Returns
        -------
        theta : float
            Departure angle on LEO (radians)
        dv1 : ndarray, shape (3,)
            Departure impulse vector (normalized)
        tof : float
            Time of flight (normalized)
        """
        theta = x[0]
        dv1 = np.array([x[1], x[2], x[3]])
        tof = x[4]
        return theta, dv1, tof

    def get_initial_state(self, theta):
        """
        Compute initial state on LEO parking orbit.

        Parameters
        ----------
        theta : float
            Departure angle on LEO (radians)

        Returns
        -------
        state : ndarray, shape (6,)
            State vector on LEO in rotating frame
        """
        mu = self.mu
        r = self.r_LEO
        inc = self.leo_inclination

        # Position in orbital plane
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # Apply inclination rotation (about x-axis in Earth-centered frame)
        x_rel = x_orb
        y_rel = y_orb * np.cos(inc)
        z_rel = y_orb * np.sin(inc)

        # Barycentric position (Earth at -mu)
        x = -mu + x_rel
        y = y_rel
        z = z_rel

        # Circular velocity magnitude (two-body approximation)
        v_circ = np.sqrt((1 - mu) / r)

        # Velocity in orbital plane (perpendicular to position)
        vx_orb = -v_circ * np.sin(theta)
        vy_orb = v_circ * np.cos(theta)

        # Apply inclination rotation
        vx_rel = vx_orb
        vy_rel = vy_orb * np.cos(inc)
        vz_rel = vy_orb * np.sin(inc)

        # Convert to rotating frame: v_rot = v_inertial + [y, -x, 0]
        vx = vx_rel + y
        vy = vy_rel - x
        vz = vz_rel

        return np.array([x, y, z, vx, vy, vz])

    def propagate(self, state0, tof):
        """
        Propagate state forward in time using CR3BP dynamics.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial state
        tof : float
            Time of flight (normalized)

        Returns
        -------
        state_f : ndarray, shape (6,)
            Final state
        sol : OdeSolution
            Full solution object (for trajectory extraction)
        """
        t_span = (0.0, tof)
        states, times, sol = self.system.solve(
            state0, t_span,
            rtol=self.rtol, atol=self.atol,
            dense_output=True
        )
        state_f = states[:, -1]
        return state_f, sol

    def propagate_with_stm(self, state0, tof):
        """
        Propagate state and STM forward in time using CR3BP dynamics.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial state
        tof : float
            Time of flight (normalized)

        Returns
        -------
        state_f : ndarray, shape (6,)
            Final state
        stm : ndarray, shape (6, 6)
            State Transition Matrix Φ(tf, t0)
        sol : OdeSolution
            Full solution object
        """
        t_span = (0.0, tof)
        states, times, stm, sol = self.system.solve_with_stm(
            state0, t_span,
            rtol=self.rtol, atol=self.atol,
            dense_output=True
        )
        state_f = states[:, -1]
        return state_f, stm, sol

    def get_initial_state_derivatives(self, theta):
        """
        Compute derivatives of initial state with respect to theta.

        Parameters
        ----------
        theta : float
            Departure angle on LEO (radians)

        Returns
        -------
        dstate_dtheta : ndarray, shape (6,)
            Partial derivative ∂state/∂theta
        """
        mu = self.mu
        r = self.r_LEO
        inc = self.leo_inclination

        # Position derivatives in orbital plane
        dx_orb_dtheta = -r * np.sin(theta)
        dy_orb_dtheta = r * np.cos(theta)

        # Apply inclination rotation
        dx_rel_dtheta = dx_orb_dtheta
        dy_rel_dtheta = dy_orb_dtheta * np.cos(inc)
        dz_rel_dtheta = dy_orb_dtheta * np.sin(inc)

        # Barycentric position derivatives
        dx_dtheta = dx_rel_dtheta
        dy_dtheta = dy_rel_dtheta
        dz_dtheta = dz_rel_dtheta

        # Velocity derivatives in orbital plane
        v_circ = np.sqrt((1 - mu) / r)
        dvx_orb_dtheta = -v_circ * np.cos(theta)
        dvy_orb_dtheta = -v_circ * np.sin(theta)

        # Apply inclination rotation
        dvx_rel_dtheta = dvx_orb_dtheta
        dvy_rel_dtheta = dvy_orb_dtheta * np.cos(inc)
        dvz_rel_dtheta = dvy_orb_dtheta * np.sin(inc)

        # Convert to rotating frame: v_rot = v_inertial + [y, -x, 0]
        # So dv_rot/dtheta = dv_inertial/dtheta + [dy/dtheta, -dx/dtheta, 0]
        dvx_dtheta = dvx_rel_dtheta + dy_dtheta
        dvy_dtheta = dvy_rel_dtheta - dx_dtheta
        dvz_dtheta = dvz_rel_dtheta

        return np.array([dx_dtheta, dy_dtheta, dz_dtheta,
                         dvx_dtheta, dvy_dtheta, dvz_dtheta])

    def compute_arrival_constraint(self, state_f):
        """
        Compute arrival altitude constraint at Moon.

        Parameters
        ----------
        state_f : ndarray, shape (6,)
            Final state

        Returns
        -------
        constraint : float
            r_moon - r_LMO (equals zero when at correct altitude)
        """
        x, y, z = state_f[:3]
        r_rel = np.array([x - (1 - self.mu), y, z])
        r_moon = np.linalg.norm(r_rel)
        return r_moon - self.r_LMO

    def compute_dv2(self, state_f, prograde=True):
        """
        Compute insertion delta-V at Moon.

        Parameters
        ----------
        state_f : ndarray, shape (6,)
            Final state at arrival
        prograde : bool
            Insert into prograde LMO

        Returns
        -------
        dv2 : ndarray, shape (3,)
            Delta-V vector (normalized)
        dv2_mag : float
            Delta-V magnitude (normalized)
        """
        return compute_lmo_insertion_dv(
            state_f, self.lmo_altitude, self.mu, prograde
        )

    def evaluate(self, x):
        """
        Evaluate the shooting function.

        This is the main evaluation method that computes the objective
        and constraints for a given set of decision variables.

        Parameters
        ----------
        x : ndarray, shape (5,)
            Decision variables [theta, dv1_x, dv1_y, dv1_z, TOF]

        Returns
        -------
        objective : float
            Total delta-V (dv1_mag + dv2_mag)
        constraint : float
            Altitude constraint at Moon (should be zero)
        state_f : ndarray, shape (6,)
            Final state at arrival
        sol : OdeSolution
            Full trajectory solution object
        """
        theta, dv1, tof = self.unpack_variables(x)

        # Initial state on LEO
        state_leo = self.get_initial_state(theta)

        # Apply departure impulse
        state_post_dv1 = state_leo.copy()
        state_post_dv1[3:6] += dv1

        # Propagate to Moon
        state_f, sol = self.propagate(state_post_dv1, tof)

        # Compute constraint (altitude at Moon)
        constraint = self.compute_arrival_constraint(state_f)

        # Compute DV2 for LMO insertion
        dv2, dv2_mag = self.compute_dv2(state_f)

        # Total delta-V objective
        dv1_mag = np.linalg.norm(dv1)
        objective = dv1_mag + dv2_mag

        return objective, constraint, state_f, sol

    def get_trajectory(self, x, n_points=1000):
        """
        Get the full trajectory for given decision variables.

        Parameters
        ----------
        x : ndarray, shape (5,)
            Decision variables
        n_points : int
            Number of points for trajectory output

        Returns
        -------
        states : ndarray, shape (6, n_points)
            Trajectory states
        times : ndarray, shape (n_points,)
            Time points (normalized)
        """
        theta, dv1, tof = self.unpack_variables(x)

        # Initial state with impulse
        state_leo = self.get_initial_state(theta)
        state0 = state_leo.copy()
        state0[3:6] += dv1

        # Propagate with dense output
        t_eval = np.linspace(0, tof, n_points)
        states, times, sol = self.system.solve(
            state0, (0, tof), t_eval=t_eval,
            rtol=self.rtol, atol=self.atol
        )

        return states, times
