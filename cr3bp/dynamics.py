"""
CR3BP Dynamics

This module contains the dynamics functions for CR3BP trajectories.

Functions:
cr3bp_eom
solve_CR3BP
circular_orbit_state_earth
decision_to_state0

"""

import numpy as np
from scipy.integrate import solve_ivp

def cr3bp_eom(t, state, mu):
    """
    Equations of motion for the Circular Restricted Three-Body Problem in the 
    rotating frame.
    
    The rotating frame rotates with angular velocity ω = 1 (normalized) such that
    the two primaries remain fixed on the x-axis:
    - Primary 1 (larger, mass = 1-μ) at position (-μ, 0, 0)
    - Primary 2 (smaller, mass = μ) at position (1-μ, 0, 0)
    
    Inputs
    t : float
        Time (not used explicitly, system is autonomous in rotating frame)
    state : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter μ = m2/(m1+m2)
    
    Outputs
    dstate : ndarray, shape (6,)
        Time derivative of state [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    
    # Distances to primaries
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    # Accelerations in rotating frame
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    return np.array([vx, vy, vz, ax, ay, az])


def solve_CR3BP(state0, t_span, mu, t_eval=None, rtol=1e-12, atol=1e-12, 
                dense_output=False, events=None):
    """
    Solve the EoM using DOP853.
    
    Inputs
    state0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz] in normalized coordinates
    t_span : tuple
        Integration time span (t0, tf) in normalized time units
    mu : float
        Mass parameter (mu = m2/(m1+m2))

    Optional Inputs    
    t_eval : array_like, optional
        Times at which to store the solution. If None, solver chooses times.
    rtol : float, optional
        Relative tolerance for integration (default: 1e-12)
    atol : float, optional
        Absolute tolerance for integration (default: 1e-12)
    dense_output : bool, optional
        Whether to compute a continuous solution (default: False)
    events : callable or list of callables, optional
        Event functions for detection during integration
    
    Outputs
    states : ndarray, shape (6, N)
        State vectors at each time point
    times : ndarray, shape (N,)
        Time points corresponding to each state
    sol : OdeResult
        Full solution object from solve_ivp (includes dense_output if requested)
    """
    sol = solve_ivp(
        fun=lambda t, y: cr3bp_eom(t, y, mu),
        t_span=t_span,
        y0=state0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        events=events
    )
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    
    return sol.y, sol.t, sol

def circular_orbit_state_earth(cr3bp, theta, altitude_m):
    """
    Compute state vector for circular orbit around Earth in CR3BP rotating frame.

    Parameters
    ----------
    cr3bp : Class
        cr3bp class containing all relevant information
    theta : float
        True anomaly / angle on orbit (radians), measured from +x axis
    altitude_m : float
        Altitude above Earth surface (m)

    Returns
    -------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in normalized rotating frame
    v_circ : float
        Circular orbit velocity magnitude (normalized)
    """
    # Normalized radii
    mu = cr3bp.mu

    r_earth = cr3bp.r1 / cr3bp.l_star
    r_orbit = r_earth + altitude_m / cr3bp.l_star

    # Position in orbital plane 
    x_orb = r_orbit * np.cos(theta)
    y_orb = r_orbit * np.sin(theta)


    # Shift to barycentric frame (Earth at -mu)
    x = -mu + x_orb
    y = y_orb
    z = 0

    # Circular velocity magnitude (Earth-centered two-body approximation)
    v_circ = np.sqrt((1 - mu) / r_orbit)

    # Velocity in orbital plane (perpendicular to position)
    vx_orb = -v_circ * np.sin(theta)
    vy_orb = v_circ * np.cos(theta)

    # Convert to rotating frame velocity
    # v_rot = v_inertial - omega x r, where omega = [0, 0, 1]
    # omega x r = [-y, x, 0]
    # So v_rot = v_inertial + [y, -x, 0]
    vx = vx_orb + y
    vy = vy_orb - x
    vz = 0

    return np.array([x, y, z, vx, vy, vz])

def ciruclar_orbit_trajectory_moon(cr3bp, state_f_inertial, state_moon_f_inertial, tof, num_points=1000):
    """
    Helper function which determines the trajectory of the llo which the rocket is at
    Parameters
    ----------  
    cr3bp : Class
    state_f_inertial : array 6,
        state vector (x,y,z,vx,vy,vz) of rocket in the inertial ref frame in normalized units
    state_moon_f_inertial : array 6,
        state vector (x,y,z,vx,vy,vz) of moon in the inertial ref frame in normalized units
    tof : float
        Time of flight (normalized units)
    num_points : int, optional
        Number of points in the trajectory (default 1000)
    Return
    ------
    llo_traj : array (6, num_points)
        x,y,z,vx,vy,vz trajectory of the llo in the inertial frame in normalized units
    """
    pos_rocket = state_f_inertial[0:3]
    pos_moon = state_moon_f_inertial[0:3]
    r_rocket_moon = pos_rocket - pos_moon

    r_orbit = np.linalg.norm(r_rocket_moon)
    phi_0 = np.arctan2(r_rocket_moon[1], r_rocket_moon[0])
    omega = -1 * np.sqrt(cr3bp.mu / r_orbit**3)

    moon_r = np.linalg.norm(pos_moon)
    moon_phi_0 = np.arctan2(pos_moon[1], pos_moon[0])

    t_array = np.linspace(0, tof, num_points)

    llo_traj = np.zeros((6, num_points))
    for i, t in enumerate(t_array):
        moon_phi = moon_phi_0 + t
        phi = phi_0 + omega * t

        llo_traj[0, i] = moon_r * np.cos(moon_phi) + r_orbit * np.cos(phi)
        llo_traj[1, i] = moon_r * np.sin(moon_phi) + r_orbit * np.sin(phi)
        llo_traj[2:, i] = 0

    return llo_traj

def decision_to_state0(cr3bp, x0, leo_alt_m):
    # Unpack decision variables
    theta = x0[0] # Angle around LEO
    delta_v = x0[1] # Magnitude of boost
    alpha = x0[2] # Angle of boost

    state_leo_rotating = circular_orbit_state_earth(cr3bp, theta, leo_alt_m)
    state_leo_inertial = cr3bp.rotating_to_inertial(state_leo_rotating,0)
    
    vx = state_leo_inertial[3]
    vy = state_leo_inertial[4]

    theta = np.arctan2(vy, vx)
    mag = np.sqrt(vx**2 + vy**2)
    
    delta_v_x = delta_v*np.cos(theta + alpha)
    delta_v_y = delta_v*np.sin(theta + alpha)

    state_leo_inertial_boost = state_leo_inertial
    state_leo_inertial_boost[3] += delta_v_x
    state_leo_inertial_boost[4] += delta_v_y

    return cr3bp.inertial_to_rotating(state_leo_inertial_boost,0)


def v_llo_at_rocket_pos(cr3bp, state_f_inertial, moon_f_inertial):
    """
    Helper function which determines the velocity at the llo which the rocket is at
    
    Parameters
    ----------
    cr3bp : Class

    state_f_inertial : array 6,
        state vector (x,y,z,vx,vy,vz) of rocket in the inertial ref frame in normalized units
    moon_f_inertial : array 6,
        state vector (x,y,z,vx,vy,vz) of moon in the inertial ref frame in normalized units

    Return
    ------
    v_llo: array 3,
        velocity at the llo at pos_f_rocket_inertial
    """
    pos_rocket = state_f_inertial[0:3]
    pos_moon = moon_f_inertial[0:3]
    r_rocket_moon = pos_moon - pos_rocket

    v_llo_mag = np.sqrt(cr3bp.mu / np.linalg.norm(r_rocket_moon))
    v_llo_dir = np.cross(np.array([0,0,1]),r_rocket_moon / np.linalg.norm(r_rocket_moon))

    return v_llo_mag * v_llo_dir

def jacobi_integral(cr3bp, state):
    """
    Compute the Jacobi integral (Jacobi constant) for a given state in the rotating frame.
    
    Parameters
    ----------
    cr3bp : Class
        CR3BP system containing mu and l_star for normalization
    state : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz] in normalized rotating frame

    Returns
    -------
    C : float
        Jacobi integral value for the given state
    """
    x, y, z, vx, vy, vz = state
    mu = cr3bp.mu
    
    # Distances to primaries
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    # Effective potential
    U = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    
    # Kinetic energy
    T = 0.5 * (vx**2 + vy**2 + vz**2)
    
    # Jacobi integral
    C = 2*U - 2*T
    
    return C
