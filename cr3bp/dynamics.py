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

def cr3bp_jacobian(state, mu):
    """
    Compute the Jacobian matrix (A matrix) of the CR3BP equations of motion.

    This is used for STM propagation: dΦ/dt = A(t) @ Φ

    Parameters
    ----------
    state : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter μ = m2/(m1+m2)

    Returns
    -------
    A : ndarray, shape (6, 6)
        Jacobian matrix ∂f/∂state where f is the EoM
    """
    x, y, z = state[:3]

    # Distances to primaries
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5

    # Partial derivatives of the pseudo-potential Uxx, Uxy, etc.
    # U = (1/2)(x^2 + y^2) + (1-mu)/r1 + mu/r2
    Uxx = 1 - (1-mu)/r1_3 - mu/r2_3 + 3*(1-mu)*(x+mu)**2/r1_5 + 3*mu*(x-(1-mu))**2/r2_5
    Uyy = 1 - (1-mu)/r1_3 - mu/r2_3 + 3*(1-mu)*y**2/r1_5 + 3*mu*y**2/r2_5
    Uzz = -(1-mu)/r1_3 - mu/r2_3 + 3*(1-mu)*z**2/r1_5 + 3*mu*z**2/r2_5
    Uxy = 3*(1-mu)*(x+mu)*y/r1_5 + 3*mu*(x-(1-mu))*y/r2_5
    Uxz = 3*(1-mu)*(x+mu)*z/r1_5 + 3*mu*(x-(1-mu))*z/r2_5
    Uyz = 3*(1-mu)*y*z/r1_5 + 3*mu*y*z/r2_5

    # Jacobian matrix A = ∂f/∂state
    # f = [vx, vy, vz, ax, ay, az]
    # state = [x, y, z, vx, vy, vz]
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [Uxx, Uxy, Uxz, 0, 2, 0],
        [Uxy, Uyy, Uyz, -2, 0, 0],
        [Uxz, Uyz, Uzz, 0, 0, 0]
    ])

    return A


def cr3bp_eom_with_stm(t, state_stm, mu):
    """
    Equations of motion for CR3BP with State Transition Matrix propagation.

    The state vector is augmented with the flattened STM (6x6 = 36 elements).

    Parameters
    ----------
    t : float
        Time
    state_stm : array_like, shape (42,)
        Augmented state [x, y, z, vx, vy, vz, Φ11, Φ12, ..., Φ66]
        where Φij are the STM elements in row-major order
    mu : float
        Mass parameter

    Returns
    -------
    dstate_stm : ndarray, shape (42,)
        Time derivative of augmented state
    """
    # Extract state and STM
    state = state_stm[:6]
    stm_flat = state_stm[6:]
    stm = stm_flat.reshape((6, 6))

    # State derivatives
    dstate = cr3bp_eom(t, state, mu)

    # STM derivatives: dΦ/dt = A @ Φ
    A = cr3bp_jacobian(state, mu)
    dstm = A @ stm

    return np.concatenate([dstate, dstm.flatten()])


def solve_CR3BP_with_STM(state0, t_span, mu, t_eval=None, rtol=1e-12, atol=1e-12,
                         dense_output=False):
    """
    Solve CR3BP equations of motion with STM propagation.

    Parameters
    ----------
    state0 : array_like, shape (6,)
        Initial state vector [x, y, z, vx, vy, vz]
    t_span : tuple
        Integration time span (t0, tf)
    mu : float
        Mass parameter
    t_eval : array_like, optional
        Times at which to store the solution
    rtol : float, optional
        Relative tolerance (default: 1e-12)
    atol : float, optional
        Absolute tolerance (default: 1e-12)
    dense_output : bool, optional
        Whether to compute continuous solution (default: False)

    Returns
    -------
    states : ndarray, shape (6, N)
        State vectors at each time point
    times : ndarray, shape (N,)
        Time points
    stm_final : ndarray, shape (6, 6)
        State Transition Matrix at final time
    sol : OdeResult
        Full solution object from solve_ivp
    """
    # Initialize STM as identity matrix
    stm0 = np.eye(6).flatten()
    state_stm0 = np.concatenate([state0, stm0])

    sol = solve_ivp(
        fun=lambda t, y: cr3bp_eom_with_stm(t, y, mu),
        t_span=t_span,
        y0=state_stm0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract state and final STM
    states = sol.y[:6, :]
    stm_final = sol.y[6:, -1].reshape((6, 6))

    return states, sol.t, stm_final, sol


def circular_orbit_state_earth(theta, altitude_m, inclination=0.0, mu=0.01215):
    """
    Compute state vector for circular orbit around Earth in Inertial Frame in 
    Normalized Units

    Parameters
    ----------
    theta : float
        True anomaly / angle on orbit (radians), measured from +x axis
    altitude_m : float
        Altitude above Earth surface (m)
    inclination : float, optional
        Orbital inclination (radians), default 0 (equatorial)
    mu : float
        CR3BP mass parameter 

    Returns
    -------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in inertial Frame in normalized units
    """

    # Normalized radii
    L_STAR_M = 384400000.0  # Earth-Moon distance in km
    R_EARTH_M = 6371000.0   # Earth radius in km

    r_earth = R_EARTH_M / L_STAR_M
    r_orbit = r_earth + altitude_m / L_STAR_M

    # Position in orbital plane (before inclination)
    x_orb = r_orbit * np.cos(theta)
    y_orb = r_orbit * np.sin(theta)

    # Apply inclination rotation (about the Earth-Moon x-axis)
    x_rel = x_orb
    y_rel = y_orb * np.cos(inclination)
    z_rel = y_orb * np.sin(inclination)

    # Shift to barycentric frame (Earth at -mu)
    x = -mu + x_rel
    y = y_rel
    z = z_rel

    # Circular velocity magnitude (Earth-centered two-body approximation)
    v_circ = np.sqrt((1 - mu) / r_orbit)

    # Velocity in orbital plane (perpendicular to position)
    vx_orb = -v_circ * np.sin(theta)
    vy_orb = v_circ * np.cos(theta)

    # Apply inclination rotation
    vx = vx_orb
    vy = vy_orb * np.cos(inclination)
    vz = vy_orb * np.sin(inclination)

    return np.array([x, y, z, vx, vy, vz])