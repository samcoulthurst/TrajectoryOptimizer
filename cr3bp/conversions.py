import numpy as np


# ============================================================================
# FRAME CONVERSIONS
# ============================================================================

def rotating_to_inertial(state_rot, t, mu=0.012):
    """
    Convert state from rotating (synodic) frame to inertial frame.
    
    In the CR3BP rotating frame, the two primaries are fixed on the x-axis.
    The inertial frame is centered at the barycenter with the primaries
    orbiting with angular velocity omega = 1 (in normalized units).
    
    Parameters
    ----------
    state_rot : array_like, shape (6,)
        State vector in rotating frame [x, y, z, vx, vy, vz]
    t : float
        Current time in normalized units
    mu : float
        Mass parameter (not used in transformation, included for consistency)
    
    Returns
    -------
    state_iner : ndarray, shape (6,)
        State vector in inertial frame [X, Y, Z, VX, VY, VZ]
    """
    x, y, z, vx, vy, vz = state_rot
    
    # Rotation angle (omega = 1 in normalized units)
    theta = t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Position transformation: R(theta) * r_rot
    X = cos_t * x - sin_t * y
    Y = sin_t * x + cos_t * y
    Z = z
    
    # Velocity transformation: R(theta) * v_rot + omega × R(theta) * r_rot
    VX = cos_t * vx - sin_t * vy - sin_t * x - cos_t * y
    VY = sin_t * vx + cos_t * vy + cos_t * x - sin_t * y
    VZ = vz
    
    return np.array([X, Y, Z, VX, VY, VZ])


def inertial_to_rotating(state_iner, t, mu=0.012):
    """
    Convert state from inertial frame to rotating (synodic) frame.
    
    Parameters
    ----------
    state_iner : array_like, shape (6,)
        State vector in inertial frame [X, Y, Z, VX, VY, VZ]
    t : float
        Current time in normalized units
    mu : float
        Mass parameter (not used in transformation, included for consistency)
    
    Returns
    -------
    state_rot : ndarray, shape (6,)
        State vector in rotating frame [x, y, z, vx, vy, vz]
    """
    X, Y, Z, VX, VY, VZ = state_iner
    
    # Rotation angle (omega = 1 in normalized units)
    theta = t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Position transformation: R(-theta) * r_iner
    x = cos_t * X + sin_t * Y
    y = -sin_t * X + cos_t * Y
    z = Z
    
    # Velocity transformation: R(-theta) * v_iner - omega × r_rot
    vx = cos_t * VX + sin_t * VY + y
    vy = -sin_t * VX + cos_t * VY - x
    vz = VZ
    
    return np.array([x, y, z, vx, vy, vz])


def convert_trajectory_to_inertial(states_rot, times, mu):
    """
    Convert an entire trajectory from rotating to inertial frame.
    
    Parameters
    ----------
    states_rot : ndarray, shape (6, N)
        State vectors in rotating frame at each time point
    times : ndarray, shape (N,)
        Time points corresponding to each state
    mu : float
        Mass parameter
    
    Returns
    -------
    states_iner : ndarray, shape (6, N)
        State vectors in inertial frame at each time point
    """
    N = states_rot.shape[1]
    states_iner = np.zeros_like(states_rot)
    
    for i in range(N):
        states_iner[:, i] = rotating_to_inertial(states_rot[:, i], times[i], mu)
    
    return states_iner


def convert_trajectory_to_rotating(states_iner, times, mu):
    """
    Convert an entire trajectory from inertial to rotating frame.
    
    Parameters
    ----------
    states_iner : ndarray, shape (6, N)
        State vectors in inertial frame at each time point
    times : ndarray, shape (N,)
        Time points corresponding to each state
    mu : float
        Mass parameter
    
    Returns
    -------
    states_rot : ndarray, shape (6, N)
        State vectors in rotating frame at each time point
    """
    N = states_iner.shape[1]
    states_rot = np.zeros_like(states_iner)
    
    for i in range(N):
        states_rot[:, i] = inertial_to_rotating(states_iner[:, i], times[i], mu)
    
    return states_rot


# ============================================================================
# UNIT CONVERSION HELPERS
# ============================================================================

def compute_characteristic_scales(m1, m2, d, G=6.6726e-11):
    """
    Compute characteristic scales for CR3BP normalization.
    
    Parameters
    ----------
    m1 : float
        Mass of primary 1 in kg
    m2 : float
        Mass of primary 2 in kg
    d : float
        Distance between primaries in meters
    G : float, optional
        Gravitational constant (default: 6.67430e-11 m^3 kg^-1 s^-2)
    
    Returns
    -------
    scales : dict
        Dictionary containing:
        - 'l_star': characteristic length (m)
        - 't_star': characteristic time (s)
        - 'v_star': characteristic velocity (m/s)
        - 'a_star': characteristic acceleration (m/s^2)
        - 'mu': mass parameter
    """
    m_total = m1 + m2
    l_star = d
    t_star = np.sqrt(d**3 / (G * m_total))
    v_star = l_star / t_star
    a_star = l_star / t_star**2
    mu = m2 / m_total
    
    return {
        'l_star': l_star,
        't_star': t_star,
        'v_star': v_star,
        'a_star': a_star,
        'mu': mu
    }


def position_to_normalized(r_SI, l_star):
    """Convert position from SI (m) to normalized units."""
    return r_SI / l_star


def position_to_SI(r_norm, l_star):
    """Convert position from normalized units to SI (m)."""
    return r_norm * l_star


def velocity_to_normalized(v_SI, v_star):
    """Convert velocity from SI (m/s) to normalized units."""
    return v_SI / v_star


def velocity_to_SI(v_norm, v_star):
    """Convert velocity from normalized units to SI (m/s)."""
    return v_norm * v_star


def time_to_normalized(t_SI, t_star):
    """Convert time from SI (s) to normalized units."""
    return t_SI / t_star


def time_to_SI(t_norm, t_star):
    """Convert time from normalized units to SI (s)."""
    return t_norm * t_star


def state_to_normalized(state_SI, l_star, v_star):
    """
    Convert state vector from SI to normalized units.
    
    Parameters
    ----------
    state_SI : array_like, shape (6,)
        State in SI units [x(m), y(m), z(m), vx(m/s), vy(m/s), vz(m/s)]
    l_star : float
        Characteristic length (m)
    v_star : float
        Characteristic velocity (m/s)
    
    Returns
    -------
    state_norm : ndarray, shape (6,)
        State in normalized units
    """
    state_SI = np.asarray(state_SI)
    state_norm = np.zeros(6)
    state_norm[:3] = position_to_normalized(state_SI[:3], l_star)
    state_norm[3:] = velocity_to_normalized(state_SI[3:], v_star)
    return state_norm


def state_to_SI(state_norm, l_star, v_star):
    """
    Convert state vector from normalized to SI units.
    
    Parameters
    ----------
    state_norm : array_like, shape (6,)
        State in normalized units
    l_star : float
        Characteristic length (m)
    v_star : float
        Characteristic velocity (m/s)
    
    Returns
    -------
    state_SI : ndarray, shape (6,)
        State in SI units [x(m), y(m), z(m), vx(m/s), vy(m/s), vz(m/s)]
    """
    state_norm = np.asarray(state_norm)
    state_SI = np.zeros(6)
    state_SI[:3] = position_to_SI(state_norm[:3], l_star)
    state_SI[3:] = velocity_to_SI(state_norm[3:], v_star)
    return state_SI


def trajectory_to_normalized(states_SI, l_star, v_star):
    """
    Convert trajectory from SI to normalized units.
    
    Parameters
    ----------
    states_SI : ndarray, shape (6, N)
        States in SI units at each time point
    l_star : float
        Characteristic length (m)
    v_star : float
        Characteristic velocity (m/s)
    
    Returns
    -------
    states_norm : ndarray, shape (6, N)
        States in normalized units
    """
    states_norm = np.zeros_like(states_SI)
    states_norm[:3, :] = position_to_normalized(states_SI[:3, :], l_star)
    states_norm[3:, :] = velocity_to_normalized(states_SI[3:, :], v_star)
    return states_norm


def trajectory_to_SI(states_norm, l_star, v_star):
    """
    Convert trajectory from normalized to SI units.
    
    Parameters
    ----------
    states_norm : ndarray, shape (6, N)
        States in normalized units at each time point
    l_star : float
        Characteristic length (m)
    v_star : float
        Characteristic velocity (m/s)
    
    Returns
    -------
    states_SI : ndarray, shape (6, N)
        States in SI units
    """
    states_SI = np.zeros_like(states_norm)
    states_SI[:3, :] = position_to_SI(states_norm[:3, :], l_star)
    states_SI[3:, :] = velocity_to_SI(states_norm[3:, :], v_star)
    return states_SI