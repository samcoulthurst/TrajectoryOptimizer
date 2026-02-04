"""
Circular orbit state computation utilities for CR3BP.

Provides functions to compute state vectors for circular orbits around
Earth or Moon in the CR3BP rotating frame.
"""

import numpy as np

# Default Earth-Moon system parameters
L_STAR_KM = 384400.0  # Earth-Moon distance in km
R_EARTH_KM = 6371.0   # Earth radius in km
R_MOON_KM = 1737.0    # Moon radius in km


def circular_orbit_state_earth(theta, altitude_km, inclination=0.0, mu=0.01215):
    """
    Compute state vector for circular orbit around Earth in CR3BP rotating frame.

    Parameters
    ----------
    theta : float
        True anomaly / angle on orbit (radians), measured from +x axis
    altitude_km : float
        Altitude above Earth surface (km)
    inclination : float, optional
        Orbital inclination (radians), default 0 (equatorial)
    mu : float
        CR3BP mass parameter (default Earth-Moon value)

    Returns
    -------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in normalized rotating frame
    v_circ : float
        Circular orbit velocity magnitude (normalized)
    """
    # Normalized radii
    r_earth = R_EARTH_KM / L_STAR_KM
    r_orbit = r_earth + altitude_km / L_STAR_KM

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
    vx_rel = vx_orb
    vy_rel = vy_orb * np.cos(inclination)
    vz_rel = vy_orb * np.sin(inclination)

    # Convert to rotating frame velocity
    # v_rot = v_inertial - omega x r, where omega = [0, 0, 1]
    # omega x r = [-y, x, 0]
    # So v_rot = v_inertial + [y, -x, 0]
    vx = vx_rel + y
    vy = vy_rel - x
    vz = vz_rel

    return np.array([x, y, z, vx, vy, vz]), v_circ


def circular_orbit_state_moon(theta, altitude_km, inclination=0.0, mu=0.01215):
    """
    Compute state vector for circular orbit around Moon in CR3BP rotating frame.

    Parameters
    ----------
    theta : float
        True anomaly / angle on orbit (radians), measured from +x axis
    altitude_km : float
        Altitude above Moon surface (km)
    inclination : float, optional
        Orbital inclination (radians), default 0 (equatorial)
    mu : float
        CR3BP mass parameter (default Earth-Moon value)

    Returns
    -------
    state : ndarray, shape (6,)
        State vector [x, y, z, vx, vy, vz] in normalized rotating frame
    v_circ : float
        Circular orbit velocity magnitude (normalized)
    """
    # Normalized radii
    r_moon = R_MOON_KM / L_STAR_KM
    r_orbit = r_moon + altitude_km / L_STAR_KM

    # Position in orbital plane
    x_orb = r_orbit * np.cos(theta)
    y_orb = r_orbit * np.sin(theta)

    # Apply inclination rotation
    x_rel = x_orb
    y_rel = y_orb * np.cos(inclination)
    z_rel = y_orb * np.sin(inclination)

    # Shift to barycentric frame (Moon at 1-mu)
    x = (1 - mu) + x_rel
    y = y_rel
    z = z_rel

    # Circular velocity magnitude (Moon-centered)
    v_circ = np.sqrt(mu / r_orbit)

    # Velocity in orbital plane
    vx_orb = -v_circ * np.sin(theta)
    vy_orb = v_circ * np.cos(theta)

    # Apply inclination rotation
    vx_rel = vx_orb
    vy_rel = vy_orb * np.cos(inclination)
    vz_rel = vy_orb * np.sin(inclination)

    # Convert to rotating frame velocity
    vx = vx_rel + y
    vy = vy_rel - x
    vz = vz_rel

    return np.array([x, y, z, vx, vy, vz]), v_circ


def compute_lmo_insertion_dv(state_arrival, altitude_km, mu=0.01215, prograde=True):
    """
    Compute delta-V required to insert into circular LMO from arrival state.

    Parameters
    ----------
    state_arrival : ndarray, shape (6,)
        Arrival state at Moon [x, y, z, vx, vy, vz] in rotating frame
    altitude_km : float
        Target LMO altitude (km)
    mu : float
        CR3BP mass parameter
    prograde : bool
        If True, insert into prograde orbit; if False, retrograde

    Returns
    -------
    dv2 : ndarray, shape (3,)
        Required delta-V vector for insertion (normalized)
    dv2_mag : float
        Magnitude of delta-V (normalized)
    """
    x, y, z, vx, vy, vz = state_arrival

    # Position relative to Moon
    r_rel = np.array([x - (1 - mu), y, z])
    r_mag = np.linalg.norm(r_rel)
    r_hat = r_rel / r_mag

    # Target circular velocity magnitude at LMO
    r_moon = R_MOON_KM / L_STAR_KM
    r_target = r_moon + altitude_km / L_STAR_KM
    v_circ = np.sqrt(mu / r_target)

    # Velocity direction: perpendicular to r_hat, in orbital plane
    # For prograde orbit, angular momentum should point roughly in +z direction
    z_hat = np.array([0.0, 0.0, 1.0])
    v_dir = np.cross(z_hat, r_hat)
    v_dir_norm = np.linalg.norm(v_dir)

    if v_dir_norm < 1e-10:
        # Polar arrival (r_hat aligned with z), use alternate reference
        v_dir = np.array([0.0, 1.0, 0.0])
    else:
        v_dir = v_dir / v_dir_norm

    if not prograde:
        v_dir = -v_dir

    # Required velocity in Moon-centered inertial frame
    v_circ_inertial = v_circ * v_dir

    # Convert to rotating frame: v_rot = v_inert - omega x r_bary
    # omega x [x, y, z] = [-y, x, 0]
    # v_rot = v_inert + [y, -x, 0]
    v_required = v_circ_inertial.copy()
    v_required[0] += y
    v_required[1] -= x

    # Delta-V is difference between required and actual
    v_arrival = np.array([vx, vy, vz])
    dv2 = v_required - v_arrival
    dv2_mag = np.linalg.norm(dv2)

    return dv2, dv2_mag
