"""
CR3BP Plotting Utilities

This module contains visualization functions for CR3BP trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, mu, times=None, frame="Rotating"):
    """
    Plot a trajectory in the CR3BP rotating or inertial frame.

    Parameters
    ----------
    states : ndarray, shape (6, N)
        State vectors [x, y, z, vx, vy, vz] at each time point
    mu : float
        Mass parameter of the CR3BP system
    times : ndarray, shape (N,), optional
        Time points (required for inertial frame)
    frame : str, optional
        "Rotating" or "Inertial" (default: "Rotating")
    """
    if frame == "Rotating":
        plt.figure(figsize=(10, 8))

        # Plot trajectory
        plt.plot(states[0, :], states[1, :], 'b-', linewidth=1.5, label='Trajectory')

        # Plot primaries
        plt.plot(-mu, 0, 'ko', markersize=15, label='Earth')
        plt.plot(1-mu, 0, 'gray', marker='o', markersize=8, label='Moon')

        # Plot start/end
        plt.plot(states[0, 0], states[1, 0], 'go', markersize=10, label='Start')
        plt.plot(states[0, -1], states[1, -1], 'ro', markersize=10, label='End')

        plt.xlabel('x (normalized)')
        plt.ylabel('y (normalized)')
        plt.title('Rotating Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

    elif frame == "Inertial":
        if times is None:
            raise ValueError("times is required for inertial frame plotting")

        from .conversions import convert_trajectory_to_inertial
        states_inertial = convert_trajectory_to_inertial(states, times, mu)
        plt.figure(figsize=(10, 8))

        # Plot trajectory in inertial frame
        plt.plot(states_inertial[0, :], states_inertial[1, :],
                'b-', linewidth=1.5, label='Trajectory')

        # Plot start and end positions
        plt.plot(states_inertial[0, 0], states_inertial[1, 0],
                'go', markersize=10, label='Start')
        #plt.plot(states_inertial[0, -1], states_inertial[1, -1],
        #        'ro', markersize=10, label='End')

        # Plot Earth and Moon at multiple time snapshots to show rotation
        n_snapshots = 20
        snapshot_indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

        for i, idx in enumerate(snapshot_indices):
            t = times[idx]

            # Earth position (rotates around barycenter)
            earth_x = -mu * np.cos(t)
            earth_y = -mu * np.sin(t)

            # Moon position (rotates around barycenter)
            moon_x = (1 - mu) * np.cos(t)
            moon_y = (1 - mu) * np.sin(t)

            # Fade from light to dark as time progresses
            alpha = 0.2 + 0.8 * (i / n_snapshots)

            if i == 0:
                plt.plot(earth_x, earth_y, 'ko', markersize=8, alpha=alpha, label='Earth positions')
                plt.plot(moon_x, moon_y, 'gray', marker='o', markersize=5, alpha=alpha, label='Moon positions')
            else:
                plt.plot(earth_x, earth_y, 'ko', markersize=8, alpha=alpha)
                plt.plot(moon_x, moon_y, 'gray', marker='o', markersize=5, alpha=alpha)

        # Plot Earth and Moon orbits
        theta = np.linspace(0, 2*np.pi, 100)
        earth_orbit_x = -mu * np.cos(theta)
        earth_orbit_y = -mu * np.sin(theta)
        moon_orbit_x = (1 - mu) * np.cos(theta)
        moon_orbit_y = (1 - mu) * np.sin(theta)

        plt.plot(earth_orbit_x, earth_orbit_y, 'k--', alpha=0.3, linewidth=1, label='Earth orbit')
        plt.plot(moon_orbit_x, moon_orbit_y, 'gray', linestyle='--', alpha=0.3, linewidth=1, label='Moon orbit')

        plt.xlabel('X (normalized)', fontsize=12)
        plt.ylabel('Y (normalized)', fontsize=12)
        plt.title('Inertial Frame', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


from scipy.interpolate import griddata

def contour_plot(results_df, var1_name, var2_name, grid_size, optimal_params):
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    idx1 = var_names.index(var1_name)
    idx2 = var_names.index(var2_name)
    
    V1 = results_df[var1_name].values.reshape(grid_size, grid_size)
    V2 = results_df[var2_name].values.reshape(grid_size, grid_size)
    objective = results_df['total_delta_v'].values.reshape(grid_size, grid_size)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.pcolormesh(V1, V2, objective, cmap='viridis', shading='auto')
    fig.colorbar(contour, ax=ax, label='Total Delta-v')
    
    ax.plot(optimal_params[idx1], optimal_params[idx2], 'r*', markersize=15, label='Optimal')
    
    ax.set_xlabel(var1_name)
    ax.set_ylabel(var2_name)
    ax.set_title(f'Total Delta-v: {var1_name} vs {var2_name}')
    ax.legend()
    plt.tight_layout()
    plt.show()