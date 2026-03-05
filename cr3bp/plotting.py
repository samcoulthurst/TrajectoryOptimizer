"""
CR3BP Plotting Utilities

This module contains visualization functions for CR3BP trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

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
        #plt.plot(states[0, 0], states[1, 0], 'go', markersize=10, label='Start')
        #plt.plot(states[0, -1], states[1, -1], 'ro', markersize=10, label='End')

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
        #plt.plot(states_inertial[0, 0], states_inertial[1, 0],
        #       'go', markersize=10, label='Start')
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

def contour_plot(results_df, var1_name, var2_name, grid_size, optimal_params, fontsize=14):
    
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    idx1 = var_names.index(var1_name)
    idx2 = var_names.index(var2_name)
    
    feasible = results_df.dropna(subset=['total_delta_v'])
    feasible = feasible.sort_values(var1_name)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(feasible[var1_name], feasible[var2_name], 
                        c=feasible['total_delta_v'], cmap='turbo', 
                        vmin=3.7, vmax=4.5, s=10)
    cbar = fig.colorbar(scatter, ax=ax, label='Total Δv')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('Total Δv', fontsize=fontsize)
    
    ax.plot(optimal_params[idx1], optimal_params[idx2], 'r*', markersize=15, label='Optimal')
    ax.set_xlabel(var1_name, fontsize=fontsize)
    ax.set_ylabel(var2_name, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # max 6 ticks
    ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_1d(results_df, var_name, var_delta, optimal_params):
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    idx = var_names.index(var_name)
    
    feasible = results_df.dropna(subset=['total_delta_v'])
    feasible = feasible.sort_values(var_name)

    plt.figure(figsize=(10, 6))
    plt.plot(feasible[var_name], feasible['total_delta_v'], 'b-', linewidth=1.5, label='Total Delta-v')
    plt.plot(optimal_params[idx], feasible['total_delta_v'].min(), 'r*', markersize=15, label='Optimal')

    plt.xlim(optimal_params[idx] - var_delta, optimal_params[idx] + var_delta)
    plt.xlabel(var_name)
    plt.ylabel('Total Delta-v')
    plt.title(f'Total Delta-v vs {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyse_sweep(results_df, cr3bp, fontsize=14):
    """
    Find the optimal (lowest total_delta_v) from the sweep results and
    plot all 6 pairwise 2D scatter slices through the 4D parameter space.

    Values are converted to dimensional units and plotted relative to the
    optimal solution.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from parameter_sweep. Must contain columns:
        theta, delta_v, delta_v_angle, tof, total_delta_v, distance_to_lmo.
        All values in normalised CR3BP units.
    cr3bp : object
        CR3BP system object with attributes l_star, t_star, v_star.
    fontsize : int, optional
        Font size for labels and ticks.

    Returns
    -------
    optimal_row : pd.Series
        The row with the lowest total_delta_v (normalised units).
    fig : matplotlib.figure.Figure
        The 2x3 subplot figure.
    """
    # Find new optimal
    feasible = results_df.dropna(subset=['total_delta_v']).copy()

    if feasible.empty:
        print("No feasible solutions found in sweep.")
        return None, None

    optimal_idx = feasible['total_delta_v'].idxmin()
    optimal_row = feasible.loc[optimal_idx]

    print("Optimal solution from sweep:")
    print(f"  theta:         {optimal_row['theta']:.6f} rad")
    print(f"  delta_v:       {optimal_row['delta_v'] * cr3bp.v_star * 1e-3:.4f} km/s")
    print(f"  delta_v_angle: {optimal_row['delta_v_angle']:.6f} rad")
    print(f"  tof:           {optimal_row['tof'] * cr3bp.t_star / 86400:.4f} days")
    print(f"  total_delta_v: {optimal_row['total_delta_v'] * cr3bp.v_star * 1e-3:.4f} km/s")
    print(f"  distance_to_lmo: {optimal_row['distance_to_lmo'] * cr3bp.l_star * 1e-3:.4f} km")

    # Dimensional conversion factors and labels
    var_names = ['theta', 'delta_v', 'delta_v_angle', 'tof']
    var_config = {
        'theta':         {'symbol': r'$\theta$',      'unit': 'rad',  'scale': 1.0},
        'delta_v':       {'symbol': r'$\Delta v$',    'unit': 'km/s', 'scale': cr3bp.v_star * 1e-3},
        'delta_v_angle': {'symbol': r'$\alpha$',      'unit': 'rad',  'scale': 1.0},
        'tof':           {'symbol': r'$T$',           'unit': 'days', 'scale': cr3bp.t_star / 86400},
    }

    pairs = list(combinations(var_names, 2))  # 6 pairs

    # Convert total_delta_v to km/s for colour scale
    dv_scale = cr3bp.v_star * 1e-3
    feasible_dv_dim = feasible['total_delta_v'] * dv_scale
    vmin = 3.7 * dv_scale
    vmax = 4.5 * dv_scale

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, (var1, var2) in enumerate(pairs):
        ax = axes[i]
        cfg1 = var_config[var1]
        cfg2 = var_config[var2]

        x = (feasible[var1] - optimal_row[var1]) * cfg1['scale']
        y = (feasible[var2] - optimal_row[var2]) * cfg2['scale']

        scatter = ax.scatter(x, y,
                             c=feasible_dv_dim, cmap='viridis',
                             vmin=vmin, vmax=vmax, s=15)

        ax.plot(0, 0, 'r*', markersize=15)

        ax.set_xlabel(f"{cfg1['symbol']} ({cfg1['unit']})", fontsize=fontsize)
        ax.set_ylabel(f"{cfg2['symbol']} ({cfg2['unit']})", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.ticklabel_format(style='scientific', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(fontsize - 2)
        ax.yaxis.get_offset_text().set_fontsize(fontsize - 2)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.subplots_adjust(right=0.88, wspace=0.35, hspace=0.35)

    # Place colourbar in dedicated axis on the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(r'Total $\Delta v$ (km/s)', fontsize=fontsize)
    plt.show()

    return optimal_row, fig