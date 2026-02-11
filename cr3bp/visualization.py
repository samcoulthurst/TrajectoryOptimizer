"""
CR3BP Animation Utilities

This module contains animation functions for CR3BP trajectory visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .conversions import convert_trajectory_to_inertial


def animate_trajectory(cr3bp, states, t_span, frame='rotating', fps=30,
                       trail_length=None, save_path=None):
    """
    Create an animated visualization of a CR3BP trajectory.

    Parameters
    ----------
    cr3bp : CR3BPSystem
        CR3BP system instance (uses .mu for mass parameter).
    states : ndarray, shape (6, N)
        Spacecraft trajectory state vectors [x, y, z, vx, vy, vz] in the
        rotating frame at each time point.
    t_span : ndarray, shape (N,)
        Non-dimensional time points corresponding to each state.
    frame : str, optional
        Reference frame for visualization. Either 'rotating' or 'inertial'.
        Default is 'rotating'.
    fps : int, optional
        Frames per second for the animation. Default is 30.
    trail_length : int or None, optional
        Number of past trajectory points to display as a fading trail.
        If None, the full trajectory history is shown. Default is None.
    save_path : str or None, optional
        File path to save the animation. Supports '.mp4' (requires ffmpeg)
        and '.gif' (uses pillow). If None, the animation is not saved.
        Default is None.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. Must be kept in a variable to prevent
        garbage collection. Can be displayed in a Jupyter notebook via
        ``HTML(anim.to_html5_video())``.
    """
    mu = cr3bp.mu
    N = states.shape[1]

    if frame not in ('rotating', 'inertial'):
        raise ValueError(f"frame must be 'rotating' or 'inertial', got '{frame}'")

    if states.shape[0] != 6:
        raise ValueError(f"states must have shape (6, N), got {states.shape}")

    if t_span.shape[0] != N:
        raise ValueError(
            f"t_span length {t_span.shape[0]} does not match states columns {N}"
        )

    # Prepare trajectory data in target frame
    if frame == 'inertial':
        plot_states = convert_trajectory_to_inertial(states, t_span, mu)
    else:
        plot_states = states

    x_traj = plot_states[0, :]
    y_traj = plot_states[1, :]

    # Pre-compute axis limits
    x_min, x_max = x_traj.min(), x_traj.max()
    y_min, y_max = y_traj.min(), y_traj.max()

    if frame == 'rotating':
        x_min = min(x_min, -mu, 1 - mu)
        x_max = max(x_max, -mu, 1 - mu)
        y_min = min(y_min, 0)
        y_max = max(y_max, 0)
    else:
        body_extent = max(mu, 1 - mu)
        x_min = min(x_min, -body_extent)
        x_max = max(x_max, body_extent)
        y_min = min(y_min, -body_extent)
        y_max = max(y_max, body_extent)

    x_pad = 0.1 * (x_max - x_min) or 0.1
    y_pad = 0.1 * (y_max - y_min) or 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # Trail color (lime green RGB)
    trail_rgb = np.array([0.0, 1.0, 0.0])

    # Set up figure with dark background
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(1.3*x_min, 1.3*x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')

    frame_label = 'Rotating' if frame == 'rotating' else 'Inertial'
    ax.set_title(f'LEO to LMO Trajectory - {frame_label} Frame')
    ax.grid(True, alpha=0.2)

    # Create artists
    earth_marker, = ax.plot([], [], 'o', color='deepskyblue', markersize=24,
                            label='Earth', zorder=5)
    moon_marker, = ax.plot([], [], 'o', color='silver', markersize=14,
                           label='Moon', zorder=5)
    sc_marker, = ax.plot([], [], 'o', color='lime', markersize=10,
                           label='Rocket', zorder=10)

    trail_collection = LineCollection([], linewidths=1.5, zorder=4)
    ax.add_collection(trail_collection)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        color='white', fontsize=11,
                        verticalalignment='top',
                        fontfamily='monospace')

    ax.legend(loc='upper right', fontsize=9)

    def init():
        earth_marker.set_data([], [])
        moon_marker.set_data([], [])
        sc_marker.set_data([], [])
        trail_collection.set_segments([])
        time_text.set_text('')
        return earth_marker, moon_marker, sc_marker, trail_collection, time_text

    def update(i):
        # Update body positions
        if frame == 'rotating':
            earth_marker.set_data([-mu], [0])
            moon_marker.set_data([1 - mu], [0])
        else:
            theta = t_span[i]
            earth_marker.set_data([-mu * np.cos(theta)],
                                  [-mu * np.sin(theta)])
            moon_marker.set_data([(1 - mu) * np.cos(theta)],
                                 [(1 - mu) * np.sin(theta)])

        # Update spacecraft position
        sc_marker.set_data([x_traj[i]], [y_traj[i]])

        # Update fading trail
        if trail_length is None:
            start = 0
        else:
            start = max(0, i - trail_length)

        if i > start:
            trail_x = x_traj[start:i + 1]
            trail_y = y_traj[start:i + 1]

            # Build LineCollection segments: (M-1, 2, 2)
            points = np.column_stack([trail_x, trail_y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # RGBA colors with fading alpha
            num_seg = len(segments)
            colors = np.zeros((num_seg, 4))
            colors[:, 0] = trail_rgb[0]
            colors[:, 1] = trail_rgb[1]
            colors[:, 2] = trail_rgb[2]
            colors[:, 3] = np.linspace(0.05, 1.0, num_seg)

            trail_collection.set_segments(segments)
            trail_collection.set_colors(colors)
        else:
            trail_collection.set_segments([])

        # Update time counter
        time_text.set_text(f't = {t_span[i]:.4f}')

        return earth_marker, moon_marker, sc_marker, trail_collection, time_text

    # Create animation
    interval = 250.0 / fps
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N,
        interval=interval,
        blit=True
    )

    # Save if requested
    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.mp4':
            writer = 'ffmpeg'
        elif ext == '.gif':
            writer = 'pillow'
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Use '.mp4' or '.gif'."
            )
        anim.save(save_path, writer=writer, fps=fps)

    return anim
