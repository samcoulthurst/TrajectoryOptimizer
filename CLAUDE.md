# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trajectory optimization for the Circular Restricted Three-Body Problem (CR3BP), computing two-impulse Earth-Moon transfers from Low Earth Orbit (LEO) to Low Moon Orbit (LMO). Academic computing project.

## Setup

```bash
pip install -e .
```

Core dependencies: numpy, scipy, matplotlib. Optional: cyipopt (NLP optimization), pandas (results handling). Animation export requires ffmpeg (MP4) or pillow (GIF).

## Architecture

All source code lives in the `cr3bp/` package. Module dependency flow:

```
system.py          CR3BPSystem class, physical constants, create_earth_moon_system()
    ↓
dynamics.py        CR3BP equations of motion, DOP853 integration (solve_CR3BP),
                   orbit state generators (circular_orbit_state_earth, decision_to_state0)
    ↓
conversions.py     Rotating ↔ Inertial frame transforms, SI ↔ Normalized unit conversions
    ↓
objective.py       Cost function (minimize total Δv) and constraint (reach LMO radius)
    ↓
grid_search.py     Brute-force 4D grid search over decision variables
nlp.py             IPOPT-based NLP optimizer via cyipopt (finite-difference gradients)
    ↓
plotting.py        Static 2D trajectory plots (rotating or inertial frame)
visualization.py   Animated trajectory visualization with fading trails
```

## Key Conventions

- **Normalized units**: All dynamics use CR3BP non-dimensional units scaled by characteristic quantities (l_star, t_star, v_star) from `CR3BPSystem`. Convert to/from SI via `conversions.py`.
- **Rotating frame**: Primary coordinate system — primaries are fixed on the x-axis. Inertial frame used only for visualization.
- **Decision variables** (4D optimization): `theta` (departure angle on LEO), `delta_v` (impulse magnitude), `delta_v_angle` (boost direction relative to velocity), `tof` (time of flight).
- **Integration**: DOP853 with rtol=1e-12, atol=1e-12 throughout.
- **Mass parameter**: μ = m2/(m1+m2) ≈ 0.012145 for Earth-Moon.

## Running Optimizations

- **Grid search**: `notebooks/main.ipynb` — runs 4D grid search (default 40×20×20×40 = 640k evaluations), saves feasible solutions to `.npy` backup files.
- **NLP optimization**: `notebooks/new_main.ipynb` — refines solutions with IPOPT (currently has integration step-size errors).
- **View results**: `notebooks/results_viewer.ipynb` — loads cached `.npy` results and generates plots/animations.

## Known Issues

- NLP optimizer (`nlp.py`) fails with "Required step size is less than spacing between numbers" during integration — likely caused by extreme initial conditions from the optimizer.
- No automated test suite exists.
