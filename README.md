# TrajectoryOptimizer
Trajectory Optimizier for CR3BP completed for Computing Project Module

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a virtual environment with the pinned Python 3.11 and installs
`cr3bp` in editable mode.

## Running the notebooks

```bash
uv run jupyter lab
```

Open `notebooks/main.ipynb` (grid search) or `notebooks/results_viewer.ipynb`
(trajectory plots and animation).

> Animation export to `.mp4` requires `ffmpeg` on your PATH
> (`cr3bp/visualization.py` falls back to the `pillow` writer for `.gif`).
