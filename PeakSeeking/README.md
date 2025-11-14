# Peak Seeking RL Demo

Standardized reinforcement learning demo that separates the custom Gymnasium environment from the training logic.

## Project Layout

```
├── peak_seeking/            # Python package with env + utilities
│   ├── envs/
│   └── training/
├── configs/                 # YAML config files
├── scripts/                 # Entry points (training, evaluation, etc.)
├── tests/                   # Unit tests
└── PeakSeekingEnv.py        # Thin wrapper kept for backwards compatibility
```

## Quickstart

1. Create/activate a virtual environment (optional but recommended).
2. Install dependencies:

```cmd
.venv\Scripts\python.exe -m pip install -e .
```

3. Train a Q-learning agent using the default configuration:

```cmd
python scripts\train_peak_seeking.py
```

You can override parameters on the CLI, for example limiting the number of episodes or enabling rendering:

```cmd
python scripts\train_peak_seeking.py --episodes 50 --render
```

Artifacts such as the learned Q-table are stored in the `runs/` directory. Adjust hyper-parameters via `configs/peak_seeking.yaml` or pass overrides through command-line options.

## Testing

```cmd
python -m pytest
```
