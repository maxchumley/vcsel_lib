# Reinforcement-learning coupling design

This package contains the RL coupling-design experiments and their inspection
tools. Run programs from the repository root with module syntax so package
imports and artifact paths are consistent:

```bash
python -m rl.reinforcement_matrix_design
python -m rl.neural_policy_matrix_design
python -m rl.conditional_coupling_run
python -m rl.conditional_coupling_run_variable_m
python -m rl.inspect_saved_coupling_model
python -m rl.inspect_saved_coupling_model_variable_m
python -m rl.inspect_power_vs_lasers
```

Generated files are kept beneath `rl/artifacts/`:

- `models/`: PyTorch checkpoints.
- `results/`: plots and training progress.
- `data/`: replay data and NumPy design outputs.

The artifact directories are ignored by Git. Source code, tests, and this
documentation remain version-controlled.
