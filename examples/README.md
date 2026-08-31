# Examples

These programs are research workflows, not part of the installed library API.
Many generate large arrays, plots, or videos; the generated-data directories
are ignored by Git.

The examples are grouped by purpose, with generated inputs and outputs stored
beside the workflows that use them:

- `basics/`: introductory simulations and order-parameter demonstrations
- `coupling/`: coupling sweeps, bifurcations, and phase-dependent coupling
- `injection/`: injection locking, steering, and robustness studies
- `linewidth/`: linewidth, optical-spectrum, and RIN calculations
- `stability/`: equilibrium and stability-manifold workflows
- `paper/`: paper-specific validation and figure reproduction
- `visualization/`: far-field and animation helpers

Start with `basics/simple_example.py`, `stability/verify_stability.py`, or
`stability/stability_manifold.py`. Shared paths are defined in `_paths.py`, so
scripts resolve category-local `data/` and `results/` directories independently
of the current working directory. From the repository root, module execution is
the most reliable form, for example:

```bash
python -m examples.basics.simple_example
```

New examples should live in the closest category and write generated files to
that category's `data/` or `results/` directory.
