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

To scan three-laser coupling topologies for the first budget with a stable
frequency-locked equilibrium, run:

```bash
python -m examples.stability.three_laser_budget_map --grid 5 --budget-step 0.5 --max-budget 40
```

In an editor that supports `#%%` cells, open
`stability/three_laser_budget_map.py`, run the imports/functions cell first,
then run the scan cell. Its settings are editable there, including
`JOBS` for parallel pixel workers, and it displays the completed heatmap inline.
The following `#%%` cell loads the lowest threshold from the saved CSV,
ramps that exact coupling matrix from zero, and plots three laser frequencies,
wrapped relative phases, and optical powers like `basics/simple_example.py`.
It saves a separate `*_optimum_timeseries.png` figure.
The last `#%%` cell independently loads the same CSV and sweeps the coupling
budget at the optimal topology, plotting all found equilibrium frequency and
coherent-intensity branches by stability. It saves `*_optimum_branches.png`
and `*_optimum_branches.csv`. Its horizontal axis and CSV use the sum of all
six off-diagonal matrix entries, twice the unique-link budget used in the
heatmap. Its equilibrium seed grid and sparse DDE stability settings match
`basics/equilibrium_stability_tutorial.ipynb`. Run the imports/functions cell before either
plotting cell; the time simulation cell is optional.
For terminal runs, use `--jobs 4` (or `--jobs 1` for serial execution).
Each worker builds the trial coupling matrices for one pixel with NumPy;
equilibrium roots and stability spectra are still solved one trial at a time
by `VCSEL`. The run prints each pixel's status, periodic budget updates for
long pixels, and the number of completed pixels.

The script writes a CSV and PNG under `stability/results/`. Its detuning gaps
are 1 and 2 GHz by default. Edit `UNCENTERED_DETUNINGS_GHZ` to set the three
laser frequencies in GHz; the script subtracts their mean before converting
them to angular frequencies. Reciprocal coupling phases are zero, the intrinsic phase
model is `standard_lk`, and the plotted budget
is `a12+a13+a23` in ns^-1. Increase `--grid`, `--phase-count`,
`--freq-count`, and `--collocation` for a more resolved scan.
Reduce `--budget-step` for a finer threshold estimate. The search always tests
the exact `--max-budget` (40 ns^-1 by default), even when the step does not
divide it evenly. Disconnected topologies and connected topologies with no
stable branch found by that maximum appear black; their different statuses are
recorded in the CSV. The heatmap records the first *sampled* stable budget,
not a noisy contrast measurement.

New examples should live in the closest category and write generated files to
that category's `data/` or `results/` directory.
