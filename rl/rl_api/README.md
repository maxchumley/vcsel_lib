# VCSEL reinforcement learning API

Configure an experiment, train a policy, propose couplings, evaluate them, and
plot the results. Each step is explicit. The implementation uses ordinary
functions, small configuration classes, and NumPy arrays, following the style
of `vcsel_lib.py`.

**This version reproduces the current paired-topology training recipe.** It
reuses the existing numerical model, simulator, rewards, and gradient routines.
All new source and default outputs are under `rl/rl_api/`. Existing training
scripts and checkpoints are separate. Bernoulli gate training is deferred.

## Start here

**Next experiment:** use `experiments/train_mirrored_root_encoder_1000.py`.
It adds a binary root indicator to each laser's encoder features so the
network knows which reference it is using. All other settings match the
completed root-reference run, including the phase formula. Restart the
Python kernel, then run the new file's single `#%%` cell to start fresh.
See [the root-encoder setup](docs/mirrored_root_encoder_run.md).

**Root-reference comparison:** `experiments/train_mirrored_root_delay_1000.py`
It matches the random-reachable mirrored run, except each topology uses its
sampled root's detuning for the coupling-phase delay reference. Training,
validation, and loaded inspection retain the same root choice. It starts
fresh under a separate run name; run its single `#%%` cell when ready.
See [the root-reference setup](docs/mirrored_root_delay_run.md).

**Mirrored baseline:** `experiments/train_mirrored_1000.py` compares
three ordinary mirrored action pairs plus one wide pair, with a baseline
that excludes the evaluated pair and averages the other six actions. It
retains the 1,000-iteration schedule, RMS, random link counts, 21 PCGrad tasks,
and 1,400 simulations per iteration. Run its single `#%%` cell when ready;
`START_TRAINING=True`. See [the mirrored-action setup](docs/mirrored_1000_run.md).

**Independent 1k baseline:** `experiments/train_anneal_1000.py` uses five
topologies per pair (star, random sparse/middle/dense, all-to-all), 1,400
simulations per iteration, and 21 size/density PCGrad tasks. All action SDs
taper from iteration 1 to exact means at 1000; LR decays over 900–1000.
RMS, target-plus-residual phases and `standard_lk` are enabled. The final
iteration is a mean-only check with no optimizer update. Run its single
`#%%` cell when ready; `START_TRAINING=True`.
See [the 1,000-iteration setup](docs/anneal_1000_run.md).

**Current residual baseline:** `experiments/train_phase_residual.py` adds the
known source-minus-receiver target phase difference to the existing phase
mean, treating the learned output as a residual. All other settings match
the current four-count run. The parameter count and simulations are identical.
Run its single `#%%` cell when ready; `START_TRAINING=True`.
See [the focused phase-residual setup](docs/phase_residual_run.md).

**Current baseline:** `experiments/train_four_counts.py` uses the working
5-bin/four-topology sampler and its 16 populated density PCGrad tasks.
It retains `standard_lk` and the current 3,000-iteration precision schedule.
Run its single `#%%` cell when ready; `START_TRAINING=True`.
See [the four-count setup](docs/four_counts_run.md).

**Five-count comparison:** `experiments/train_fixed_counts.py` compares five
fixed, evenly spaced link counts against the random-count precision run.
Wiring is still redrawn independently for every pair. All other settings,
including `standard_lk`, 21 density PCGrad tasks and the 3,000-iteration
precision schedule, match that run. It has one `#%%` cell with
`START_TRAINING=True`. See [the fixed-count setup](docs/fixed_counts_run.md).

**Random-count comparison:** `experiments/train_precision.py` uses five topologies
per pair: star, random sparse/middle/dense, and all-to-all. It keeps RMS and
3,000 iterations, fills all 21 size/density PCGrad tasks, and adds quieter
precision refinement plus a final 1,000 iterations at low LR.
Its native `phys` dictionary selects `"phase_model": "standard_lk"`; training,
validation and checkpoint inspection all use that convention.
See [the stages and run details](docs/precision_run.md).
It has one `#%%` cell with `START_TRAINING=True`; run that cell to start.
Set the flag to False if you only want to preview the settings.

The previous experiments remain available in `experiments/train_backup_sampling.py`
and `experiments/train_shared_masks.py`; their settings were not modified by this setup.

Activate your `vcsel` environment and run commands from the repository root.
No new dependency installation is needed in that environment.

**Edit settings directly in the run files:**

- `experiments/train_current.py`: native VCSEL parameters and `phys` dictionary
  first, then run/training/model/sampling/exploration/reward/validation settings
  in its `CONFIG` block.
- `experiments/inspect_current.py`: checkpoint, targets, detunings, topology,
  action sampling, evaluation, selection and plotting settings at the top.

Copy a run file to set up another experiment. JSON presets are not loaded
for training or inspection. `presets/current_training.json` is historical
reference data used only by comparison diagnostics.

Preview the recipe; this does not train, simulate, or create a run folder:

```bash
python -m rl.rl_api.experiments.train_current
```

When you decide to train, use a fresh name:

```bash
python -m rl.rl_api.experiments.train_current --start --name paired_api_01
```

The same workflow in Python is:

```python
from rl.rl_api import Experiment

experiment = Experiment("paired_api_01")
experiment.describe()
# Call explicitly when ready:
# result = experiment.train()
```

The training example has a `__main__` guard for multiprocessing. Keep that
guard when calling `.train()` from a standalone script. A nonempty run folder
raises an error by default. Call `.train(restart=True)` to archive the previous
attempt under `runs/restarts/<name>_<UTC timestamp>/` and initialize a fresh
model at the same output path. The backup-sampling run file exposes this as
`RESTART_EXISTING_RUN=True`. This starts from iteration 1; it does not resume
the saved weights or optimizer. Importing a module does not begin training.

To start from your editor, set `START_TRAINING=True` in the training run file
and run it. The default is `False`, which previews settings only. Command-line
options are optional conveniences; they override the matching run-file values.

## Run with `# %%` cells

The run files support VS Code's Python Interactive window and other IPython
cell workflows. Select the `vcsel` environment.
Start the kernel anywhere inside the repository; `__file__` is not required.

Training cells are **Imports → VCSEL parameters → Run settings → Preview → Train**. The Train
cell only starts when `START_TRAINING=True`. Rerun the settings cell after
editing it. The Train cell builds a fresh experiment from those current
values using `Experiment(...)` directly. The preview uses the same direct
construction; there is no extra factory wrapper. Completed training leaves
`result.model` and `result.history` available
in the kernel. A new full run needs a fresh run name/output folder, or an
explicit restart. After an API code update, restart the notebook kernel and
rerun the cells to load the new implementation.

Inspection uses **one `#%%` cell** in `experiments/inspect_current.py`. Edit
the familiar laser/cluster/manual-phase, detuning, topology, noise and plotting
settings at the top, then run the cell. It plots the selected design and settled
phases, dynamics with the far field included, then ensemble-average coupling
matrices before the optional topology-ensemble time series. Set
`ENSEMBLE_SELECTED_TOPOLOGY = None` for only the two ensemble figures.
Results and coupling arrays remain
available as `inspection`, `results`, `figures`, `kappa_per_ns` and `phi_p_rad`.
Rerunning the cell performs a fresh inspection. See
[inspection settings and examples](docs/inspection.md).

Jupyter's own command-line arguments are never parsed. Parallel workers call
functions defined in importable library modules, so they do not depend on
functions defined in notebook cells.

## Direct `vcsel_lib` access

The training run follows `examples/basics/simple_example.py`: ordinary
variables such as `tau_p`, `g0`, and `I`, followed by the native `phys` dictionary
and `vcsel = VCSEL(phys)`. It passes that model into the experiment:

```python
vcsel = VCSEL(phys)
experiment = Experiment("paired_api_01", vcsel=vcsel, config=CONFIG)
```

There is no `PhysicsConfig` or second set of physical parameter names to edit.
`Tmax`, `dt`, `tau`, and the other native entries retain SI units. The pump
current uses the same expression as the simple example. Current training still
uses its original 500 ns duration, zero noise, and saving stride of 10.

The template's `N_lasers`, `delta`, `kappa_c_mat`, and `phi_p_mat` are replaced
for every RL batch: array size comes from `CONFIG.sizes`, detunings come from
sampling, and coupling strengths/phases come from the decoder. Shared physical
parameters are snapshotted when constructing the experiment. Rerun the VCSEL
parameter cell after editing it, then rebuild/preview the experiment.

Evaluation uses the real `VCSEL` class, its `scale_params()`, `cosine_ramp()`,
`generate_history()` and `integrate()` methods. Training's existing worker
adapter uses those same methods. There is no separate RL simulator.

To inspect the actual parameter arrays without integrating, use:

```python
from rl.rl_api import prepare_vcsel

vcsel, history, initial_frequency_ghz = prepare_vcsel(
    designs.coupling_strengths_per_ns,
    designs.coupling_phases_rad,
    designs.config,
    detunings_ghz=designs.context.detunings_ghz,
)
physical_parameters = vcsel.phys  # SI parameters and case-dependent arrays
scaled_parameters = vcsel.nd    # VCSEL's nondimensional parameter arrays
```

The helper only maps named configuration fields and supplies coupling arrays
to the VCSEL API. It retains the original coupling-conversion arithmetic and
`max_iter=1` integration setting when called by evaluation. Strength arrays use
`(cases,M,M)`, phase arrays use the same shape, and all cases share a one-dimensional
ramp. Predicted strengths are passed directly; they are not renormalized by
the library's general topology-construction helper.

## Inspect a saved model

Existing topology-input checkpoints and new API checkpoints use the same
format. Pass an existing checkpoint's full path, or use an API run:

```python
from rl.rl_api import Context, CouplingModel, PlotOptions

model = CouplingModel.load("rl/rl_api/runs/paired_api_01/best.pt", device="cpu")
context = Context(
    target_phases_rad=[0., -1., 2.],
    detunings_ghz=[-1.5, -.5, 1.5],
)
designs = model.propose(context, count=100, continuous_actions="mean", seed=7)
results = designs.evaluate(seed=7)
chosen = results.select("min_budget")
print(chosen.metrics)
results.plot(select=chosen.index, options=PlotOptions(trajectory_mode="mean_sd"))
```

Here `phi2-phi1=-1` and `phi3-phi1=2` radians. Targets and detunings must have
the same length; that length determines M. Prediction sorts the lasers by
detuning and permutes targets and supplied matrices together. Results are in
that sorted order. `continuous_actions="mean"` corresponds to the current
inspector's `SAMPLE_CONTINUOUS_ACTIONS=0`.

`propose()` predicts without simulation. `evaluate()` simulates once, returning
both reward definitions. `select()` and `plot()` reuse those trajectories.
Calling `evaluate()` again explicitly runs another evaluation.

| Selection | Rule |
| --- | --- |
| `best` | Highest cosine reward, preserving current inspection behavior |
| `best_rms` | Highest RMS reward |
| `sparsest` | Fewest enabled links; existing inspector breaks ties |
| `min_budget` | Smallest total directed coupling budget; existing inspector breaks ties |

Selections exclude invalid trajectories. `sparsest` and `min_budget` do not
impose a hidden accuracy threshold. Always inspect the selected metrics.
Plots save under the model's inspection output folder by default, not beside
the loaded checkpoint. Supply `output_directory` to choose another location.

The editable inspection example is `experiments/inspect_current.py`:

```bash
python -m rl.rl_api.experiments.inspect_current
```

## Supply a topology

The API uses **matrix[receiver, source]** for masks, strengths, and phases.
For example, an outward star from laser 1 is:

```python
from rl.rl_api import Topology

star = Topology.from_receiver_source([
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
])
designs = model.propose(context, topology=star, count=1)
```

The old inspector's manual masks use source rows and receiver columns. Convert
those explicitly with `Topology.from_source_receiver(old_matrix)`. A stack
of shape `(count,M,M)` supplies several masks; its length sets the candidate
count. Manual masks must be binary, have a zero diagonal, and be rooted.

## Current recipe

| Setting | Default |
| --- | --- |
| Sizes | M=3,4,5,6,7,8,9 |
| Pairs per size | 5; one continuous detuning span sampled in each of 5 bins |
| Detuning span bins | 0–0.8, 0.8–1.6, 1.6–2.4, 2.4–3.2, 3.2–4 GHz |
| Topologies per pair | 4, covering 4 density bands; each sees the same pair |
| Actions per topology | 10; 8 ordinary and 2 with extra 180-degree phase SD initially |
| Total per iteration | 35 pairs, 140 topology contexts, 1,400 simulations |
| PCGrad | 28 size/density tasks; clip each at 1, project, combine, clip at 1 |
| Simulation workers | 14 processes; 28 jobs of 50 cases with default settings |
| Learning objective | Wrapped phase RMS reward, equal mean/worst-laser weights |
| Monitoring | RMS and cosine metrics; per-M validation reward curves use cosine |
| Detuning/topology curricula | Disabled |
| Rooted connectivity | Lowest-detuned laser's outward star, plus optional links |
| Iterations | 3,000 |
| Ordinary and wide annealing | Printed iterations 1,000–1,500 |
| Final refinement | 1,501–2,000 |
| Learning rate | 1e-4 through 2,000; exponential decay toward 1e-5 by 3,000 |

The phase SD is learned within clamped log-SD bounds. The two wider actions
add variance to that learned SD; 180 degrees is an extra raw-phase standard
deviation, not a maximum allowed phase. Their extra width reaches zero at 1,500.

## Files to read

| File | Purpose |
| --- | --- |
| `experiments/train_current.py` | Short preview/start entrypoint |
| `experiments/inspect_current.py` | Short load/propose/evaluate/plot example |
| `api.py` | Public `Experiment` and `CouplingModel` methods |
| `config.py` | Python configuration sections and mapping into the numerical engine |
| `training.py` | Explicit initialization, iteration, validation, saving loop |
| `data.py` | Context vectors and predicted design arrays |
| `models/` | Policy construction and topology convention |
| `sampling.py` | Inspect the pair/topology sampler |
| `training_batch.py` | Active shared-mask batch, with existing numerical kernels |
| `simulation.py` | Direct VCSEL preparation/integration and scoped evaluation seed |
| `rewards.py` | Both existing reward definitions |
| `evaluation.py` | Simulation results, metrics, candidate selection |
| `plotting/` | Matrices, graphs, trajectories, phase comparison, far field |
| `checkpoints.py` | Existing checkpoint format and API metadata |
| `_compat.py` | Explicit dependencies on existing numerical modules |
| `presets/` | Historical comparison data and original source fingerprints; not runtime settings |
| `diagnostics/` | Explicit reference and real-physics checks |
| `tests/` | Reproduction and workflow tests using throwaway files |

Read [structure](docs/structure.md), [configuration](docs/configuration.md),
[reproducibility](docs/reproducibility.md), and the [Bernoulli plan](docs/bernoulli_plan.md)
for details. The first API version depends on the existing numerical files;
do not delete those as part of a cleanup yet.

## Cleanup status

DE teacher controls and counters have been removed from this package's preset,
histories, and exported checkpoint/run configuration. Unused gate-probability,
dense-reference, and lexicographic-feasibility history placeholders are also
omitted. All active metric names and model/optimizer values are preserved.

This package still calls the shared legacy numerical engine. That untouched
file contains disabled branches for older experiments, and its configuration
object still has legacy attributes. Removing those implementations requires
moving the active numerical path into this package; that separation has not
been completed. This is currently a public interface with a legacy backend
dependency, rather than a fully independent minimal implementation.
