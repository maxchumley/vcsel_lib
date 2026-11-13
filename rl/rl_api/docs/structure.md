# How the code fits together

Start with the short examples, then follow the public methods. There is no
registry, plugin mechanism, inheritance framework, or generated trainer.

## Training flow

```text
Experiment(name, config)
    .describe()              counts only
    .train()                explicit start
        config.to_backend() Python run settings -> numerical engine config
        build_policy()      identical model/parameter initialization
        calibrate budgets   reference behavior
        make_validation_bank()
        for each iteration:
            set_learning_rate()
            existing run_training_batch()
                sample shared target/detuning pairs
                sample rooted topology masks across densities
                predict 10 actions per mask
                dispatch vectorized VCSEL jobs
                calculate RMS advantages and REINFORCE loss
                per-task clipping, PCGrad, global clipping, Adam
            evaluate_bank() when due
            record_history()
            save checkpoint/history/progress figure
        TrainingResult(model, history, directory)
```

The run constructs `vcsel = VCSEL(phys)` using native SI parameter keys, then
passes `Experiment(name, vcsel=vcsel, config=CONFIG)`. The experiment snapshots
the shared scalar physical parameters. The template is not the simulation
state shared across processes: workers create their own VCSEL objects with the
current M, detunings, and predicted coupling arrays. The compatibility mapping
to the existing worker's field names stays inside `_compat.py`.

`training.py` owns the readable loop. The numerical batch routine remains in
`rl/conditional_coupling_bernoulli_sparse.py`. This deliberately preserves the
sampling order, baseline normalization, task weights, clipping, PCGrad, and
optimizer sequence of the current run. `models/policy.py` constructs that same
policy class. Moving those algorithms into new files is a separate future
refactor that should keep the parity tests passing.

Some inactive settings remain because this version still constructs the
shared legacy engine's configuration. Teacher controls are omitted from the
preset and exported metadata; the backend supplies their disabled defaults.
Unused teacher, gate-probability, dense-reference and feasibility placeholders
are omitted from history. User settings are explicitly populated in the run
files. Training does not read the historical JSON recipe. The existing numerical engine itself has not been
pruned, so this is not yet a fully independent minimal implementation.

## Design and evaluation flow

```text
CouplingModel.load(checkpoint)
    .propose(Context, topology=None, count=100)
        sort detunings and permute targets/masks together
        sample rooted masks if none are supplied
        topology-conditioned policy predicts continuous couplings
        DesignBatch                 no simulation yet
            .evaluate()
                original VCSEL simulator
                cosine + temporal RMS scoring
                EvaluationBatch     retains trajectories
                    .select(mode) -> EvaluatedDesign
                    .plot(mode)     reuses saved trajectories
```

`Context` stores copied, read-only vectors. `Topology` stores a copied,
read-only binary matrix or stack. `DesignBatch` stores proposals and the
backend metadata. `EvaluationBatch` stores results for every candidate and
noise realization. `EvaluatedDesign` points to one candidate inside that batch;
it does not copy or simulate it again.

## Array and unit conventions

| Quantity | Shape | Unit / meaning |
| --- | --- | --- |
| Context targets | `(M,)` | radians; relative to laser 1 for scoring |
| Context detunings | `(M,)` | GHz, ordinary frequency |
| Proposed masks | `(count,M,M)` | boolean; `[receiver,source]` |
| Coupling strengths | `(count,M,M)` | ns^-1; `[receiver,source]` |
| Coupling phases | `(count,M,M)` | radians; `[receiver,source]` |
| Budgets | `(count,)` | sum of all directed matrix entries, ns^-1 |
| Simulation times | `(Nt,)` | seconds |
| Simulation states | `(count*R,3*M,Nt)` | interleaved carrier, photon, phase |
| Simulation frequencies | `(count*R,M,Nt)` | backend frequency units, plotted in GHz |
| Reward by realization | `(count,R)` | dimensionless |

R is the requested noise-realization count. Repeated cases use candidate-major
order: all R realizations of candidate 0, then candidate 1, and so on. With
zero noise, repeating realizations does not create independent evidence of
robustness.

The internal model's layer sizes do not depend on M: shared node encoders,
message passing on the supplied graph, pooled context, and a shared edge
decoder operate on variable numbers of nodes and edges. Topology masks enter
the current decoder path. Two legacy scalar SD parameters are retained so
state dictionaries and initialization remain compatible.

## Outputs

`Experiment("paired_api_01")` owns `rl/rl_api/runs/paired_api_01/`:

| Artifact | Meaning |
| --- | --- |
| `run.json` | Public settings, calibrated backend settings without teacher fields, source check |
| `best.pt` | Best validation RMS reward so far |
| `current.pt` | Most recent completed iteration |
| `final.pt` | Final iteration, on successful completion |
| `current_history.json` | Active history keys; written at validation/final steps |
| `current_training_progress.png` | Progress figure when live plotting is enabled |

Each plot update also writes `../current_training_progress.png`, one folder
above the individual run directory. With the default output layout, point a
live image viewer at `rl/rl_api/runs/`. The PNG is rendered once and complete
files are replaced atomically in both locations.

Inspection defaults to `rl/rl_api/runs/inspection/<checkpoint-stem>/figures/`.
Pass `output_directory` to `CouplingModel.load()` or `.plot()` to separate
inspection sessions. Replotting the same selection replaces its PNGs.
`use_tex=False` is the portable default; set `PlotOptions(use_tex=True)` if your
LaTeX environment is configured. Numerical quantities are preserved; figure
layout and fonts are not intended to be identical to the old script.

## Existing dependencies

`_compat.py` names the existing training engine and variable-M numerical core.
Plotting and connectivity reuse pure helpers in the existing inspector and
`inference_plots.py`. The actual simulation reaches `VCSEL` in `vcsel_lib.py`.
`simulation.prepare_vcsel()` prepares its physical/scaled arrays and returns
the actual object and free-running history; `simulate()` calls that object's
`integrate()`. The shared training workers use the original adapter, which
calls those same VCSEL methods. The new adapter is checked against that path
using real trajectories. No rate equation, history solver, ramp formula, or
integrator is reimplemented by the RL API.
The far-field calculation is a local, small NumPy implementation checked
against the existing example's pure functions; it avoids importing that
standalone example and its global font settings.

No existing runner's global output directory is changed by the API. This is a
new interface to the verified numerics, not an independent simulator or a
complete removal of the old numerical modules.
