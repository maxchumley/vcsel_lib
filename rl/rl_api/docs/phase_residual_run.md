# Next run: target difference plus learned phase residual

Run the single `#%%` cell in `experiments/train_phase_residual.py` when ready.
It has `START_TRAINING=True`. Setup and tests have not started training.
Before starting this next experiment, restart the Python kernel so it imports
the new model setting and policy class rather than cached baseline modules.

Run name: **star_rms_bins5_fixedcounts4_phase_residual_precision_standard_lk_3k**

Output: `rl/rl_api/runs/star_rms_bins5_fixedcounts4_phase_residual_precision_standard_lk_3k/`

The baseline is the currently running `train_four_counts.py` experiment,
`star_rms_bins5_fixedcounts4_precision_standard_lk_3k`. This is a fresh run
from the same seed and parameter initialization, with a new output directory.

## The one mathematical change

The existing phase output becomes a residual mean. For a directed edge from
source j to receiver i:

```python
phase_mean = predicted_residual + wrap(target_phase[j] - target_phase[i])
```

The combined mean is used by the original Gaussian sampler and its log
probability. The original decoder wraps sampled phases to physical angles.
Deterministic validation and inference use the same combined mean.

The known offset is read from the cos/sin target features already supplied
to the GNN, so it uses the same relative-phase and global-conjugacy convention.
For an outward link from laser 1 to a laser targeted at +90 degrees, the known
coupling-phase contribution is -90 degrees. This sign follows the VCSEL
coupling argument `phase_source - phase_receiver - coupling_phase`.

`models/policy.py` contains the small `TargetResidualPhasePolicy` subclass.
It calls the existing policy and changes only its returned phase mean. There
are no added weights, layers or stochastic draws. The existing GNN inputs are
retained, so the learned residual can still depend on the target features.
This test does not train a separate zero-target model.

## Run settings

| Setting | Value |
| --- | --- |
| Phase parameterization | `ModelConfig(phase_parameterization="target_residual")` |
| Sizes | 3 through 9 |
| Detuning/target pairs and span bins per size | 5 |
| Span range | 0–4 GHz, five 0.8 GHz bins |
| Topologies per pair | Four evenly spaced counts, independent random wiring/order |
| Rooted connectivity | Outward star from lowest-detuned laser 1 enabled |
| Actions per topology | 8; initially six ordinary and two extra-wide |
| Training simulations per iteration | 1,120 |
| PCGrad | 16 populated size/density tasks with the baseline weights |
| Per-task / final gradient norm limits | 1.0 / 1.0 |
| Model parameters | 127,304, same keys and initialization |
| Phase model | Native `standard_lk` |
| Learning reward | Wrapped RMS; cosine monitored |
| Topology / detuning curricula | Disabled / disabled |
| Workers | 14, receiving 28 vectorized jobs |
| Duration / noise | 500 ns / zero |
| Training iterations | 3,000 |

The exact baseline schedules are retained: ordinary and wide exploration
anneal over 1000–1500; precision refinement and LR decay run over 1500–2000;
then the final widths and LR 1e-5 hold through 3000. Final ordinary log-SD
bounds are -4.5 to -3.0, and nominal budget log-SD is .05. Initial extra wide
phase SD is 180 degrees. Validation remains every 50 iterations on the same
64 contexts per size, with best checkpoints selected by RMS.

Strength and budget prediction, their sampling and redistribution, phase SDs,
the loss, PCGrad, validation, plots and sampling counts all retain baseline
behavior. There is no feedback correction, extra auxiliary loss, added
strength-to-phase conditioning, or additional physics evaluation in training.

Adding the target offset changes the initial phase predictions; that is an
intended part of this parameterization experiment. Existing weights are not
resumed or reinterpreted as residuals.

## Checkpoints and older experiments

The model setting is explicit in `run.json` and saved as
`phase_parameterization="target_residual"` in checkpoint configuration.
`CouplingModel.load()` reconstructs the residual policy and actual saved SD
bounds, so inspection applies the offset exactly once. Old checkpoints without
the setting retain absolute phase prediction. `absolute` remains the default
for every existing run file, and its backend configuration remains compatible
with the old numerical schema.

## Verification

Tests check source/receiver sign, angle wrapping, both target-orientation modes,
unchanged strengths/budget/SDs, identical weights and random draws, corrected
Gaussian log probabilities and gradients, all 1,120 cases and 16 PCGrad tasks,
checkpoint reload, validation, inference and one-cell startup. These establish
implementation behavior, not the learning outcome of the new run.
