# Random density bands and precision refinement

The next comparison uses [five fixed counts](fixed_counts_run.md) with this
same precision recipe. This document describes the preserved random-count run.

Start in `experiments/train_precision.py`. All physical and RL settings are
explicit in the run file. There is no factory wrapper or runtime JSON preset.
It has one `#%%` cell and `START_TRAINING=True`: running that cell starts the
experiment. Set False to preview only. Setup has not started training.

Run name: **star_rms_bins5_randbands5_precision_standard_lk_3k**

Output: `rl/rl_api/runs/star_rms_bins5_randbands5_precision_standard_lk_3k/`

The previous run files and output folders are preserved.

The native VCSEL dictionary selects `"phase_model": "standard_lk"`. This is
an additional physics change from the previous runs' `chumley_2026` default.
The API passes the choice to VCSEL for training and validation and saves it
in checkpoints, so inspection uses the same model. No phase equations are
implemented in the API. All other physical parameters retain their values.

## Per-iteration topology coverage

For each M=3…9, generate five fresh detuning/target pairs with one span per
0.8 GHz bin over 0–4 GHz. Each pair independently gets:

1. The exact outward star from the lowest-detuned laser.
2. One uniformly sampled link count in the sparse band, excluding the star.
3. One uniformly sampled link count in the middle band.
4. One uniformly sampled link count in the dense band, excluding all-to-all.
5. The all-to-all graph.

Optional edges are independently drawn for each intermediate mask, always
retaining the star. Counts and wiring are redrawn each iteration. Counts can
repeat across pairs; uniqueness is not forced. M=3 has only one interior count
per band, but its optional wiring still varies. Every permitted integer count
has an opportunity to appear over time.

The physical band coordinate is `(links-(M-1))/(M-1)**2`: sparse [0,1/3),
middle [1/3,2/3), dense [2/3,1]. For six lasers, the five counts are
**5, random 6–13, random 14–21, random 22–29, 30**.

| Setting | Value |
| --- | --- |
| Array sizes | 3 through 9 |
| Pairs per M | 5 |
| Topologies per pair | 5 |
| Actions per topology | 8, with 2 initially receiving extra phase exploration |
| Contexts per iteration | 7 x 5 x 5 = 175 |
| Training simulations per iteration | 7 x 5 x 5 x 8 = 1,400 |
| PCGrad tasks | 7 x 3 = 21, all populated |
| Worker dispatch | 35 vectorized jobs of 40 cases across 14 workers |
| Training iterations | 3,000 |
| Curricula | Both disabled |
| Learning reward | Wrapped RMS; cosine monitored |
| Model/physics | Existing 127,304-parameter GNN and 500 ns VCSEL rollout |
| Phase convention | Native VCSEL `standard_lk` |

Each task includes all five pairs. Sparse and dense each contain 80 cases per
M; middle contains 40. Existing sample-fraction weights preserve the equal-M
objective: 40%, 20%, 40% within each M. Task/global clipping remain 1.0.

The supplied fraction fluctuates with the random counts. Its expected value
is close to the previous mean; discrete band boundaries make it slightly
different from exactly 0.5949263. It is a sampling statistic, not learned pruning.

## Stages within 3,000 iterations

All boundaries are printed iteration numbers.

| Iterations | Ordinary phase SD limits | Extra phase SD | Nominal log-budget SD | LR |
| --- | --- | --- | --- | --- |
| 1–999 | Existing initial bounds, about 12.8–121.3 degrees | 180 degrees on 2 of 8 actions | .35 | 1e-4 |
| 1000–1500 | Anneal to about 2.85–12.8 degrees | Cosine taper to zero | .35 | 1e-4 |
| 1501–2000 | Cosine-refine to about .64–2.85 degrees | Zero | Cosine taper .35 to .05 | Exponential decay to 1e-5 |
| 2001–3000 | About .64–2.85 degrees | Zero | .05 | Hold 1e-5 |

The policy predicts actual SD within these clipping limits. The shared ordinary
bounds narrow raw-strength exploration too; no new model weights are added.
The precision stage starts only after the original taper finishes.

The existing learned budget multiplier `exp(.25*tanh(budget_log_std))` is
retained. Nominal .05 means actual log-budget SD is approximately .039–.064.
Sampling and log-probability calculations use the same scheduled SD.

The LR reaches its final value by iteration 2000, leaving 1,000 iterations at
low LR. No iterations are added. Short test overrides stop early without
rescaling these schedule boundaries.

## Implementation and checkpoints

`topology_sampling_mode="band_random"` selects `sampling.py` and the API batch
in `training_batch.py`. Physical-band PCGrad uses the original gradient
projection/clipping kernels. `exploration.py` returns an iteration-local config
for the refined bounds and budget SD; the base experiment config is not mutated.
Numerical source files outside `rl_api` are untouched. Older recipes keep their
defaults and are checked by the existing parity tests.

`run.json` stores the full base/public configuration. Checkpoints and history
store the sampling recipe, full precision schedule, current nominal budget SD
and actual ordinary SD bounds. New API checkpoints restore those actual bounds
so stochastic inspection matches the saved iteration. Legacy checkpoints keep
their existing loading behavior. Validation uses the same fixed held-out bank
and deterministic mean actions, with best checkpoints selected by RMS.

`RESTART_EXISTING_RUN=True` archives an existing attempt only when you explicitly
start training. This is a fresh run, not continuation from the current model.
After starting, set `RUN_DIRECTORY` in the one-cell inspector to the new folder.

Plots retain the LaTeX font, 300 DPI and the parent-folder live image at
`rl/rl_api/runs/current_training_progress.png`.

## Verification

Tests cover rooted connectivity and all density bands for every M, independent
draws, changing counts/wiring, full six-laser intermediate-count coverage, both
exploration stages, LR endpoints and the fixed 3,000-iteration limit. A complete
synthetic-physics batch verifies 1,400 cases in 35 jobs and all 21 tasks. A parity
test injects the original sampler to check unchanged weights, Adam, normalization
and RNG behavior. Precision-stage updates and checkpoint round trips check model
weights and stochastic bounds. These checks do not start the user's full run or
establish final model accuracy.
