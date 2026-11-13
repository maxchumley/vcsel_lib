# Baseline: working 5-bin setup with new annealing

The next experiment is the [phase-residual comparison](phase_residual_run.md).
This document describes the preserved four-count baseline.

Run the single `#%%` cell in `experiments/train_four_counts.py` when ready.
It has `START_TRAINING=True`. Importing it does not train, and setup has not
started the run.

Run name: **star_rms_bins5_fixedcounts4_precision_standard_lk_3k**

Output: `rl/rl_api/runs/star_rms_bins5_fixedcounts4_precision_standard_lk_3k/`

This restores the exact `per_target` topology sampler used by the successful
`star_rms_backup_sampling_bins5_topo4_actions8_3k` experiment: each of five
detuning/target pairs receives four evenly spaced link counts, with independent
optional wiring and randomized topology ordering. The outward star from laser
1 is always retained. Existing run files and results are preserved.

This is the **working 5-bin run** baseline, rather than the original BACKUP's
two-bin/nine-topology recipe. All working 5-bin settings are retained except
the native phase convention and new precision/LR schedule. RMS was already
the working 5-bin run's learning reward and remains enabled here.

## Topology comparison

| Setting | Five-count run | Next four-count run |
| --- | --- | --- |
| Pairs per size | 5 | 5 |
| Topologies per pair | 5 | 4 |
| Actions per topology | 8 | 8 |
| Topology instances per size | 25 | 20 |
| Simulations per iteration | 1,400 | 1,120 |
| Vectorized jobs across 14 workers | 35 | 28 |
| Populated PCGrad tasks | 21 | 16 |
| Six-laser link counts | 5, 11, 18, 24, 30 | 5, 13, 22, 30 |

Four counts restore the old physical-band occupancy and weights. For M=3,
5, 6, 8 and 9, sparse/dense each receive 50% of their size's weight, with no
middle task. For M=4 and 7, sparse/middle/dense receive 25%/25%/50%.
All seven sizes receive equal total weight. No new PCGrad formula or band
boundaries are introduced; empty bands are skipped using the existing rule.

## Settings retained from the current comparison

Native `standard_lk` physics, 500 ns rollouts, the RMS learning reward, cosine
monitoring, seed 7, five 0.8 GHz detuning-span bins, model architecture,
per-task/final clipping, held-out bank and plot settings are retained.

Training still lasts 3,000 iterations. Ordinary and wide phase exploration
anneal over 1000–1500; precision refinement and LR decay run over 1500–2000;
then LR stays at 1e-5 through 3000. Two of eight actions initially receive
extra 180-degree phase SD. Both curricula are disabled. The added precision
stage is retained so this comparison changes the topology allocation only.

The API now supports that existing precision stage with `per_target`
sampling. It passes the same schedule to the same batch implementation and
saves actual exploration bounds in checkpoints. `run.json`, history and
checkpoints identify the four-mask sampler and full schedule.

The working 5-bin run used `chumley_2026`, LR decay over 2000–3000, and no
extra precision stage. The new run selects `standard_lk`, precision refinement
over 1500–2000, and LR decay over 1500–2000 followed by a hold through 3000.
The ordinary/wide taper over 1000–1500 remains the same. Training performance
will be established by the run; setup checks do not predict its final reward.

Tests check exact sampler/context/RNG parity with the working sampler,
configuration differences, 1,120 simulations in 28 jobs, all 16 task weights,
precision-stage settings, checkpoint reload and one-cell startup. These are
implementation checks, not a completed training comparison.
