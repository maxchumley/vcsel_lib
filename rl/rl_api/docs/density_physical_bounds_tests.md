# Additional physical-bound symmetry tests

These tests are additional jobs. They do not replace, cancel or change the
running latent-blend refinement or any fixed-topology experiments.

All three load the same working laser-1 density checkpoint at iteration 950
(SHA-256 `83ad841f91a2022cd869b8d9cb21a82a57ce11a60f99f85e3ef5b7766008e43b`).
Their initial weights, inputs, random seed, network, REINFORCE objective,
mirrored sampling, validation bank and adaptation schedule are matched.

| Component constrained | Job | Progress image |
| --- | --- | --- |
| Strengths and phases | `density_bound_both` | `density_bound_both_progress.png` |
| Strengths only; phases free | `density_bound_kappa` | `density_bound_kappa_progress.png` |
| Phases only; strengths free | `density_bound_phase` | `density_bound_phase_progress.png` |

Both values remain trainable in every job. The isolated-component jobs identify
which constraint causes the loss of accuracy; their endpoints are not fully
symmetric experimental designs.

## The actual constraints

Let `s` be the tightening fraction and `b=1-s`. Selected strength pairs obey:

```text
abs(kappa_ij - kappa_ji)/(kappa_ij + kappa_ji) <= b
```

Pairs already satisfying the bound are untouched. For a violating pair, clip
its difference to `[-b*pair_total,+b*pair_total]` while preserving pair total.
The operation occurs after physical decoding, so unbounded latent outputs
cannot evade the relative limit. Total coupling budget remains learned; the
bound is relative, not an absolute coupling-budget limit.

Selected phases obey `abs(wrap(phi_ij-phi_ji)) <= b*pi`. Violating pairs move
only far enough along their shortest circular arc to meet the bound. Pairs
straddling ±pi are handled circularly. Inactive phase entries are also capped
for a consistent returned matrix; they have no effect on dynamics.

At zero tightening the source predictions are bit-identical. At full tightening
the selected components are exactly symmetric. Strength and phase flags are
checkpointed, so inspection uses the same constraints. Pair totals and zero
strength masks are preserved. Topology selection explicitly ranks original
unconstrained pair totals, so tightening alone cannot change its graph through
roundoff or altered pair rankings. Model learning and strength sampling can
still select different graphs.

## Schedule and quality guard

As in the existing refinement: at most 1000 updates, increments 0.05, validation
every 50, fixed-baseline reward tolerance 0.01 and success tolerance 5 percentage
points. A failed validation holds the current bound while training continues;
it does not increase tightening at the end. Narrow exploration scale 0.03→0.003
and LR 1e-5→1e-6 taper with tightening; wide phase samples remain disabled.
There are 1,400 training simulations/update (sizes 3–9, five contexts/size, five
densities/context, eight actions/density), 448 validation cases/check, original
500 ns standard-LK simulations and two-pane plots every 10 updates.

This is still the same latent Gaussian REINFORCE policy; only its deterministic
physical output constraint differs. Scores are not changed to an incorrectly
assumed Gaussian density of clipped outputs.

`initial.pt`, `current.pt`, `final.pt` and `best_accepted.pt` retain their usual
meanings. The both-values run additionally writes `fully_symmetric.pt`. The
isolated runs instead write `constraint_endpoint.pt`. Every `result.json`
reports component, achieved tightening, measured endpoint accuracy and whether
both values actually reached symmetry. A constraint endpoint is not a claim
of target accuracy.

## Great Lakes

```sh
git pull --ff-only origin updates
bash scripts/submit_density_physical_bounds_tests.sh
```

The working checkpoint is already in ignored `runs/initialization/` on Great
Lakes. Each independent job requests 32 CPUs (31 workers), 64 GB and seven days. They use
unique run directories and image names discovered by the existing plot sync.
Add `--dry-run` for commands only or `--smoke-test` for two updates on sizes 3,4
with coarse 0.5 increments. Running full jobs requires the command above without
`--smoke-test`.

Existing checkpoints without a method field default to their original
`latent_blend` semantics. The old launcher and its defaults remain intact.
