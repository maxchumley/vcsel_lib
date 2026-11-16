# Five fixed topologies per laser count

Run the single editable cell in
`experiments/train_mirrored_laser1_fixed_topologies_1000.py`, or from the repository root:

```sh
python -m rl.rl_api.experiments.train_mirrored_laser1_fixed_topologies_1000
```

Importing only previews the recipe. Running the cell/module starts training in a
separate directory named:

```text
rl/rl_api/runs/laser1_fixed5_topologies_from_star_rms_bins5_phase_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/
```

The experiment tests whether the model needs more practice on each graph before
learning across a much larger topology space. It uses **five distinct allowed
graphs per size**, for **35 total graphs across sizes 3–9**, throughout training.
They are shared across target/detuning pairs and iterations. Targets and detuning
distributions continue to be redrawn on every iteration.

The five graphs are chosen once by the existing random-reachability sampler
using independent bank seed **60007 + M**. Duplicate-bank draws and banks missing
an actual sparse/middle/dense band are rejected. The selection does not inspect
policy outputs, rewards or whether any graph is easy. Each bank includes the
all-to-all endpoint. The exact graphs, root indices and counts are written to
`fixed_topology_bank.json` and saved in checkpoint sampling metadata.

Laser 1, meaning the lowest-detuned laser after sorting, can reach every laser
in each graph. Optional cycles and feedback remain eligible. There is **no
strength prior, tree-biased decoder, new model input or feedback teacher**.

For the controlled comparison, the run uses the **same successful delay/star
`final.pt` initialization as `train_mirrored_laser1_reachable_1000.py`**, not the
later topology-trained or diagnostic weights. Learned weights and budget
calibration are copied; optimizer, history and schedules start fresh. All other
training settings match that varying-topology recipe: 1,000 iterations, seed 7,
five target/detuning pairs per size, five graph slots, eight mirrored actions,
0–4 GHz spans, the same reference-frequency phase formula, wrapped RMS reward,
PCGrad, 14 workers, 500 ns native integration and the original exploration/LR
annealing schedules. The final iteration uses exact means without an update.

**Training remains 1,400 simulations per iteration.** Each fixed graph sees five
new target/detuning contexts per iteration instead of being replaced by a new
graph. The bank sampler advances the original mask RNG calls before returning
the fixed masks; no extra physical simulation is associated with that bookkeeping.
Context distributions and seed remain unchanged. Different occupied task counts
and learned weights mean the full trajectory is not promised to be bit-identical
to the varying-topology run.

## Two validation groups

At iteration 1, every 50 iterations and the final iteration, evaluate the same
32 held-out target/detuning contexts per size in both groups:

1. **Trained graphs, new targets:** cycle through that size's five bank graphs
   as evenly as possible, one graph per context.
2. **Unseen graphs, new targets:** draw one graph per context with the existing
   validation seed, rejecting any graph present in that size's training bank.

Unseen graphs may repeat across held-out contexts, but none is a training-bank
graph. In particular, all-to-all is excluded from the unseen group because it
is in the training bank. This makes the groups a useful diagnostic rather than
an identical topology-distribution comparison with the older validation score.

Each group contains 224 simulations, so validation has **448 simulations per
check**, matching the previous total validation budget. Each group's estimate
uses fewer cases than the previous single-group validation. Training cost is unchanged. No corrections
are applied to validation outputs.

Existing `held_out_*` history fields and `best.pt` selection refer to the
**unseen-topology group**. New `seen_topology_held_out_*` fields record trained-graph
performance. Console progress adds `seen-RMS` and `seen-3deg`. The training plot
adds a third panel comparing the 3-degree success fraction on the two groups;
the existing reward panel also includes the trained-graph RMS curve. Font,
resolution and live PNG saving use the existing plotting settings.

Strong trained-graph performance with weak unseen-graph performance supports a
topology-generalization bottleneck. Poor performance on both means fewer graphs
alone has not resolved optimization across the still-random target/detuning
space. Neither result by itself proves a universal architecture limitation.

## Terminal or cluster launch

```sh
python -m rl.rl_api.experiments.train_batch train_mirrored_laser1_fixed_topologies_1000 --preview
```

To supply the source checkpoint from another location:

```sh
python -m rl.rl_api.experiments.train_batch train_mirrored_laser1_fixed_topologies_1000 \
  --initial-checkpoint /path/to/star/final.pt
```

Standard checkpoint inference can still propose arbitrary rooted graphs. To
inspect a particular trained graph, use a matrix from the saved bank as a manual
topology; the saved laser labels are in sorted detuning order. Automatic inference
does not promise exclusion of training-bank graphs; the separate validation
procedure enforces that exclusion explicitly.
