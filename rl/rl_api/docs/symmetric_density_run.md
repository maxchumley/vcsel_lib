# Density input with learned symmetric connectivity

Run the single editable cell in
`experiments/train_mirrored_symmetric_density_1000.py`, or:

```sh
python -m rl.rl_api.experiments.train_mirrored_symmetric_density_1000
```

The run starts from **fresh seed-7 random weights**. `INITIAL_CHECKPOINT=None`
disables loading another model. Budget calibration is computed from the new
initial policy; the optimizer, action exploration, reward scale and 1k schedules
also start fresh. **No layers, heads or parameters are added** relative to the
successful star architecture. The node widths, three message-passing steps,
edge decoder, budget head and SD outputs are unchanged. Importing the module
only previews; running it starts training in its own `symmetric_density_fresh_...`
directory.

## Requested density and selected graph

The four node features are unchanged. Both existing directed adjacency-feature
slots carry the same retained-pair density `K / (M*(M-1)/2)` on every computational
edge. This is calculated from the requested `s` and M; no specific physical
topology is an input. All-to-all computational messages and the existing
message/decoder architecture are retained. At `s=1`, the conditioning inputs
match the original all-to-all inputs. At lower densities the network responds
to the changed density features. Both coupling values and selected connectivity
are learned from scratch.

The requested number of reciprocal pairs is:

```text
K = M - 1 + round(s * (M*(M-1)/2 - (M-1)))
```

`s=0` requests a tree and `s=1` requests all-to-all. Counts use NumPy's
nearest-integer rounding. There are discrete attainable counts even though
the input is continuous; for M=3 only two and three pairs are possible.

For each sampled prediction, score a reciprocal pair by the sum of its two
predicted strengths. Select a maximum-score spanning tree, then add the
highest-score remaining pairs until exactly K are present. No particular
tree or construction root is fixed. Enable or disable both directions
together; strengths and phases in the two directions remain independent.
Normalize surviving strengths to preserve the predicted total budget.

The original full Gaussian action supplies the REINFORCE score. Removed
strengths remain scored because they can change which pairs survive. Action
perturbations therefore explore connectivity as well as coupling values.
The discrete selector has no pathwise gradient or Bernoulli logits. Its
mathematical connectivity guarantee does not guarantee physical locking.

## Controlled comparison with the successful star run

All other public settings match `train_mirrored_delay_1000.py`: sizes 3–9,
five random target/detuning contexts per size, five stratified span bins across
0–4 GHz, eight mirrored actions per density, the native corrected integrator,
500 ns, 14 workers, ordinary/wide annealing over 1–1000, LR decay over 900–1000,
and the existing wrapped-RMS phase reward and pair-excluded baseline.

Each context has five density requests: zero, an independent uniform fraction
in each third of [0,1], and one. Eight actions share each requested fraction,
but select their own masks. Thus each iteration has:

```text
7 sizes * 5 target/detuning contexts * 5 densities * 8 actions = 1,400 simulations
```

PCGrad retains physical density groups, now measured in excess reciprocal
pairs above a symmetric tree. There are at most 21 tasks; M=3 has no middle
pair-count band. The reference-delay phase convention is unchanged:
`target_source - target_receiver - 2*pi*f_1*tau_effective + residual`, where
laser 1 has the lowest mean-centered detuning after sorting.

This first experiment optimizes phase accuracy. It adds **no sparsity penalty,
budget penalty, phase teacher, or correction rollouts**. The all-to-all density
slot is evaluated normally with its own density-conditioned action samples;
there is no extra paired ATA rollout for every sparse action. The purpose is
to test learned connectivity at a prescribed count before changing the loss.

Validation retains 64 fixed new contexts per size (448 simulations/check),
with fixed density requests balanced across the five slots. Histories and console
separately report tree and ATA success; the training plot keeps two panes. These subsets
use new target/detuning examples; their density requests are fixed. The final
iteration evaluates exact means, selects deterministic masks and skips the
optimizer update.

## Inspection

```python
model = CouplingModel.load("/path/to/new/run/final.pt")
designs = model.propose(context, density_fraction=0.0, count=1)
results = designs.evaluate()
```

Pass a scalar density, one fraction per candidate, or `None` for an ensemble
spread across the density range. With at least two candidates, `None` includes
both tree and ATA endpoints; other fractions are stratified random draws and
their order is shuffled. `seed` / `INFERENCE_RANDOM_SEED` makes this repeatable.
The actual requested fractions are saved in `designs.raw['density_fractions']`.
A supplied topology is
rejected for this model. In `inspect_current.py`, point `RUN_DIRECTORY` at the
new run and set `DENSITY_FRACTION=None` for this ensemble (the inspector default).
Older topology-input checkpoints ignore that setting. Mean inference repeats
the same design for requests rounding to the same pair count; an ensemble
across densities is not guaranteed to contain `count` distinct topologies.
For different designs at the same count, sample continuous actions while the
checkpoint still has nonzero exploration.

Preview from a terminal:

```sh
python -m rl.rl_api.experiments.train_batch train_mirrored_symmetric_density_1000 --preview
```

No existing checkpoint or results directory is required. Use the default
`INITIAL_CHECKPOINT=None` when launching the recipe or train_batch.
