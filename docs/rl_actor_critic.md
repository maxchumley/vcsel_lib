# Density actor–critic comparison

Launch both additional jobs on Great Lakes from the repository root:

```bash
bash scripts/submit_density_actor_critic_tests.sh
```

The jobs are `density_ac_current` and `density_ac_replay`. Each requests 32 CPUs,
64 GB and seven days on `arrays0`. Existing jobs continue running. Add `--dry-run`
to preview submission commands or `--smoke-test` for the two-iteration launch check.

Both start from identical fresh seed-7 actor weights. They match the existing
conditioned-phase density recipe: 3–9 lasers, random target phases and detuning
distributions spanning up to 4 GHz, five density requests and eight mirrored
actions for five contexts per size. Each iteration uses 1,400 real 500 ns
simulations. Connectivity, coupling strengths and coupling phases are symmetric.
Laser 1 in detuning order supplies the phase reference. Validation remains 448
real held-out simulations every 50 iterations, plus the first and final iteration.

The actor architecture, true-reward REINFORCE/PCGrad update, learning-rate and
exploration schedules are unchanged. A separate three-message-pass GNN predicts
the measured wrapped-RMS reward from context, requested density and the actual
executed mask, strengths and phases. It has independent weights and optimizer.
It learns directly from successes and failures; there is no future-return target,
labelled optimum or gradient through the simulator.

Both critics take 20 supervised minibatch updates of 256 examples per iteration,
with Adam at 0.0003. `density_ac_current` fits only the latest 1,400 designs.
`density_ac_replay` fits a rolling capacity of 100,000 designs, divided evenly by
array size, and keeps old critic examples. Sampling balances array size and
physical density band in both arms. One complete context per size (all its
density/action siblings) is withheld from critic fitting. With five contexts,
80% of records are available for fitting; all records count toward capacity.

After ten warmup iterations, up to four additional actor updates use fresh
current-policy action samples and detached critic rewards. They are enabled only
when the current withheld designs have reward MAE ≤ 0.10 and pairwise ranking
accuracy ≥ 55%, with at least 16 informative comparisons. Rankings compare
actions within the same context and density request, ignoring true reward gaps
≤ 0.001. This tests useful local action ranking rather than just recognising
easy contexts. Gate decisions and actual update counts are logged; the attempted
schedule and eligibility rule are identical between arms. Historical action
log-probabilities are never reused for actor updates. Virtual updates use
separate RNG streams and make no extra simulator calls. The exact-mean final
iteration skips all actor updates, including Adam momentum.

The two-panel progress figures show **real** training and validation rewards,
never critic predictions. Unique names `density_ac_current_progress.png` and
`density_ac_replay_progress.png` are discovered by the existing image sync.
Console output and `current_history.json` additionally record critic loss,
withheld MAE, ranking accuracy, buffer size and actual extra actor updates.

Each run saves compact measured examples to `experiences/batch_*.npz`, the
critic to `critic_current.pt`, and its latest buffer snapshot at validation
intervals. Archives are for diagnostics; the current-only run never reads older
files. Actor `current.pt`, `best.pt` and `final.pt` retain the inference API and
do not require the critic. This experiment does not implement exact resume.

Compare the two new arms to isolate replay, and compare both to the existing
fresh laser-1-reference REINFORCE run to assess critic assistance. Success still
means improved real held-out phase accuracy, especially the fraction within 3°;
a rising predicted reward alone is insufficient.
