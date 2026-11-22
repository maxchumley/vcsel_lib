# Refining the working density model toward symmetric coupling

This replaces the two fresh asymmetric-value runs. It is **one warm-started,
laser-1-reference continuation**, with the same GNN, native LK simulator and
mirrored REINFORCE training. No new network weights, labels or teacher solver
are introduced.

The source is the original density experiment's **best checkpoint at iteration
950**, whose saved validation reward is 0.9910601943. SHA-256:
`83ad841f91a2022cd869b8d9cb21a82a57ce11a60f99f85e3ef5b7766008e43b`.
The checkpoint is copied to ignored `runs/initialization/` on both machines;
it is not stored in GitHub.

## What changes during refinement

A symmetry fraction starts at zero. Before simulation, reciprocal latent
strengths move toward their pair average. Reciprocal phases move toward each
other along the shortest circular arc, including phases across the ±pi cut.
At zero, deterministic predictions exactly match the loaded asymmetric model.
At one, both decoded physical matrices are exactly symmetric. Connectivity
remains connected and symmetric throughout. Total budget remains learned and
is normalized by the existing decoder.

Sampled latent actions keep their original Gaussian REINFORCE scores. This is
a deterministic action transformation, not a new loss or a claimed Gaussian
marginal for the circular phase projection. The same transformation is saved
in checkpoints and applied during deterministic/stochastic inspection.

## Training and accuracy guard

- First measure the unchanged model on the fixed 448-case validation bank.
- Train at symmetry 0.05, with five contexts per size, five density requests and
  eight mirrored actions: 1,400 simulations per update across sizes 3–9.
- Every 50 updates, compare deterministic validation against the **original**
  baseline. Tighten by another 0.05 only when reward has fallen by at most 0.01
  and 3° success by at most five percentage points. Otherwise hold the current
  fraction and keep training there. The tolerances do not accumulate.
- Limit refinement to 1,000 updates. Without holds, the final 50 updates train
  at full symmetry. A hold can prevent reaching full symmetry; that is reported
  explicitly rather than hidden by a forced final increase.
- Start fresh Adam at narrow LR 1e-5 and exploration scale 0.03, with no wide
  phase actions. Both taper tenfold as symmetry reaches one, not as time passes,
  so a stalled level retains its adaptation settings. All other source physics,
  distributions, RMS reward, pair baseline and PCGrad settings are retained.

The guard uses the fixed validation bank for curriculum decisions. Its results
are validation measurements, not an independent final generalization test.

## Outputs and launch

```sh
git pull --ff-only origin updates
bash scripts/submit_density_symmetry_refinement.sh
```

`submit_asymmetric_value_density_tests.sh` forwards to the same replacement,
so the previous command no longer launches the two fresh experiments. No
already-submitted job is cancelled automatically.

The new job is `density_sym_refine`: 32 CPUs, 31 workers, 64 GB, seven days.
The existing plot sync discovers `density_symmetry_refinement_progress.png`.
Plots retain two panes and update every ten iterations; the right pane shows
validation success and the enforced symmetry fraction.

`initial.pt` preserves the source weights with zero projection. `current.pt`
and `final.pt` record the actual trained fraction. `best_accepted.pt` selects
the highest fraction that retained original accuracy, then highest reward.
`fully_symmetric.pt` always measures the exact endpoint with the final weights,
even if the curriculum stalled; `result.json` explicitly reports whether that
forced endpoint retained accuracy. A fully symmetric checkpoint is not a
claim of accurate phase prediction.

Use `--dry-run` to preview or `--smoke-test` for a two-update, two-size launch
check (its large 0.5 increments are not the real curriculum). The Python runner
also supports `--preview` and overrides for update count, checkpoint, increment,
validation interval and retention tolerances.
