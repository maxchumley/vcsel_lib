# Reproduction checks and limits

The reference is the current paired run in
`rl/conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse.py`, suffix
`_star_rms_pairs5_topo4_wide180_3k`, captured on 2026-11-05.

## What is preserved

The default API backend configuration matches all 183 fields of that runner
except the three checkpoint paths. The explicit Python run-file settings are
independent of future edits to the original runner. The historical JSON is
kept only for comparison diagnostics and is not used at runtime. Original source
fingerprints are recorded in `presets/reference_manifest.json`. The original
hashes remain intact; `reviewed_source_revisions` additionally records the
verified correction to the trapezoidal delayed histories. Training accepts
only an exact original or explicitly reviewed hash for each of the three
numerical files. If they change again, recheck parity;
`verify_reference=False` explicitly skips that guard when intentionally testing
a revised backend.

The existing policy, sampler, REINFORCE, baseline normalization, clipping,
PCGrad, Adam, evaluation, and checkpoint format are preserved. The new loop
keeps the same initialization and RNG consumption sequence. It does not load
the live run or write to its output paths.

Unused teacher controls and counters are removed from the new preset,
exported configuration, and histories. Gate-probability, dense-reference, and
lexicographic-feasibility history placeholders are also removed. The legacy
loader fills omitted teacher configuration with disabled defaults, so model
files remain loadable. Reproduction tests compare all active history fields
and the exact model/optimizer state. This is a metadata cleanup; the shared
numerical engine still contains its old disabled branches.

## Verification commands

From the repository root in the `vcsel` environment:

```bash
python -m rl.rl_api.diagnostics.verify_reference
python -m rl.rl_api.diagnostics.verify_reference --physics
python -m rl.rl_api.diagnostics.verify_reference --parallel
```

The first command reads settings and fingerprints only. `--physics` additionally
compares two real 500 ns candidate trajectories and both reward arrays against
a direct call to the existing simulator. It does not train or save a model.
`--parallel` additionally compares serial scoring with two spawned simulation
workers. It uses the same physical cases and never starts a training loop.

The automated tests write only throwaway artifacts within this new package:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR="$PWD/rl/rl_api/tests/_cache/matplotlib" \
NUMBA_CACHE_DIR="$PWD/rl/rl_api/tests/_cache/numba" \
python -m pytest rl/rl_api/tests -q -p no:cacheprovider \
  --basetemp=rl/rl_api/tests/_tmp
```

Tests check:

- Complete default configuration, model keys, and exact initial weights.
- Pair/mask samples for every M=3…9, including the next RNG draw.
- Three numerical training updates against the original loop: identical
  weights, Adam state, histories, checkpoint iterations/rewards, NumPy state,
  and Torch state. These tests substitute a small deterministic physical
  trajectory; sampling, scoring, backpropagation, PCGrad, and Adam are real.
  Comparisons cover both one M and a mixed M=3,5,9 run.
- Seeded inference with ordinary means and sampled actions, manual matrix
  conversion, unsorted detunings, and legacy checkpoint loading.
- Both evaluation metrics, selections, and plotting without resimulation.
- Far-field arrays against the original example's pure calculation.
- Existing-run overwrite refusal and explicit unsupported Bernoulli mode.
- Seven-size pool dispatch, out-of-order collection, and pool cleanup, using
  simulated worker calls so this unit test does not launch 14 processes.
- The explicit training run config matches the original even when reads of
  the JSON snapshot are blocked. Edited run values reach sampling/exploration;
  inspection values reach prediction/evaluation/plotting. Preview never trains.
- Both run files execute cell by cell without `__file__`, ignore kernel argv,
  and retain intermediate objects. Replotting inspection results does not
  resimulate. The explicit start switch controls the training cell.
- The simulation adapter delegates scaling, ramps, free-running histories,
  and integration to VCSEL. Its real trajectories match the original adapter.
- The training file constructs `VCSEL(phys)` with native parameter names.
  Its shared physical values survive snapshotting and per-M worker parameter
  construction; context-specific arrays are replaced per batch. Unknown or
  unsupported native modes fail explicitly. There is no PhysicsConfig wrapper.

The local `pytest.ini` uses importlib mode to avoid confusing the repository's
package directory with its same-named `vcsel_lib.py` module during collection.

The short optimizer comparisons isolate training orchestration; they are not
a claim that a complete 3,000-iteration physical training run was repeated.
The real-physics check tests the simulator boundary separately. Full-run and
14-worker end-to-end training have not been launched as part of this setup.

## Checkpoints and restart

The existing `.pt` payload includes model weights, optimizer state, config,
iteration, and held-out reward. It does not contain every sampler RNG state or
the running advantage-normalization state. Therefore `.load()` is for inference,
and `.train()` begins a fresh experiment. This release does not advertise exact
mid-run resume.

The existing loader reconstructs policy SD bounds using its defaults rather
than restoring the exact annealed in-memory bounds. This behavior is retained
for compatibility. Deterministic mean actions are unaffected. Sampled actions
from a late checkpoint can have different widths than the live model at save
time. Use `continuous_actions="mean"` to match current deterministic inspection.
Fixing sampled-load semantics is a separate deliberate behavior change.

For strongest numerical comparison use CPU, the same package versions, thread
settings, seed, worker layout, and zero-noise settings. Exact floating-point
identity is not promised across operating systems, accelerators, or library
versions. Nonzero noise is evaluated using the existing process-local NumPy
RNG behavior. `evaluate(seed=...)` restores the caller's global NumPy state
after its synchronous call; do not use that scoped seeding concurrently from
multiple threads.

## Deliberate interface differences

- Artifacts live in the API run folder and include an extra `run.json`.
- Console labels explicitly distinguish RMS and cosine.
- Plotting defaults to no LaTeX and separates selected, ensemble, and far-field
  figures. Font/layout identity is not part of numerical parity.
- Explicit evaluation seeds are scoped; the old inspector's global seeding
  side effect is not reproduced.
- One matrix convention is used throughout the public API, with an explicit
  converter for old manual masks.

These differences do not change the current training algorithm.
