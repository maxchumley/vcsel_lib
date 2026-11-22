# Symmetric density topologies with asymmetric coupling values

These two fresh recipes copy the last reciprocal-pair density runs, disabling
only `ModelConfig.symmetric_coupling_values` and `reciprocal_pair_actions`.
Each selected reciprocal pair still enables both directions. Directed strength
and phase values are predicted and sampled independently; neither matrix is
projected or averaged. The network layers and parameter count are unchanged.

| Recipe | Reference | Job | Progress image |
| --- | --- | --- | --- |
| `train_mirrored_density_asymmetric_values_1000` | Laser 1 / lowest detuning | `density_asym_ref` | `density_asym_ref_progress.png` |
| `train_mirrored_density_asymmetric_values_zeroref_1000` | Zero | `density_asym_zero` | `density_asym_zero_progress.png` |

Both start from seed-7 random weights (`INITIAL_CHECKPOINT=None`). Run and plot
names are distinct from the symmetric-value experiments. Existing automatic
Great Lakes plot sync discovers the new images.

Configured settings match the preceding pair recipes: sizes 3–9, random phase
targets, random detunings with spans up to 4 GHz, five contexts per size, five
density requests per context, eight mirrored actions per request, 1,400 native
500 ns simulations per iteration, 1,000 iterations, the same learning rate and
exploration schedules, reward, PCGrad and validation. Validation remains 448
cases at iteration 1, every 50 iterations and the final iteration; plots update
every 10 iterations. No precision continuation or diagnostic update changes
are included.

Removing tied actions restores the existing directed policy, including its
source-minus-receiver target-phase offset and independent Gaussian scores.
The reciprocal-pair-only division of edge SD by sqrt(2) no longer applies;
the configured schedule values themselves are unchanged. Budget sampling is
unchanged. This is the existing asymmetric-value parameterization, not a new
policy layer.

The selector still ranks reciprocal pairs by the sum of their two strengths,
keeps a connected maximum-strength spanning tree, and adds pairs to the
requested density. Surviving directed strengths are normalized proportionally
to the predicted total budget. Both directions remain enabled even when their
values differ. Symmetric connectivity therefore does not impose value symmetry.

Submit both on Great Lakes:

```sh
git pull --ff-only origin updates
bash scripts/submit_asymmetric_value_density_tests.sh
```

The launcher retains 32 CPUs, 31 workers, 64 GB and seven days per independent
job, with five contexts × eight actions vectorized per worker task. Add
`--dry-run` to preview or `--smoke-test` to run two iterations.
