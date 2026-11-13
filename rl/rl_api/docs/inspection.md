# One-cell inspection

Open `experiments/inspect_current.py`, edit the settings at the top, and run
its single `#%%` cell. It works in VS Code Interactive or as a standalone
script. It does not parse notebook arguments or start training.

The cell leaves `inspection`, `model`, `designs`, `results`, `figures`,
`kappa_per_ns` and `phi_p_rad` available. Rerunning the cell loads the selected
checkpoint again and performs a fresh inspection.

## Settings follow the original inspector

| Setting | Meaning |
| --- | --- |
| `RUN_DIRECTORY`, `CHECKPOINT_KIND` | Training run folder and current/best/final checkpoint |
| `CHECKPOINT` | Optional explicit path, including an original non-API checkpoint |
| `N_LASERS`, `N_CLUSTERS` | Array size and automatically generated phase clusters |
| `MANUAL_TARGET_PHASES_PI` | Optional target vector in units of pi; None uses clusters |
| `DETUNING_MODE` | equal, random or manual |
| `DETUNING_SPAN_GHZ` | Full width used by equal/random detuning modes |
| `MANUAL_DETUNINGS_GHZ` | One value per laser in manual mode; centered before inference |
| `TOPOLOGY_MATRIX_SOURCE_RECEIVER` | Optional mask or stack using source rows and receiver columns |
| `NUMBER_OF_CANDIDATES` | Random topology count, or action count for a supplied mask |
| `SAMPLE_CONTINUOUS_ACTIONS` | 0 uses mean coupling actions; 1 samples them |
| `MANUAL_BUDGET_PER_NS` | None uses predicted budgets; a nonnegative number sets the total directed coupling strength of every candidate in ns⁻¹ |
| `ENSEMBLE_TOPOLOGY_ONLY` | With ensemble validation, True uses mean continuous actions regardless of the sampling flag, matching the original inspector |
| `ENSEMBLE_SELECTED_TOPOLOGY` | best (cosine), best_rms, sparsest, min_budget; None shows only ensemble results |
| `PLOT_TOPOLOGY_ENSEMBLE` | Enable ensemble-average matrices and ensemble trajectories |
| `ENSEMBLE_TRAJECTORY_MODE` | all trajectories or mean_sd bands |
| `RUN_VALIDATION` | True simulates; False produces only an inferred design summary |
| `VALIDATION_TIME_NS`, `NOISE_AMPLITUDE`, `NOISE_ITERATIONS` | Physical evaluation duration and noise |
| `INFERENCE_RANDOM_SEED`, `NOISE_RANDOM_SEED` | Optional independent inference and simulation seeds |
| `OVERRIDE_COUPLING_RAMP` and ramp settings | Optional inspection-only ramp override |
| `SHOW_FAR_FIELD` and far-field settings | Include the angle/time map and optional final slice inside the dynamics figure |
| `DISPLAY_PHASE_ERROR_DEG` | Display guide, not a learning reward threshold |
| `PLOT_STYLE` | Original LaTeX Computer Modern fonts, sizes and DPI |
| `FIGURE_DIRECTORY`, `SHOW_PLOTS` | Optional output folder and display toggle |
| `PLOT_LEGEND_LOCATION` | Fixed location for automatic legends; upper right avoids scanning every point of every trajectory. best restores the slower automatic search. |

Example: three lasers with relative phases phi2-phi1=-1 and phi3-phi1=2 rad:

```python
N_LASERS = 3
MANUAL_TARGET_PHASES_PI = [0.0, -1.0 / np.pi, 2.0 / np.pi]
DETUNING_MODE = "manual"
MANUAL_DETUNINGS_GHZ = [-1.5, -0.5, 1.5]
```

To impose a total coupling budget during inspection, set
`MANUAL_BUDGET_PER_NS = 30.0`. Each candidate's strength matrix is multiplied
by `30.0 / kappa.sum()` before simulation and plotting. Relative strengths,
coupling phases, and topology masks are preserved. This is a total across all
directed links, not a per-laser or per-link value. It applies to selected and
ensemble results, including inference-only plots. Zero disables all coupling;
`None` restores the model's predicted budgets. A positive budget requires a
nonzero predicted strength matrix. Model weights and training are unaffected.

## Figures

With validation and the ensemble enabled, the figures appear in this order:

1. **Selected design:** coupling strengths, coupling phases, settled phase
   comparison and detuning-ordered topology arcs in one horizontal figure.
2. **Selected dynamics:** laser frequencies, relative phases, phase reward,
   individual/coherent optical power, and the far-field map/final slice in one
   figure. It uses the original individual-design dynamics renderer.
3. **Ensemble-average coupling matrices:** mean strengths across valid
   topologies, including absent links as zero, and circular mean phases over
   enabled links only. A phase mean with a vanishing circular resultant is
   undefined and shown in black. Source rows and receiver columns match the
   selected-design matrices. Arrays larger than ten lasers use six index
   labels per matrix axis, matching the single-case plot. A third panel uses
   the same phase circle: blue target points and tolerance arcs, orange
   ensemble circular means, and an orange ring heatmap of aligned settled
   phases across valid topologies and noise realizations (percent per 2° bin).
4. **Topology ensemble:** the original trajectories and settled distributions,
   using all or mean_sd presentation.

Set `ENSEMBLE_SELECTED_TOPOLOGY = None` to show only the ensemble-average
matrices and topology-ensemble time series. No candidate is selected, and the
selected design/dynamics/far-field figures are skipped. This requires
`RUN_VALIDATION=True` and `PLOT_TOPOLOGY_ENSEMBLE=True`. The equivalent API calls
are `inspect(select=None)` and `results.plot(select=None, ...)`.

There are no separate phase-comparison or far-field windows. Disabling the
ensemble leaves the first two figures. Disabling physical validation leaves
only the inferred design summary. Simulation is performed once for the batch;
all rendering uses its cached trajectories.

The default output is
`rl/rl_api/runs/inspection/<run-folder>/<checkpoint-kind>/figures/`.
The script does not change the original inspector's files, plotting globals,
or output directory. Old Bernoulli-gate-specific controls remain deferred;
the current topology-input model has no learned gate logits.

General API calls can request this same layout with
`results.plot(options=PlotOptions(layout="inspection"))`. The separate-panel
layout remains available through `layout="separate"`. To replot already
evaluated data without another simulation, use:

```python
with plt.rc_context(PLOT_STYLE):
    figures = results.plot(select="min_budget", options=plot_options())
```

## Plotting speed

The inspector fixes automatic legends at `PLOT_LEGEND_LOCATION="upper right"`
by default. Matplotlib's `"best"` placement repeatedly tests possible positions
against all trajectory vertices during image export and notebook display.
With 100 ensemble candidates this can cost more time than rendering the lines.
Explicit original legend placements, including the phase-circle center, are
preserved. Output-power legends use center right in the selected dynamics and
ensemble figures. No time points, candidates, extrema, settled profiles, DPI or LaTeX
fonts are reduced by this optimization.
