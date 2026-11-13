#%%
"""Inspect the current model: edit the settings, then run this single cell.

The settings and selected-design layout follow the original inspector. Training
is never started here. Run as a cell or with python -m
rl.rl_api.experiments.inspect_current.
"""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _folder in (_start, *_start.parents):
    if (_folder / "rl/rl_api/__init__.py").is_file():
        if str(_folder) not in sys.path:
            sys.path.insert(0, str(_folder))
        API_DIRECTORY = _folder / "rl/rl_api"
        break
else:
    raise ModuleNotFoundError("Start the kernel inside the vcsel_lib repository")

from rl.rl_api import Context, CouplingModel, Topology, PlotOptions


# Checkpoint settings.
RUN_DIRECTORY = API_DIRECTORY / "runs/reachable_rms_bins5_randbands5_phase_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k"
CHECKPOINT_KIND = "current"       # current, best, or final.
CHECKPOINT = None                  # Optional explicit .pt path instead of RUN_DIRECTORY/kind.
INFERENCE_DEVICE = "auto"
FIGURE_DIRECTORY = None            # None -> API runs/inspection/<run>/<checkpoint kind>/.

# Laser count, phase targets and detunings: same conventions as the old inspector.
N_LASERS = 5
N_CLUSTERS = 5
MANUAL_TARGET_PHASES_PI = None     # None generates clusters; otherwise N_LASERS values in units of pi.
# For three lasers with relative phases [0, -1, 2] rad, use N_LASERS=3 and
# MANUAL_TARGET_PHASES_PI = [0.0, -1.0 / np.pi, 2.0 / np.pi].
DETUNING_MODE = "equal"          # equal, random, or manual.
DETUNING_SPAN_GHZ = 1.0
MANUAL_DETUNINGS_GHZ = np.array([1.5, -0.5, -1.5])

# Optional binary matrix or stack: SOURCE ROWS, RECEIVER COLUMNS, as before.
TOPOLOGY_MATRIX_SOURCE_RECEIVER = None
NUMBER_OF_CANDIDATES = 100
SAMPLE_CONTINUOUS_ACTIONS = 0
INFERENCE_RANDOM_SEED = None
MANUAL_BUDGET_PER_NS = None       # Total sum of directed strengths per candidate; None uses model budget.

# Ensemble selection and presentation.
PLOT_TOPOLOGY_ENSEMBLE = True
ENSEMBLE_SELECTED_TOPOLOGY = None#"best"  # best (cosine), best_rms, sparsest, min_budget; None = ensemble only.
ENSEMBLE_TRAJECTORY_MODE = "all"    # all or mean_sd.
ENSEMBLE_PROFILE_SMOOTHING_BINS = 3.5
ENSEMBLE_TOPOLOGY_ONLY = True         # Ensemble uses mean coupling actions when True.
NETWORK_MAX_DISPLAY_EDGES = 10_000
NETWORK_EDGE_SELECTION = "strongest"
TOPOLOGY_BINS = N_LASERS
DISPLAY_PHASE_ERROR_DEG = 3.0
SHOW_PLOTS = True
PLOT_LEGEND_LOCATION = "upper right"  # "best" is much slower with dense ensemble traces.

# Physical validation; False produces only the inferred design summary.
RUN_VALIDATION = True
VALIDATION_TIME_NS = 200.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None
SIMULATION_PROGRESS = True
SMOOTH_FREQUENCIES = True
OVERRIDE_COUPLING_RAMP = False
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0

# Far field is part of the selected dynamics figure, not a separate window.
SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
WAVELENGTH_M = 910e-9
FAR_FIELD_THETA_RANGE_DEG = (-10.0, 10.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 5.0
SHOW_FAR_FIELD_FINAL_SLICE = True
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = None

PLOT_STYLE = {
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], "font.size": 14,
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "figure.titlesize": 20,
    "lines.linewidth": 1.8, "figure.dpi": 200,
}


# Helpers -- everything below reads the settings above.
def checkpoint_path(override=None):
    if override is not None or CHECKPOINT is not None:
        return Path(CHECKPOINT if override is None else override).expanduser()
    if CHECKPOINT_KIND not in {"current", "best", "final"}:
        raise ValueError("CHECKPOINT_KIND must be current, best, or final")
    return Path(RUN_DIRECTORY) / f"{CHECKPOINT_KIND}.pt"


def load_model(checkpoint=None):
    path = checkpoint_path(checkpoint)
    directory = FIGURE_DIRECTORY or API_DIRECTORY / "runs/inspection" / path.parent.name / path.stem
    try:
        model = CouplingModel.load(path, device=INFERENCE_DEVICE, output_directory=directory)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No checkpoint found at {path}. Edit RUN_DIRECTORY/CHECKPOINT_KIND "
            "or CHECKPOINT, or wait until training saves its first iteration."
        ) from error
    if OVERRIDE_COUPLING_RAMP:
        model.backend_config = replace(model.backend_config,
            coupling_ramp_start_delays=COUPLING_RAMP_START_DELAYS,
            coupling_ramp_rise_delays=COUPLING_RAMP_RISE_DELAYS)
    print(f"Loaded checkpoint: {path}\nIteration: {model.metadata.get('iteration', 'unknown')}")
    return model


def make_context():
    if not isinstance(N_LASERS, int) or N_LASERS < 2:
        raise ValueError("N_LASERS must be an integer >=2")
    if MANUAL_TARGET_PHASES_PI is None:
        if not isinstance(N_CLUSTERS, int) or N_CLUSTERS < 1:
            raise ValueError("N_CLUSTERS must be a positive integer")
        base, remainder = divmod(N_LASERS, N_CLUSTERS)
        counts = [base + (cluster < remainder) for cluster in range(N_CLUSTERS)]
        target = np.repeat(2*np.pi*np.arange(N_CLUSTERS)/N_CLUSTERS, counts)
    else:
        target = np.asarray(MANUAL_TARGET_PHASES_PI, dtype=float)*np.pi
    if DETUNING_MODE == "equal":
        detuning = DETUNING_SPAN_GHZ*np.linspace(-.5, .5, N_LASERS)
    elif DETUNING_MODE == "random":
        detuning = DETUNING_SPAN_GHZ*np.sort(np.random.uniform(-.5, .5, N_LASERS))
    elif DETUNING_MODE == "manual":
        detuning = np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    else:
        raise ValueError("DETUNING_MODE must be equal, random, or manual")
    if target.shape != (N_LASERS,) or detuning.shape != (N_LASERS,):
        raise ValueError("Manual phases and detunings must each have N_LASERS values")
    return Context(target_phases_rad=target, detunings_ghz=detuning-detuning.mean())


def plot_options():
    return PlotOptions(layout="inspection", use_tex=PLOT_STYLE["text.usetex"],
        dpi=PLOT_STYLE["figure.dpi"], trajectory_mode=ENSEMBLE_TRAJECTORY_MODE,
        profile_smoothing_bins=ENSEMBLE_PROFILE_SMOOTHING_BINS,
        max_display_edges=NETWORK_MAX_DISPLAY_EDGES, edge_selection=NETWORK_EDGE_SELECTION,
        topology_bins=TOPOLOGY_BINS, phase_error_deg=DISPLAY_PHASE_ERROR_DEG,
        legend_location=PLOT_LEGEND_LOCATION,
        show_ensemble=PLOT_TOPOLOGY_ENSEMBLE, show_far_field=SHOW_FAR_FIELD,
        emitter_spacing_um=EMITTER_SPACING_UM, wavelength_m=WAVELENGTH_M,
        theta_range_deg=FAR_FIELD_THETA_RANGE_DEG, n_theta=FAR_FIELD_N_THETA,
        max_time_points=FAR_FIELD_MAX_TIME_POINTS, element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
        far_field_final_slice=SHOW_FAR_FIELD_FINAL_SLICE,
        angular_blur_fwhm_deg=FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG)


def apply_manual_budget(designs, budget_per_ns):
    """Rescale each candidate's strengths before plotting or physical validation."""
    if budget_per_ns is None:
        return designs
    budget = np.asarray(budget_per_ns, dtype=float)
    if budget.ndim != 0 or not np.isfinite(budget) or budget < 0:
        raise ValueError("MANUAL_BUDGET_PER_NS must be None or a finite nonnegative total")
    totals = designs.budgets_per_ns
    if np.any(~np.isfinite(totals)) or (budget > 0 and np.any(totals <= 0)):
        raise ValueError("Cannot impose a positive budget on a zero or nonfinite coupling matrix")
    factors = np.divide(budget, totals, out=np.zeros_like(totals), where=totals > 0)
    strengths = designs.coupling_strengths_per_ns * factors[:, None, None]
    raw = {**designs.raw, "kappa_per_ns": strengths,
           "raw_total_coupling_per_ns": strengths.sum(axis=(-2, -1)),
           "redistribution_shortfall_per_ns": np.zeros_like(totals)}
    return replace(designs, raw=raw)


_CONFIG_SELECTION = object()


def inspect(checkpoint=None, *, select=_CONFIG_SELECTION, count=None, show=None):
    """Predict/simulate once; select=None plots only ensemble results."""
    selection = ENSEMBLE_SELECTED_TOPOLOGY if select is _CONFIG_SELECTION else select
    make_ensemble = bool(PLOT_TOPOLOGY_ENSEMBLE and RUN_VALIDATION)
    if selection is None and not make_ensemble:
        raise ValueError("Ensemble-only inspection requires RUN_VALIDATION=True "
                         "and PLOT_TOPOLOGY_ENSEMBLE=True")
    model = load_model(checkpoint)
    context = make_context()
    continuous_mean = ENSEMBLE_TOPOLOGY_ONLY if make_ensemble else not SAMPLE_CONTINUOUS_ACTIONS
    topology = (None if TOPOLOGY_MATRIX_SOURCE_RECEIVER is None else
                Topology.from_source_receiver(TOPOLOGY_MATRIX_SOURCE_RECEIVER))
    designs = model.propose(context, topology=topology,
        count=NUMBER_OF_CANDIDATES if count is None else count,
        continuous_actions="mean" if continuous_mean else "sample", seed=INFERENCE_RANDOM_SEED)
    designs = apply_manual_budget(designs, MANUAL_BUDGET_PER_NS)
    if MANUAL_BUDGET_PER_NS is not None:
        print(f"Manual total coupling budget: {float(MANUAL_BUDGET_PER_NS):g} ns^-1 per candidate")
    options = plot_options()
    show = SHOW_PLOTS if show is None else show
    results, chosen = None, None
    if RUN_VALIDATION:
        results = designs.evaluate(duration_ns=VALIDATION_TIME_NS, noise_amplitude=NOISE_AMPLITUDE,
            noise_realizations=NOISE_ITERATIONS, seed=NOISE_RANDOM_SEED,
            progress=SIMULATION_PROGRESS, smooth_frequencies=SMOOTH_FREQUENCIES)
        if selection is not None:
            chosen = results.select(selection if make_ensemble else "best")
            print(f"Selected candidate {chosen.index+1}/{len(designs)}: {chosen.metrics}")
        else:
            print(f"Ensemble only: {len(designs)} candidates.")
        with plt.rc_context(PLOT_STYLE):
            figures = results.plot(select=chosen.index if chosen is not None else None, output_directory=FIGURE_DIRECTORY,
                                   show=show, options=options)
    else:
        print("Inference only; no dynamics were simulated.")
        with plt.rc_context(PLOT_STYLE):
            figures = designs.plot(output_directory=FIGURE_DIRECTORY, show=show, options=options)
    return dict(model=model, context=context, designs=designs, results=results,
                chosen=chosen, figures=figures)


if __name__ == "__main__":
    inspection = inspect()
    model = inspection["model"]
    designs = inspection["designs"]
    results = inspection["results"]
    figures = inspection["figures"]
    kappa_per_ns = designs.coupling_strengths_per_ns
    phi_p_rad = designs.coupling_phases_rad
