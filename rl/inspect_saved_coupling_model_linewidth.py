#%%
"""Inspect a saved variable-M coupling policy and estimate PSD linewidth.

This follows ``inspect_saved_coupling_model_variable_m.py``: it loads one
checkpoint, designs a coupling matrix for a manually selected target and
detuning distribution, and produces the same matrix and dynamics figures.
The validation run is longer and its noisy settled tail is additionally
passed to the PSD linewidth estimator used by ``examples/linewidth``.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.conditional_coupling_designer_variable_m import (
    design_for_target,
    load_checkpoint,
)
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design,
    simulate_and_plot_best_design,
)
from rl.paths import MODEL_DIR, RESULTS_DIR

# The linewidth example selects a non-interactive backend at import time. Keep
# the backend selected by this script/notebook so ``plt.show()`` behaves as it
# does in the existing inspection workflow.
_inspection_backend = matplotlib.get_backend()

# The PSD implementation is shared with the linewidth examples.  It computes
# the combined-field, mean-laser-phase, and individual-laser frequency-noise
# PSDs and converts their white floor using Delta_nu = pi*S_nu.
from examples.linewidth import linewidth_autocorr as linewidth_engine

if matplotlib.get_backend() != _inspection_backend:
    matplotlib.use(_inspection_backend, force=True)


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "current"  # "best" or "current"
# These identify the multi-size run that produced the checkpoint.
TRAINING_N_LASERS = (2, 2)
CHECKPOINT_SIZE_LABEL = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)

N_LASERS = 2
MODEL_SUFFIX = "bigru_2_lasers_test"  # Optional checkpoint suffix.
MODEL_ARCHITECTURE = "auto"  # "auto", "bigru", or "pooled"

FIGURE_DIRECTORY = RESULTS_DIR / "analyses" / "linewidth" / "inspection"
FIGURE_SUFFIX = f"{N_LASERS}_in_phase_linewidth"

# Enter target phases as multiples of pi. This mirrors the current inspection
# script: set N_CLUSTERS=1 for in-phase, or increase it for clustered targets.
N_CLUSTERS = 1
cluster_phases_pi = 2.0 * np.arange(N_CLUSTERS) / N_CLUSTERS
base_count, remainder = divmod(N_LASERS, N_CLUSTERS)
cluster_counts = [
    base_count + (cluster < remainder)
    for cluster in range(N_CLUSTERS)
]
MANUAL_TARGET_PHASES_PI = np.repeat(
    cluster_phases_pi,
    cluster_counts,
).tolist()

# Enter one detuning per laser in GHz. Values are centered below and then
# passed to the simulator in physical GHz units.
MANUAL_DETUNINGS_GHZ = 2.0 * 0.2 * np.linspace(-5.0, 5.0, N_LASERS)

NUMBER_OF_CANDIDATES = 1

# A PSD needs a substantially longer stationary record than the normal
# inspection. Five microseconds gives a lower frequency bin near 0.2 MHz
# when the full retained record is used as one Welch segment.
VALIDATION_TIME_NS = 10_000.0
PSD_BURN_IN_NS = 1_000.0

# Noise is required for a nonzero stochastic linewidth estimate. Increase
# NOISE_ITERATIONS for a smoother ensemble median, at the cost of memory.
NOISE_AMPLITUDE = 1.0
NOISE_ITERATIONS = 50
NOISE_RANDOM_SEED = None

# Optional inspection-only coupling-ramp override. These are multiples of the
# physical delay tau, not nanoseconds directly.
OVERRIDE_COUPLING_RAMP = True
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 100.0

# PSD controls, following examples/linewidth/linewidth_autocorr.py.
PSD_CHANNEL_TO_REPORT = "common"  # "combined", "common", or "laser1"
PSD_FLOOR_BAND_HZ = (0.0, 2.0e6)
PSD_WELCH_NPERSEG = None  # None retains the complete post-burn-in record.
PSD_WELCH_OVERLAP_FRACTION = 0.5
PSD_PLOT_ALL_LASERS = True

PLOT_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "lines.linewidth": 1.8,
    "figure.dpi": 200,
}


def _linewidth_text(linewidth_hz: float) -> str:
    """Format a linewidth in a compact unit for plot labels."""
    if not np.isfinite(linewidth_hz) or linewidth_hz <= 0.0:
        return "n/a"
    if linewidth_hz >= 1.0e6:
        return f"{linewidth_hz / 1.0e6:.3g} MHz"
    if linewidth_hz >= 1.0e3:
        return f"{linewidth_hz / 1.0e3:.3g} kHz"
    return f"{linewidth_hz:.3g} Hz"


def compute_psd_linewidth(simulation: dict, run_config) -> dict:
    """Estimate PSD linewidths from the settled portion of a simulation."""
    if NOISE_AMPLITUDE <= 0.0:
        raise ValueError(
            "PSD linewidth estimation requires NOISE_AMPLITUDE > 0."
        )

    time_seconds = np.asarray(simulation["time_seconds"], dtype=float)
    states = np.asarray(simulation["states"], dtype=float)
    if states.ndim != 3 or states.shape[1] != 3 * run_config.n_lasers:
        raise ValueError(
            "Expected states with shape "
            f"(cases, {3 * run_config.n_lasers}, time), got {states.shape}."
        )
    if time_seconds.ndim != 1 or states.shape[-1] != time_seconds.size:
        raise ValueError("states and time_seconds have incompatible shapes")

    ramp_end_ns = (
        run_config.coupling_ramp_start_delays
        + run_config.coupling_ramp_rise_delays / 0.8
    ) * run_config.delay_seconds * 1.0e9
    analysis_start_ns = max(float(PSD_BURN_IN_NS), float(ramp_end_ns))
    start_index = int(
        np.searchsorted(time_seconds * 1.0e9, analysis_start_ns, side="left")
    )
    if time_seconds.size - start_index < 16:
        raise ValueError(
            "The long simulation does not contain enough saved samples after "
            f"the {analysis_start_ns:.3g} ns PSD burn-in. Increase "
            "VALIDATION_TIME_NS or reduce PSD_BURN_IN_NS."
        )

    t_analysis = time_seconds[start_index:] - time_seconds[start_index]
    photon_traces = states[:, 1::3, start_index:]
    phase_traces = states[:, 2::3, start_index:]

    linewidth_engine.psd_floor_band_hz = tuple(PSD_FLOOR_BAND_HZ)
    linewidth_engine.psd_welch_nperseg = PSD_WELCH_NPERSEG
    linewidth_engine.psd_welch_overlap_fraction = (
        PSD_WELCH_OVERLAP_FRACTION
    )
    linewidth_engine.use_available_data_if_tmax_too_short = True
    result = linewidth_engine.frequency_noise_psd_linewidth(
        t_analysis,
        photon_traces,
        phase_traces,
    )
    result["analysis_start_ns"] = analysis_start_ns
    result["analysis_time_ns"] = float(t_analysis[-1] - t_analysis[0]) * 1.0e9
    result["sample_dt_seconds"] = float(np.median(np.diff(t_analysis)))
    if PSD_CHANNEL_TO_REPORT not in result["channels"]:
        raise ValueError(
            "PSD_CHANNEL_TO_REPORT must be one of "
            f"{sorted(result['channels'])}; got {PSD_CHANNEL_TO_REPORT!r}."
        )
    return result


def plot_psd_linewidth(psd_result: dict, figure_label: str) -> plt.Figure:
    """Plot the positive frequency PSD and annotate its linewidth estimate."""
    frequencies_hz = np.asarray(psd_result["frequencies_hz"], dtype=float)
    if frequencies_hz.size < 2:
        raise ValueError("The PSD contains fewer than two frequency bins")

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    positive_frequency = frequencies_hz > 0.0
    channel_names = [PSD_CHANNEL_TO_REPORT]
    channel_names.extend(
        name for name in ("combined", "common") if name not in channel_names
    )
    if PSD_PLOT_ALL_LASERS:
        channel_names.extend(
            name
            for name in psd_result["channels"]
            if name.startswith("laser") and name not in channel_names
        )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = False
    for index, channel_name in enumerate(channel_names):
        channel = psd_result["channels"].get(channel_name)
        if channel is None:
            continue
        psd = np.asarray(channel["psd"], dtype=float)
        valid = positive_frequency & np.isfinite(psd) & (psd > 0.0)
        if not np.any(valid):
            continue
        linewidth_hz = float(channel["linewidth_hz"])
        label = (
            f"{channel_name}: $\\Delta\\nu$="
            f"{_linewidth_text(linewidth_hz)}"
        )
        axis.loglog(
            frequencies_hz[valid],
            psd[valid],
            color=colors[index % len(colors)],
            label=label,
        )
        plotted = True

    if not plotted:
        raise ValueError("The PSD contains no finite positive values to plot")

    floor_low, floor_high = linewidth_engine.psd_floor_band_hz
    positive_values = frequencies_hz[positive_frequency]
    floor_low = max(float(floor_low), float(positive_values[0]))
    floor_high = min(float(floor_high), float(positive_values[-1]))
    if floor_low < floor_high:
        axis.axvspan(
            floor_low,
            floor_high,
            color="0.5",
            alpha=0.12,
            label="linewidth floor band",
        )

    axis.set(
        xlabel="frequency offset (Hz)",
        ylabel=r"frequency-noise PSD $S_\nu$ (Hz$^2$/Hz)",
        title="PSD linewidth estimate",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=PLOT_STYLE["legend.fontsize"])
    output_path = FIGURE_DIRECTORY / f"selected_design_psd_{figure_label}.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def main() -> dict:
    """Load, design, simulate, plot, and estimate the selected linewidth."""
    if CHECKPOINT_KIND not in {"best", "current"}:
        raise ValueError('CHECKPOINT_KIND must be "best" or "current"')
    if NOISE_ITERATIONS < 1:
        raise ValueError("NOISE_ITERATIONS must be at least 1")

    suffix = MODEL_SUFFIX
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    checkpoint = MODEL_DIR / (
        f"conditional_coupling_variable_m_{CHECKPOINT_SIZE_LABEL}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy, _, config, metadata = load_checkpoint(
        checkpoint,
        device="cpu",
        architecture=MODEL_ARCHITECTURE,
    )
    config = replace(
        config,
        n_lasers=N_LASERS,
        n_jobs=1,
        jupyter_mode=False,
        noise_amplitude=0.0,
    )
    if OVERRIDE_COUPLING_RAMP:
        config = replace(
            config,
            coupling_ramp_start_delays=COUPLING_RAMP_START_DELAYS,
            coupling_ramp_rise_delays=COUPLING_RAMP_RISE_DELAYS,
        )

    target_phases_rad = np.asarray(MANUAL_TARGET_PHASES_PI, dtype=float) * np.pi
    detunings_ghz = np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    expected_shape = (config.n_lasers,)
    if target_phases_rad.shape != expected_shape:
        raise ValueError(
            f"MANUAL_TARGET_PHASES_PI must have shape {expected_shape}"
        )
    if detunings_ghz.shape != expected_shape:
        raise ValueError(
            f"MANUAL_DETUNINGS_GHZ must have shape {expected_shape}"
        )
    detunings_ghz = detunings_ghz - np.mean(detunings_ghz)

    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
        f"{metadata['iteration']} (held-out={metadata['held_out_reward']:.3f})"
    )
    print(f"Training sizes: {TRAINING_N_LASERS}; evaluating M={N_LASERS}")
    print("Centered detunings (GHz):", np.round(detunings_ghz, 4))

    design = design_for_target(
        policy,
        target_phases_rad,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        top_k=1,
        detuning_distribution_ghz=detunings_ghz,
    )[0]
    print(f"Phase reward: {design['phase_reward']:.4f}")
    print(
        "Magnitude symmetry penalty: "
        f"{design['magnitude_symmetry_penalty']:.4f}"
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    figure_suffix = FIGURE_SUFFIX
    if NOISE_AMPLITUDE > 0.0:
        figure_suffix = (
            f"{figure_suffix}_noise" if figure_suffix else "_noise"
        )
    if figure_suffix and not figure_suffix.startswith("_"):
        figure_suffix = f"_{figure_suffix}"
    figure_label = f"manual_target{figure_suffix}"

    plotted_kappa_max = max(float(np.max(design["kappa_per_ns"])), 1.0e-12)
    with plt.rc_context(PLOT_STYLE):
        plot_selected_design(
            design,
            config,
            desired_state=figure_label,
            vertical_layout=False,
            kappa_vmax_per_ns=plotted_kappa_max,
        )
        plt.show()

        np.random.seed(NOISE_RANDOM_SEED)
        validation_config = replace(
            config,
            noise_amplitude=NOISE_AMPLITUDE,
        )
        simulation = simulate_and_plot_best_design(
            design,
            validation_config,
            validation_delay_count=(
                VALIDATION_TIME_NS / (config.delay_seconds * 1.0e9)
            ),
            desired_state=figure_label,
            detuning_distribution_ghz=design["detuning_distribution_ghz"],
            legend_fontsize=PLOT_STYLE["legend.fontsize"],
            noise_iterations=NOISE_ITERATIONS,
        )
        psd_result = compute_psd_linewidth(simulation, validation_config)
        psd_figure = plot_psd_linewidth(psd_result, figure_label)
        plt.show()

    print(
        f"PSD analysis starts at {psd_result['analysis_start_ns']:.3g} ns; "
        f"retained {psd_result['analysis_time_ns']:.3g} ns."
    )
    print("PSD linewidth estimates:")
    for channel_name, channel in psd_result["channels"].items():
        print(
            f"  {channel_name:>8s}: "
            f"median={_linewidth_text(float(channel['linewidth_hz']))}, "
            f"mean={_linewidth_text(float(channel['linewidth_mean_hz']))}, "
            f"std={_linewidth_text(float(channel['linewidth_std_hz']))}"
        )
    print(
        "Saved figures:\n"
        f"  {FIGURE_DIRECTORY / f'selected_design_{figure_label}.png'}\n"
        f"  {FIGURE_DIRECTORY / f'selected_design_dynamics_{figure_label}.png'}\n"
        f"  {FIGURE_DIRECTORY / f'selected_design_psd_{figure_label}.png'}"
    )
    return {
        "policy": policy,
        "config": config,
        "metadata": metadata,
        "design": design,
        "simulation": simulation,
        "psd": psd_result,
        "psd_figure": psd_figure,
    }


if __name__ == "__main__":
    results = main()

# %%
