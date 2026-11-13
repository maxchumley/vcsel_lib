#%%
"""Inspect a direct Bernoulli-z sparse checkpoint in VS Code Interactive."""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c, hbar
from scipy.ndimage import gaussian_filter1d

for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_designer_variable_m as designer
from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.conditional_coupling_bernoulli_sparse import (
    dense_reference_designs, dense_reference_reward, encode_context, infer_designs,
    load_checkpoint, reference_budget_for_m,
    relative_coupling_budget, reward_function, root_reach_fractions,
    lexicographic_candidate_scores, maximum_phase_error_deg,
)
from rl.conditional_coupling_designer_variable_m import (
    calculate_phase_reward, simulate_with_vcsel,
)
from rl.conditional_coupling_run_variable_m import (
    PHASE_TICKS, PHASE_TICK_LABELS, phase_reward_trace,
    plot_selected_design, simulate_and_plot_best_design,
)
from rl.inference_plots import plot_inference_summary
from rl.paths import RESULTS_DIR
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import (
    config as training_config, MODEL_SUFFIX as TRAINING_MODEL_SUFFIX,
)


CHECKPOINT_KIND = "current"  # "current", "best", or "final"
# Change only MODEL_FILE_SUFFIX to switch checkpoints. The completed run
# used cosine phase reward, topology PCGrad, and no DE teacher.
PREVIOUS_MODEL_FILE_SUFFIX = "_star_wide_phase_topology_pcgrad_no_de"
# For the previous run, set MODEL_FILE_SUFFIX = PREVIOUS_MODEL_FILE_SUFFIX.
# Any other saved model suffix can also be entered as a quoted string.
MODEL_FILE_SUFFIX = TRAINING_MODEL_SUFFIX


# TRAINING_MODEL_SUFFIX
N_LASERS = 4
N_CLUSTERS = 1#np.copy(N_LASERS)


DETUNING_SPAN_GHZ = 0.1
DETUNING_MODE = "manual"  # "equal", "random", or "manual"
MANUAL_DETUNINGS_GHZ = np.array([-1.5, -0.5, 0.5, 1.5])
RUN_VALIDATION = 1 # True runs a validation simulation; False skips it

# In topology-input mode, provide one source-row/receiver-column MxM matrix,
# a stack of (topologies,M,M) matrices, or None to sample rooted masks.
TOPOLOGY_MATRIX_SOURCE_RECEIVER = None

# np.array([[0, 1, 1, 1, 1],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0]])

# np.array([
#     [0, 1, 1],
#     [1, 0, 0],
#     [0, 0, 0],
# ], dtype=np.uint8)  # Fixed search mask; source rows, receiver columns.
# The options below apply only when inspecting older Bernoulli-gate models.
# Topology-input mode requires TARGET_ACTIVE_FRACTION = None.
TARGET_ACTIVE_FRACTION = None  # None or a number strictly between 0 and 1
SAMPLE_OPTIONAL_GATES = 1  # True samples optional z; tree links stay on.
NUMBER_OF_CANDIDATES = 100  # Sampled topologies, or candidate actions for a fixed mask.
PLOT_TOPOLOGY_ENSEMBLE = True  # Show trajectories and settled distributions.
ENSEMBLE_SELECTED_TOPOLOGY = "best"  # "best", "sparsest", or "min_budget".
# "min_budget" selects the smallest total kappa across enabled directed links.
# Ties favor higher phase reward, then fewer links. Applies to the individual
# topology and dynamics plots shown alongside the full ensemble.
# "sparsest" and "min_budget" have no phase-accuracy cutoff.
ENSEMBLE_TRAJECTORY_MODE = "all"  # "mean_sd" or "all" candidate trajectories.
ENSEMBLE_PROFILE_SMOOTHING_BINS = 3.5  # Gaussian width for right-side distributions.
ENSEMBLE_TOPOLOGY_ONLY = True  # Use mean continuous actions for each ensemble topology.
PRINT_GATE_LOGITS = True
MAX_PRINTED_GATE_LOGITS = 75
INFERENCE_RANDOM_SEED = None  # Reproducible continuous-action samples.
SAMPLE_CONTINUOUS_ACTIONS = 0  # True also samples kappa/phi Gaussians
INFERENCE_DEVICE = "auto"
VALIDATION_TIME_NS = 500.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None
NETWORK_MAX_DISPLAY_EDGES = 10_000
NETWORK_EDGE_SELECTION = "strongest"
TOPOLOGY_BINS = N_LASERS  # display-only binning at large M
DISPLAY_PHASE_ERROR_DEG = 3.0  # plot annotation only; not a model input/reward threshold

OVERRIDE_COUPLING_RAMP = False
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0

SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-10.0, 10.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 5.0
SHOW_FAR_FIELD_FINAL_SLICE = True
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = None
FIGURE_DIRECTORY = RESULTS_DIR / "inference" / f"bernoulli_sparse{MODEL_FILE_SUFFIX}"
FIGURE_SUFFIX = f"M{N_LASERS}_{N_CLUSTERS}clusters_z" + (
    "natural" if TARGET_ACTIVE_FRACTION is None else f"mean{TARGET_ACTIVE_FRACTION:g}"
) + ("_sampled" if SAMPLE_OPTIONAL_GATES else "") + (
    "_fixed_topology" if TOPOLOGY_MATRIX_SOURCE_RECEIVER is not None else ""
)

# N_CLUSTERS defines the phase spacing around the full circle. If there are
# more phase slots than lasers, only the first N_LASERS slots are occupied.
if N_CLUSTERS < 1:
    raise ValueError("N_CLUSTERS must be positive")
cluster_phases_pi = 2.0 * np.arange(N_CLUSTERS) / N_CLUSTERS
base_count, remainder = divmod(N_LASERS, N_CLUSTERS)
cluster_counts = [base_count + (cluster < remainder) for cluster in range(N_CLUSTERS)]
MANUAL_TARGET_PHASES_PI = np.repeat(cluster_phases_pi, cluster_counts).tolist()


PLOT_STYLE = {
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], "font.size": 14,
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "figure.titlesize": 20,
    "lines.linewidth": 1.8, "figure.dpi": 200,
}


def _device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("INFERENCE_DEVICE must be auto, cpu, cuda, or mps")
    return requested


def _detunings() -> np.ndarray:
    if DETUNING_MODE == "equal":
        return DETUNING_SPAN_GHZ * np.linspace(-0.5, 0.5, N_LASERS)
    if DETUNING_MODE == "random":
        return DETUNING_SPAN_GHZ * np.sort(np.random.uniform(-0.5, 0.5, N_LASERS))
    if DETUNING_MODE == "manual":
        return np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    raise ValueError('DETUNING_MODE must be "equal", "random", or "manual"')


def _resolve_topology_input_override(config, topology, active_fraction):
    """Interpret an effectively all-on legacy gate setting as an input mask."""
    if not config.topology_input_mode or active_fraction is None:
        return topology, active_fraction
    if topology is not None:
        raise ValueError(
            "Set TARGET_ACTIVE_FRACTION = None when supplying an explicit topology")
    n_edges = config.n_lasers * (config.n_lasers - 1)
    if not np.isfinite(active_fraction) or not 0 < active_fraction <= 1:
        raise ValueError("TARGET_ACTIVE_FRACTION must lie in (0, 1]")
    if round(active_fraction * n_edges) != n_edges:
        raise ValueError(
            "Topology-input models do not have Bernoulli gates. Set "
            "TARGET_ACTIVE_FRACTION = None and supply "
            "TOPOLOGY_MATRIX_SOURCE_RECEIVER for a specific sparse topology, "
            "or leave the matrix None to sample rooted topologies."
        )
    all_on = np.ones((config.n_lasers, config.n_lasers), dtype=np.uint8)
    np.fill_diagonal(all_on, 0)
    return all_on, None


def select_ensemble_topology(
    phase_rewards: np.ndarray, active_link_counts: np.ndarray, mode: str,
    *, valid_topologies: np.ndarray | None = None,
    coupling_budgets: np.ndarray | None = None,
) -> int:
    """Select by reward, link count, or realized total coupling budget."""
    if mode not in {"best", "sparsest", "min_budget"}:
        raise ValueError("ENSEMBLE_SELECTED_TOPOLOGY must be 'best', 'sparsest', or 'min_budget'")
    rewards = np.asarray(phase_rewards, dtype=float)
    counts = np.asarray(active_link_counts)
    if rewards.ndim != 1 or counts.shape != rewards.shape:
        raise ValueError("phase rewards and link counts must be matching vectors")
    valid = np.isfinite(rewards) & np.isfinite(counts)
    if mode == "min_budget":
        if coupling_budgets is None:
            raise ValueError("min_budget selection requires coupling budgets")
        budgets = np.asarray(coupling_budgets, dtype=float)
        if budgets.shape != rewards.shape:
            raise ValueError("coupling budgets must match phase rewards")
        valid &= np.isfinite(budgets)
    if valid_topologies is not None:
        topology_valid = np.asarray(valid_topologies, dtype=bool)
        if topology_valid.shape != rewards.shape:
            raise ValueError("valid topology mask must match phase rewards")
        valid &= topology_valid
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("No valid ensemble topology is available for selection")
    if mode == "sparsest":
        candidates = candidates[counts[candidates] == counts[candidates].min()]
    elif mode == "min_budget":
        candidates = candidates[budgets[candidates] == budgets[candidates].min()]
    candidates = candidates[rewards[candidates] == rewards[candidates].max()]
    return int(candidates[np.argmin(counts[candidates])])


def summarize_topology_trajectories(
    simulation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    target: np.ndarray,
    config,
    *,
    n_topologies: int,
    noise_iterations: int,
    orientations: np.ndarray,
    include_trajectories: bool = False,
) -> dict:
    """Summarize vectorized simulations; noise is averaged within each topology first."""
    time_seconds, states, frequencies_nd, initial_frequency_ghz = simulation
    n_cases = n_topologies * noise_iterations
    if states.shape[0] != n_cases or len(orientations) != n_cases:
        raise ValueError("simulation and orientation counts must match topology x noise")
    valid_cases = np.all(np.isfinite(states), axis=(1, 2)) & np.all(
        np.isfinite(frequencies_nd), axis=(1, 2))
    valid_topologies = valid_cases.reshape(n_topologies, noise_iterations).all(axis=1)
    if not np.any(valid_topologies):
        raise ValueError("No topology has finite trajectories for every noise realization")
    chosen = np.repeat(valid_topologies, noise_iterations)
    states = states[chosen]
    frequencies_nd = frequencies_nd[chosen].copy()
    initial_frequency_ghz = initial_frequency_ghz[chosen]
    orientations = np.asarray(orientations)[chosen]
    n_valid = int(valid_topologies.sum())
    n_times = states.shape[-1]
    n_lasers = config.n_lasers

    phases = states[:, 2::3]
    relative = np.angle(np.exp(1j * (phases - phases[:, :1])))
    aligned = np.angle(np.exp(1j * orientations[:, None, None] * relative))
    target_relative = designer.make_relative_to_laser_1(target)
    phase_errors = np.angle(np.exp(1j * (aligned - target_relative[None, :, None])))
    reward_cases, tail_start = phase_reward_trace(time_seconds, phase_errors, config)

    frequencies_ghz = frequencies_nd / (
        2 * np.pi * config.photon_lifetime_seconds * 1e9)
    history = initial_frequency_ghz[:, :, ::config.save_every]
    history_points = min(history.shape[-1], n_times)
    frequencies_ghz[:, :, :history_points] = history[:, :, :history_points]
    intensity_to_mw = (
        1e3 * hbar * (2 * np.pi * c / 910e-9)
        / (config.gain_per_second * config.carrier_lifetime_seconds
           * config.photon_lifetime_seconds)
    )
    powers_mw = states[:, 1::3] * intensity_to_mw
    total_field = np.sum(
        np.sqrt(np.maximum(states[:, 1::3], 0.0)) * np.exp(1j * phases), axis=1)
    total_powers_mw = np.abs(total_field) ** 2 * intensity_to_mw

    def per_topology(values: np.ndarray) -> np.ndarray:
        return values.reshape(n_valid, noise_iterations, *values.shape[1:]).mean(axis=1)

    frequency_by_topology = per_topology(frequencies_ghz)
    settled_frequencies_ghz = frequency_by_topology[:, :, tail_start:].mean(axis=-1)
    power_by_topology = per_topology(powers_mw)
    total_power_by_topology = per_topology(total_powers_mw)
    settled_powers_mw = power_by_topology[:, :, tail_start:].mean(axis=-1)
    settled_total_powers_mw = total_power_by_topology[:, tail_start:].mean(axis=-1)
    reward_by_topology = per_topology(reward_cases)
    # Phase is circular: average noise realizations, then topologies on the unit circle.
    phase_by_topology = np.angle(
        np.mean(np.exp(1j * aligned.reshape(
            n_valid, noise_iterations, n_lasers, n_times)), axis=1))
    phase_mean = np.angle(np.mean(np.exp(1j * phase_by_topology), axis=0))
    phase_residuals = np.angle(np.exp(1j * (phase_by_topology - phase_mean[None])))
    settled_phases_rad = np.angle(
        np.mean(np.exp(1j * phase_by_topology[:, :, tail_start:]), axis=-1))

    summary = dict(
        time_ns=time_seconds * 1e9,
        free_running_end_ns=(
            config.coupling_ramp_start_delays * config.delay_seconds * 1e9),
        valid_topologies=valid_topologies,
        n_valid_topologies=n_valid,
        frequency_mean=frequency_by_topology.mean(axis=0),
        frequency_std=frequency_by_topology.std(axis=0),
        settled_frequencies_ghz=settled_frequencies_ghz,
        phase_mean=phase_mean,
        phase_std=phase_residuals.std(axis=0),
        settled_phases_rad=settled_phases_rad,
        reward_mean=reward_by_topology.mean(axis=0),
        reward_std=reward_by_topology.std(axis=0),
        power_mean=power_by_topology.mean(axis=0),
        power_std=power_by_topology.std(axis=0),
        settled_powers_mw=settled_powers_mw,
        total_power_mean=total_power_by_topology.mean(axis=0),
        total_power_std=total_power_by_topology.std(axis=0),
        settled_total_powers_mw=settled_total_powers_mw,
    )
    if include_trajectories:
        summary.update(
            frequency_trajectories=frequency_by_topology,
            phase_trajectories=phase_by_topology,
            reward_trajectories=reward_by_topology,
            power_trajectories=power_by_topology,
            total_power_trajectories=total_power_by_topology,
        )
    return summary


def plot_topology_ensemble(
    summary: dict, target: np.ndarray, *, trajectory_mode: str = "mean_sd",
    profile_smoothing_bins: float = 3.5,
) -> plt.Figure:
    """Plot candidate traces or ensemble statistics with settled profiles."""
    if trajectory_mode not in {"mean_sd", "all"}:
        raise ValueError("trajectory_mode must be 'mean_sd' or 'all'")
    if trajectory_mode == "all" and "frequency_trajectories" not in summary:
        raise ValueError("Candidate trajectories were not included in the summary")
    if not np.isfinite(profile_smoothing_bins) or profile_smoothing_bins <= 0:
        raise ValueError("profile_smoothing_bins must be finite and positive")
    time = summary["time_ns"]
    figure = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = figure.add_gridspec(
        4, 2, width_ratios=(1.0, .22), height_ratios=(1, 1, .5, 1))
    axes = []
    for row in range(4):
        axes.append(figure.add_subplot(
            grid[row, 0], sharex=axes[0] if axes else None))
    frequency_distribution_axis = figure.add_subplot(grid[0, 1], sharey=axes[0])
    phase_distribution_axis = figure.add_subplot(grid[1, 1], sharey=axes[1])
    power_distribution_axis = figure.add_subplot(grid[3, 1], sharey=axes[3])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    target_relative = designer.make_relative_to_laser_1(target)

    def draw_profile(axis, values, vertical_range, *, color, circular=False,
                     linestyle="-", fill=False):
        counts, edges = np.histogram(values, bins=128, range=vertical_range)
        profile = gaussian_filter1d(
            counts.astype(float), sigma=profile_smoothing_bins,
            mode="wrap" if circular else "constant")
        if profile.max() > 0:
            profile /= profile.max()
        centers = (edges[:-1] + edges[1:]) / 2
        if fill:
            axis.fill_betweenx(centers, 0, profile, color=color, alpha=.2)
        axis.plot(profile, centers, color=color, linestyle=linestyle)

    for laser in range(len(target)):
        color = colors[laser % len(colors)]
        for axis, key, label in (
            (axes[0], "frequency", rf"$\dot{{\phi}}_{{{laser + 1}}}$"),
            (axes[1], "phase", rf"$\phi_{{{laser + 1}}}-\phi_1$"),
            (axes[3], "power", rf"$P_{{{laser + 1}}}$"),
        ):
            if trajectory_mode == "all":
                for candidate, trace in enumerate(summary[f"{key}_trajectories"][:, laser]):
                    axis.plot(time, trace, color=color, alpha=.15, lw=.8,
                              label=label if candidate == 0 else None)
            else:
                mean, std = summary[f"{key}_mean"][laser], summary[f"{key}_std"][laser]
                axis.plot(time, mean, color=color, label=label)
                axis.fill_between(time, mean - std, mean + std, color=color, alpha=.18)
        axes[1].axhline(target_relative[laser], color=color, ls="--", alpha=.5)
        phase_distribution_axis.axhline(
            target_relative[laser], color=color, ls="--", alpha=.5)
    subtitle = (rf"mean $\pm$ 1 SD across {summary['n_valid_topologies']} topologies"
                if trajectory_mode == "mean_sd" else
                f"all {summary['n_valid_topologies']} candidate trajectories")
    axes[0].set(ylabel=r"$\dot{\phi}$ (GHz)",
                title="Topology-conditioned trajectories from a free-running state\n"
                      + subtitle)
    # Each candidate contributes one settled frequency per laser. Keep the
    # profile on the same frequency axis as the time-resolved mean traces.
    settled = np.asarray(summary["settled_frequencies_ghz"]).ravel()
    frequency_low, frequency_high = axes[0].get_ylim()
    margin = max(.05 * float(np.ptp(settled)), .005)
    axes[0].set_ylim(min(frequency_low, float(settled.min()) - margin),
                     max(frequency_high, float(settled.max()) + margin))
    frequency_low, frequency_high = axes[0].get_ylim()
    draw_profile(frequency_distribution_axis, settled,
                 (frequency_low, frequency_high), color="tab:orange", fill=True)
    frequency_distribution_axis.set(
        title="Settled laser frequencies\nall candidates",
        xlabel="normalized count", xlim=(0, 1.05))
    frequency_distribution_axis.tick_params(axis="y", left=False, labelleft=False)
    axes[1].set(ylabel=r"wrapped $\phi_i-\phi_1$", ylim=(-1.1*np.pi, 1.1*np.pi),
                yticks=PHASE_TICKS)
    axes[1].set_yticklabels(PHASE_TICK_LABELS)
    settled_phases = np.asarray(summary["settled_phases_rad"])
    if len(target) <= 25:
        for laser in range(1, len(target)):
            draw_profile(phase_distribution_axis, settled_phases[:, laser],
                         (-np.pi, np.pi), color=colors[laser % len(colors)],
                         circular=True)
    else:
        draw_profile(phase_distribution_axis, settled_phases[:, 1:].ravel(),
                     (-np.pi, np.pi), color="tab:orange", circular=True, fill=True)
    phase_distribution_axis.set(
        title="Settled relative phases\nall candidates",
        xlabel="normalized count", xlim=(0, 1.05))
    phase_distribution_axis.tick_params(axis="y", left=False, labelleft=False)
    if trajectory_mode == "all":
        for trace in summary["reward_trajectories"]:
            axes[2].plot(time, trace, color="tab:blue", alpha=.15, lw=.8)
    else:
        axes[2].plot(time, summary["reward_mean"], color="tab:blue")
        axes[2].fill_between(time,
                             np.maximum(summary["reward_mean"] - summary["reward_std"], 0),
                             np.minimum(summary["reward_mean"] + summary["reward_std"], 1),
                             color="tab:blue", alpha=.2)
    axes[2].set(ylabel="phase reward", ylim=(0, 1.02))
    if trajectory_mode == "all":
        for candidate, trace in enumerate(summary["total_power_trajectories"]):
            axes[3].plot(time, trace, "k--", alpha=.15, lw=.8,
                         label=r"$P_{\rm total}$" if candidate == 0 else None)
    else:
        axes[3].plot(time, summary["total_power_mean"], "k--", label=r"$P_{\rm total}$")
        axes[3].fill_between(
            time,
            np.maximum(summary["total_power_mean"] - summary["total_power_std"], 0),
            summary["total_power_mean"] + summary["total_power_std"],
            color="black", alpha=.15)
    axes[3].set(ylabel="output power (mW)", xlabel="time (ns)")
    settled_powers = np.asarray(summary["settled_powers_mw"])
    settled_total_powers = np.asarray(summary["settled_total_powers_mw"])
    all_settled_powers = np.concatenate((settled_powers.ravel(),
                                         settled_total_powers.ravel()))
    power_low, power_high = axes[3].get_ylim()
    margin = max(.05 * float(np.ptp(all_settled_powers)), .005)
    axes[3].set_ylim(min(power_low, float(all_settled_powers.min()) - margin),
                     max(power_high, float(all_settled_powers.max()) + margin))
    power_range = axes[3].get_ylim()
    if len(target) <= 25:
        for laser in range(len(target)):
            draw_profile(power_distribution_axis, settled_powers[:, laser],
                         power_range, color=colors[laser % len(colors)])
    else:
        draw_profile(power_distribution_axis, settled_powers.ravel(),
                     power_range, color="tab:blue", fill=True)
    draw_profile(power_distribution_axis, settled_total_powers,
                 power_range, color="black", linestyle="--")
    power_distribution_axis.set(
        title="Settled output power\nall candidates",
        xlabel="normalized count", xlim=(0, 1.05))
    power_distribution_axis.tick_params(axis="y", left=False, labelleft=False)
    for axis in axes:
        axis.axvspan(0.0, summary["free_running_end_ns"],
                     color="gray", alpha=.2, zorder=0)
    for axis in (*axes, frequency_distribution_axis,
                 phase_distribution_axis, power_distribution_axis):
        axis.grid(alpha=.25)
    for axis in (axes[0], axes[1], axes[3]):
        if len(target) <= 25:
            axis.legend(loc="best", ncol=min(len(target) + 1, 4))
    return figure


def main() -> dict:
    if CHECKPOINT_KIND not in {"current", "best", "final"}:
        raise ValueError("CHECKPOINT_KIND must be current, best, or final")
    if ENSEMBLE_TRAJECTORY_MODE not in {"mean_sd", "all"}:
        raise ValueError("ENSEMBLE_TRAJECTORY_MODE must be 'mean_sd' or 'all'")
    if ENSEMBLE_SELECTED_TOPOLOGY not in {"best", "sparsest", "min_budget"}:
        raise ValueError("ENSEMBLE_SELECTED_TOPOLOGY must be 'best', 'sparsest', or 'min_budget'")
    if NUMBER_OF_CANDIDATES < 1 or NOISE_ITERATIONS < 1:
        raise ValueError("candidate and noise counts must be positive")
    candidate_count = NUMBER_OF_CANDIDATES
    if TOPOLOGY_MATRIX_SOURCE_RECEIVER is not None:
        supplied_topologies = np.asarray(TOPOLOGY_MATRIX_SOURCE_RECEIVER)
        if supplied_topologies.ndim == 3:
            candidate_count = len(supplied_topologies)
    make_ensemble_plot = bool(PLOT_TOPOLOGY_ENSEMBLE and RUN_VALIDATION)
    configured_checkpoint = Path(getattr(training_config, f"{CHECKPOINT_KIND}_checkpoint_file"))
    if not configured_checkpoint.stem.endswith(TRAINING_MODEL_SUFFIX):
        raise ValueError("Training checkpoint path does not end with its model suffix")
    checkpoint = configured_checkpoint.with_name(
        configured_checkpoint.stem[:-len(TRAINING_MODEL_SUFFIX)]
        + MODEL_FILE_SUFFIX + configured_checkpoint.suffix
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} does not exist. Start training first.")
    policy, saved_config, metadata = load_checkpoint(checkpoint, device=_device(INFERENCE_DEVICE))
    config = replace(saved_config, n_lasers=N_LASERS, n_jobs=1,
                     jupyter_mode=False, noise_amplitude=0.0)
    if OVERRIDE_COUPLING_RAMP:
        config = replace(config,
                         coupling_ramp_start_delays=COUPLING_RAMP_START_DELAYS,
                         coupling_ramp_rise_delays=COUPLING_RAMP_RISE_DELAYS)
    topology, active_fraction = _resolve_topology_input_override(
        config, TOPOLOGY_MATRIX_SOURCE_RECEIVER, TARGET_ACTIVE_FRACTION)
    if config.topology_input_mode and TARGET_ACTIVE_FRACTION is not None:
        print("Near-unity TARGET_ACTIVE_FRACTION selects an explicit all-to-all topology.")
    target = np.asarray(MANUAL_TARGET_PHASES_PI, dtype=float) * np.pi
    detunings = _detunings()
    if target.shape != (N_LASERS,) or detunings.shape != (N_LASERS,):
        raise ValueError("target phases and detunings must each have N_LASERS values")
    detunings = detunings - detunings.mean()
    inference = infer_designs(
        policy, target, detunings, config, count=candidate_count,
        active_fraction=active_fraction, seed=INFERENCE_RANDOM_SEED,
        continuous_mean=(ENSEMBLE_TOPOLOGY_ONLY if make_ensemble_plot
                         else not SAMPLE_CONTINUOUS_ACTIONS),
        sample_optional_gates=SAMPLE_OPTIONAL_GATES,
        topology=topology,
    )
    target, detunings = inference["target_phases_rad"], inference["detuning_distribution_ghz"]
    kappas, phis = inference["kappa_per_ns"], inference["phi_p_rad"]
    selected, dynamics, validation, precomputed = 0, None, None, None
    candidate_phase_rewards = None
    candidate_phase_errors_deg = None
    ensemble_summary = None
    if RUN_VALIDATION:
        validation_config = replace(config,
                                    simulation_time_seconds=VALIDATION_TIME_NS * 1e-9,
                                    noise_amplitude=NOISE_AMPLITUDE)
        np.random.seed(NOISE_RANDOM_SEED)
        case_kappa = np.repeat(kappas, NOISE_ITERATIONS, axis=0)
        case_phi = np.repeat(phis, NOISE_ITERATIONS, axis=0)
        sparse_case_count = len(case_kappa)
        if config.dense_reference_feasibility:
            dense_kappa, dense_phi = dense_reference_designs(
                policy, target[None], detunings[None], validation_config,
            )
            case_kappa = np.concatenate((case_kappa,
                                         np.repeat(dense_kappa[0], NOISE_ITERATIONS, axis=0)))
            case_phi = np.concatenate((case_phi,
                                       np.repeat(dense_phi[0], NOISE_ITERATIONS, axis=0)))
        simulations = simulate_with_vcsel(
            case_kappa, case_phi, validation_config, progress=True,
            smooth_frequencies=True,
            detuning_distribution_ghz=np.repeat(detunings[None], len(case_kappa), axis=0),
        )
        phase, _, _, achieved, aligned, orientation = calculate_phase_reward(
            simulations[0], simulations[1],
            np.repeat(target[None], len(case_kappa), axis=0), validation_config,
        )
        if make_ensemble_plot:
            sparse_simulation = (
                simulations[0],
                *(part[:sparse_case_count] for part in simulations[1:]),
            )
            ensemble_summary = summarize_topology_trajectories(
                sparse_simulation, target, validation_config,
                n_topologies=candidate_count,
                noise_iterations=NOISE_ITERATIONS,
                orientations=orientation[:sparse_case_count],
                include_trajectories=ENSEMBLE_TRAJECTORY_MODE == "all",
            )
        phase_by_candidate = phase[:sparse_case_count].reshape(
            candidate_count, NOISE_ITERATIONS).mean(axis=1)
        aligned_by_candidate = aligned[:sparse_case_count].reshape(
            candidate_count, NOISE_ITERATIONS, N_LASERS)
        aligned_candidate_mean = np.angle(np.mean(np.exp(1j * aligned_by_candidate), axis=1))
        candidate_phase_rewards = phase_by_candidate
        candidate_phase_errors_deg = maximum_phase_error_deg(
            aligned_candidate_mean, designer.make_relative_to_laser_1(target))
        budget = relative_coupling_budget(kappas, config)
        reach = (root_reach_fractions(inference["gates"], N_LASERS)
                 if config.require_rooted_connectivity else
                 np.ones(candidate_count))
        if config.objective_mode == "lexicographic":
            errors = candidate_phase_errors_deg
            ranked, feasible = lexicographic_candidate_scores(
                phase_by_candidate[None], errors[None],
                inference["active_link_counts"][None], kappas.sum(axis=(-2, -1))[None],
                config, rooted=(reach == 1.0)[None])
            score, feasible = ranked[0], feasible[0]
            dense_reference_phase = None
        elif config.dense_reference_feasibility:
            dense_reference_phase = float(phase[sparse_case_count:].reshape(
                1, NOISE_ITERATIONS).mean(axis=1).max())
            score, feasible = dense_reference_reward(
                phase_by_candidate[None], np.array([[dense_reference_phase]]),
                inference["retained_link_fractions"][None],
                sparsity_bonus_weight=config.feasible_sparsity_bonus_weight,
            )
            score, feasible = score[0], feasible[0]
        elif config.objective_mode == "phase_only":
            dense_reference_phase = None
            feasible = None
            score = phase_by_candidate
        else:
            dense_reference_phase = None
            feasible = None
            score = reward_function(phase_by_candidate,
                                    inference["retained_link_fractions"], budget, config,
                                    root_reach_fraction=reach)
        if make_ensemble_plot:
            selected = select_ensemble_topology(
                phase_by_candidate, inference["active_link_counts"],
                ENSEMBLE_SELECTED_TOPOLOGY,
                valid_topologies=ensemble_summary["valid_topologies"],
                coupling_budgets=kappas.sum(axis=(-2, -1)),
            )
        else:
            selected = int(np.argmax(score))
        chosen = slice(selected * NOISE_ITERATIONS, (selected + 1) * NOISE_ITERATIONS)
        precomputed = (simulations[0], *(part[chosen] for part in simulations[1:]))
        aligned_mean = np.angle(np.mean(np.exp(1j * aligned[chosen]), axis=0))
        achieved_mean = np.angle(np.mean(np.exp(1j * achieved[chosen]), axis=0))
        error = np.angle(np.exp(1j * (aligned_mean - designer.make_relative_to_laser_1(target))))
        validation = dict(
            phase_reward=float(phase_by_candidate[selected]),
            ranking_reward=float(score[selected]),
            dense_reference_phase_reward=dense_reference_phase,
            dense_reference_feasible=(None if feasible is None else bool(feasible[selected])),
            achieved_phases_rad=achieved_mean,
            aligned_achieved_phases_rad=aligned_mean,
            orientation=float(orientation[chosen][0]),
            maximum_phase_error_deg=float(np.rad2deg(np.max(np.abs(error[1:])))),
        )
    kappa, phi = kappas[selected], phis[selected]
    receivers, sources = np.nonzero(kappa > 0)
    design = dict(
        rank=1, target_phases_rad=target,
        detuning_distribution_ghz=detunings,
        active_link_count=len(receivers),
        active_connection_fraction=len(receivers) / (N_LASERS * (N_LASERS - 1)),
        realized_rho=len(receivers) / (N_LASERS * (N_LASERS - 1)),
        kappa_per_ns=kappa, phi_p_rad=phi,
        receiver_indices=receivers, source_indices=sources,
    )
    if validation is not None:
        design.update(validation)
    print(f"Loaded {CHECKPOINT_KIND} checkpoint at iteration {metadata['iteration']}; "
          f"device={next(policy.parameters()).device}; M={N_LASERS}; "
          f"strength transform={config.strength_action_transform}; "
          f"uncapped redistribution={config.uncapped_redistribution}")
    if make_ensemble_plot:
        print(f"Selected {ENSEMBLE_SELECTED_TOPOLOGY} ensemble topology: "
              f"candidate {selected + 1}/{candidate_count}.")
    fraction_label = ("Input on-fraction" if config.topology_input_mode
                      else "Expected on-fraction")
    print(f"{fraction_label}={inference['expected_active_fraction']:.3f}; "
          f"sampled active links={len(receivers)}/{N_LASERS * (N_LASERS - 1)}; "
          f"total kappa={kappa.sum():.4g} ns^-1; "
          f"raw total={inference['raw_total_coupling_per_ns'][selected]:.4g} ns^-1; "
          f"unallocated={inference['redistribution_shortfall_per_ns'][selected]:.4g} ns^-1; "
          f"frozen M={N_LASERS} reference={reference_budget_for_m(config, N_LASERS):.4g} ns^-1; "
          f"relative budget={float(relative_coupling_budget(kappa, config)):.3f}")
    if PRINT_GATE_LOGITS and config.topology_input_mode:
        print("No gate logits: topology is an external policy input.")
    elif PRINT_GATE_LOGITS:
        edge_receivers, edge_sources = designer.directed_link_indices(N_LASERS)
        raw = inference["gate_logits"]
        natural = inference["natural_gate_probabilities"]
        adjusted = inference["optional_gate_probabilities"]
        tree = inference["tree_gates"]
        print(f"On/off head by directed link (zero-based laser indices; "
              f"inspection logit bias={inference['gate_logit_bias']:+.3f}):")
        for edge in range(min(len(raw), MAX_PRINTED_GATE_LOGITS)):
            required = tree is not None and bool(tree[selected, edge])
            print(f"  {edge_sources[edge]} -> {edge_receivers[edge]}: "
                  f"logit={raw[edge]:+.4f}, natural p={natural[edge]:.4f}, "
                  f"adjusted p={adjusted[edge]:.4f}, "
                  f"tree={'yes' if required else 'no'}, "
                  f"active={'yes' if inference['gates'][selected, edge] else 'no'}")
        if len(raw) > MAX_PRINTED_GATE_LOGITS:
            print(f"  ... {len(raw) - MAX_PRINTED_GATE_LOGITS} more links omitted; "
                  "raise MAX_PRINTED_GATE_LOGITS to show them")
    if validation is not None:
        print(f"Phase reward={validation['phase_reward']:.4f}; "
              f"ranking reward={validation['ranking_reward']:.4f}; "
              f"maximum phase error={validation['maximum_phase_error_deg']:.3f} deg")
        if config.topology_input_mode and TOPOLOGY_MATRIX_SOURCE_RECEIVER is not None:
            passes_reward = candidate_phase_rewards >= 0.98
            passes_error = candidate_phase_errors_deg <= DISPLAY_PHASE_ERROR_DEG
            print(f"Fixed-topology test over {candidate_count} sampled actions: "
                  f"phase reward min/median/max="
                  f"{np.min(candidate_phase_rewards):.3f}/"
                  f"{np.median(candidate_phase_rewards):.3f}/"
                  f"{np.max(candidate_phase_rewards):.3f}; "
                  f"reward >= 0.98: {np.count_nonzero(passes_reward)}/{candidate_count}; "
                  f"max error <= {DISPLAY_PHASE_ERROR_DEG:g} deg: "
                  f"{np.count_nonzero(passes_error)}/{candidate_count}; "
                  f"both: {np.count_nonzero(passes_reward & passes_error)}/{candidate_count}")
        if config.dense_reference_feasibility:
            print(f"Current-policy all-on phase={dense_reference_phase:.4f}; "
                  f"sparse feasible={validation['dense_reference_feasible']}")
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    label = f"bernoulli_sparse_{FIGURE_SUFFIX}"
    selected_label = (f"{label}_{ENSEMBLE_SELECTED_TOPOLOGY}_topology"
                      if make_ensemble_plot else label)
    with plt.rc_context(PLOT_STYLE):
        if RUN_VALIDATION:
            figure = plot_selected_design(
                design, config, desired_state=selected_label, vertical_layout=False,
                kappa_vmax_per_ns=max(float(kappa.max()), 1e-12),
                topology_backbone_network=True,
                topology_backbone_edge_limit_factor=max(
                    design["active_link_count"] / N_LASERS, (N_LASERS - 1) / N_LASERS,
                ),
                topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                topology_edge_selection=NETWORK_EDGE_SELECTION,
                topology_sparsity_mask=False, topology_detuning_arc_layout=True,
                show_total_coupling_budget=True, disabled_links_black=True,
                allowable_rms_phase_error_deg=DISPLAY_PHASE_ERROR_DEG,
                allowable_phase_error_metric="maximum",
            )
            figure.savefig(FIGURE_DIRECTORY / f"selected_design_{selected_label}.png", dpi=200)
            plt.show()
            dynamics = simulate_and_plot_best_design(
                design, config,
                validation_delay_count=VALIDATION_TIME_NS / (config.delay_seconds * 1e9),
                desired_state=selected_label, detuning_distribution_ghz=detunings,
                legend_fontsize=PLOT_STYLE["legend.fontsize"],
                noise_iterations=NOISE_ITERATIONS, dual_axis_power_panel=True,
                far_field_panel=SHOW_FAR_FIELD,
                emitter_spacing_m=EMITTER_SPACING_UM * 1e-6,
                far_field_theta_range_deg=FAR_FIELD_THETA_RANGE_DEG,
                far_field_n_theta=FAR_FIELD_N_THETA,
                far_field_max_time_points=FAR_FIELD_MAX_TIME_POINTS,
                far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
                far_field_final_slice_panel=SHOW_FAR_FIELD_FINAL_SLICE,
                far_field_slice_convolution_fwhm_deg=FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG,
                precomputed_simulation=precomputed,
                allowable_phase_error_deg=DISPLAY_PHASE_ERROR_DEG,
            )
            if ensemble_summary is not None:
                print(f"Topology ensemble: {ensemble_summary['n_valid_topologies']}"
                      f"/{candidate_count} finite trajectories; "
                      + ("showing all candidate trajectories."
                         if ENSEMBLE_TRAJECTORY_MODE == "all" else
                         "bands are one standard deviation across topologies."))
                ensemble_figure = plot_topology_ensemble(
                    ensemble_summary, target,
                    trajectory_mode=ENSEMBLE_TRAJECTORY_MODE,
                    profile_smoothing_bins=ENSEMBLE_PROFILE_SMOOTHING_BINS)
                ensemble_figure.savefig(
                    FIGURE_DIRECTORY / f"topology_ensemble_{label}.png", dpi=200)
                plt.show()
        else:
            print("Inference only; no dynamics were simulated.")
            topology_design = {**design,
                               "kappa_per_ns": kappa[receivers, sources],
                               "phi_p_rad": phi[receivers, sources]}
            figure = plot_inference_summary(
                topology_design, kappa, phi, bins=TOPOLOGY_BINS,
                max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                edge_selection=NETWORK_EDGE_SELECTION,
            )
            figure.savefig(FIGURE_DIRECTORY / f"inferred_summary_{label}.png", dpi=350)
            plt.show()
    return dict(policy=policy, config=config, metadata=metadata,
                design=design, inference=inference, simulation=dynamics,
                selected_candidate_index=selected,
                ensemble_selection_mode=(ENSEMBLE_SELECTED_TOPOLOGY if make_ensemble_plot else None),
                topology_ensemble=ensemble_summary,
                candidate_phase_rewards=candidate_phase_rewards,
                candidate_phase_errors_deg=candidate_phase_errors_deg)


if __name__ == "__main__":
    bernoulli_sparse_inspection_results = main()

# %%
"""Check connectivity of the coupling matrix from the inspection above."""

import networkx as nx

# A link is present only when its strength exceeds this value. Leave at zero
# to test the exact Bernoulli on/off topology; raise it to test robustness to
# ignoring weak physical links. This cell does not modify the design.
CONNECTIVITY_MIN_KAPPA_PER_NS = 0.0


def inspect_coupling_connectivity(
    kappa_per_ns: np.ndarray, minimum_kappa_per_ns: float = 0.0
) -> dict:
    """Report directed connectivity; matrix [receiver, source] means source -> receiver."""
    kappa = np.asarray(kappa_per_ns, dtype=float)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError("kappa_per_ns must be a square matrix")
    if not np.all(np.isfinite(kappa)) or np.any(kappa < 0):
        raise ValueError("kappa_per_ns must contain finite, nonnegative strengths")
    if not np.isfinite(minimum_kappa_per_ns) or minimum_kappa_per_ns < 0:
        raise ValueError("minimum_kappa_per_ns must be finite and nonnegative")

    n_lasers = kappa.shape[0]
    active = kappa > minimum_kappa_per_ns
    np.fill_diagonal(active, False)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, n_lasers + 1))
    graph.add_edges_from(
        (int(source) + 1, int(receiver) + 1)
        for receiver, source in np.argwhere(active)
    )

    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    component_graph = nx.condensation(graph, strong_components)
    source_components = [
        component for component, degree in component_graph.in_degree() if degree == 0
    ]
    # A unique source SCC reaches the entire condensation DAG. Every laser in
    # that SCC is therefore a possible one-way source for the full network.
    source_lasers = (
        sorted(component_graph.nodes[source_components[0]]["members"])
        if len(source_components) == 1 else []
    )
    unilaterally_connected = nx.is_semiconnected(graph)

    def describe_components(components: list[set[int]]) -> str:
        if n_lasers <= 20:
            return str([sorted(component) for component in components])
        sizes = sorted((len(component) for component in components), reverse=True)
        return f"{len(components)} components; largest sizes {sizes[:10]}"

    print(f"Connectivity with kappa > {minimum_kappa_per_ns:g} ns^-1: "
          f"{graph.number_of_edges()}/{n_lasers * (n_lasers - 1)} directed links")
    print(f"  Weakly connected (ignore directions): {len(weak_components) == 1}")
    print(f"  Weak components: {describe_components(weak_components)}")
    print(f"  Rooted (one laser reaches every other): {bool(source_lasers)}")
    print(f"  Possible source lasers: {source_lasers[:20]}"
          + (f" ... ({len(source_lasers)} total)" if len(source_lasers) > 20 else ""))
    print(f"  Unilaterally connected (each pair reachable one way): "
          f"{unilaterally_connected}")
    print(f"  Strongly connected (every laser reaches every other): "
          f"{len(strong_components) == 1}")
    print(f"  Strong components: {describe_components(strong_components)}")

    return dict(
        weakly_connected=len(weak_components) == 1,
        rooted=bool(source_lasers),
        source_lasers=source_lasers,
        unilaterally_connected=unilaterally_connected,
        strongly_connected=len(strong_components) == 1,
        weak_components=[sorted(component) for component in weak_components],
        strong_components=[sorted(component) for component in strong_components],
        active_link_count=graph.number_of_edges(),
    )


if __name__ == "__main__":
    if "bernoulli_sparse_inspection_results" not in globals():
        raise RuntimeError("Run the main inspection cell before this connectivity cell")
    bernoulli_sparse_connectivity_results = inspect_coupling_connectivity(
        bernoulli_sparse_inspection_results["design"]["kappa_per_ns"],
        CONNECTIVITY_MIN_KAPPA_PER_NS,
    )


#%%

if __name__ == "__main__":
    if "bernoulli_sparse_inspection_results" not in globals():
        raise RuntimeError("Run the main inspection cell before ranking topologies")
    result = bernoulli_sparse_inspection_results
    ensemble = result["topology_ensemble"]
    valid_ids = np.flatnonzero(ensemble["valid_topologies"])

    # Rank by the phase-reward trajectory over its final 10%.
    traces = ensemble["reward_trajectories"]
    tail = max(1, traces.shape[1] // 10)
    scores = traces[:, -tail:].mean(axis=1)

    for position in np.argsort(scores)[:10]:
        candidate = int(valid_ids[position])
        links = result["inference"]["active_link_counts"][candidate]
        print(f"candidate {candidate}: reward ≈ {scores[position]:.3f}, links = {links}")

    candidate = int(valid_ids[np.argmin(scores)])
    failing_topology = (
        result["inference"]["kappa_per_ns"][candidate].T > 0
    ).astype(int)
    print("Failing topology (source rows, receiver columns):")
    print(failing_topology)


#%%
"""Optimize only link phases; hold the best inspected policy strengths fixed.

Run the main inspection cell first. This does not update the policy.
"""

DIRECT_SEARCH_POPULATION = 5
DIRECT_SEARCH_GENERATIONS = 2
DIRECT_SEARCH_SEED = 12345

if __name__ == "__main__":
    from IPython.display import clear_output, display
    from scipy.optimize import differential_evolution

    if "bernoulli_sparse_inspection_results" not in globals():
        raise RuntimeError("Run the main inspection cell before direct search")
    inspection = bernoulli_sparse_inspection_results
    if inspection["candidate_phase_rewards"] is None:
        raise RuntimeError("Run the main inspection cell with RUN_VALIDATION=True")
    if TOPOLOGY_MATRIX_SOURCE_RECEIVER is None:
        raise RuntimeError("Set a fixed TOPOLOGY_MATRIX_SOURCE_RECEIVER first")

    inferred = inspection["inference"]
    config = replace(inspection["config"],
                     simulation_time_seconds=VALIDATION_TIME_NS * 1e-9,
                     noise_amplitude=0.0)
    target = inferred["target_phases_rad"]
    detunings = inferred["detuning_distribution_ghz"]
    n_lasers = len(target)
    # Stored coupling matrices use receiver rows and source columns.
    receivers, sources = np.nonzero(inferred["kappa_per_ns"][0] > 0)
    n_links = len(receivers)
    if n_links == 0 or not np.all(
        (inferred["kappa_per_ns"] > 0) ==
        (inferred["kappa_per_ns"][0] > 0)
    ):
        raise RuntimeError("All inspected candidates must share one nonempty mask")

    policy_index = int(np.argmax(inspection["candidate_phase_rewards"]))
    fixed_kappa = inferred["kappa_per_ns"][policy_index].copy()
    bounds = [(-np.pi, np.pi)] * n_links
    policy_best_reward = float(inspection["candidate_phase_rewards"][policy_index])
    progress_evaluations = []
    progress_best_rewards = []

    def matrices(parameters):
        values = np.asarray(parameters)
        values = values[:, None] if values.ndim == 1 else values
        cases = values.shape[1]
        kappa = np.broadcast_to(fixed_kappa, (cases, n_lasers, n_lasers))
        phi = np.zeros_like(kappa)
        phi[:, receivers, sources] = values.T
        return kappa, phi

    def phase_loss(parameters):
        kappa, phi = matrices(parameters)
        cases = len(kappa)
        simulation = simulate_with_vcsel(
            kappa, phi, config,
            detuning_distribution_ghz=np.broadcast_to(
                detunings, (cases, n_lasers)),
        )
        reward = calculate_phase_reward(
            simulation[0], simulation[1],
            np.broadcast_to(target, (cases, n_lasers)), config,
        )[0]
        progress_evaluations.append(
            (progress_evaluations[-1] if progress_evaluations else 0) + cases)
        progress_best_rewards.append(max(
            progress_best_rewards[-1] if progress_best_rewards else -np.inf,
            float(np.max(reward)),
        ))
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.plot(progress_evaluations, progress_best_rewards, "o-",
                  color="tab:blue", label="best phase-only search reward")
        axis.axhline(policy_best_reward, color="tab:orange", ls="--",
                     label="best policy sample")
        axis.axhline(0.98, color="black", ls=":", label="0.98 reference")
        axis.set(xlabel="simulated designs", ylabel="phase reward",
                 ylim=(0, 1.02),
                 xlim=(0, DIRECT_SEARCH_POPULATION * (DIRECT_SEARCH_GENERATIONS + 1)))
        axis.grid(alpha=.25)
        axis.legend(loc="lower right")
        figure.tight_layout()
        clear_output(wait=True)
        display(figure)
        plt.close(figure)
        print(f"Evaluated {progress_evaluations[-1]} designs; "
              f"best phase-only reward={progress_best_rewards[-1]:.4f}")
        loss = np.where(np.isfinite(reward), 1.0 - reward, 2.0)
        return float(loss[0]) if np.asarray(parameters).ndim == 1 else loss

    rng = np.random.default_rng(DIRECT_SEARCH_SEED)
    initial = rng.uniform(-np.pi, np.pi, (DIRECT_SEARCH_POPULATION, n_links))
    # Include the actual policy design so the search cannot start below it.
    initial[0] = inferred["phi_p_rad"][policy_index, receivers, sources]

    search = differential_evolution(
        phase_loss, bounds, init=initial,
        maxiter=DIRECT_SEARCH_GENERATIONS, seed=DIRECT_SEARCH_SEED,
        vectorized=True, updating="deferred", polish=False,
    )
    best_kappa, best_phi = matrices(search.x)
    best_simulation = simulate_with_vcsel(
        best_kappa, best_phi, config,
        detuning_distribution_ghz=detunings[None],
    )
    best_reward, _, _, _, aligned, _ = calculate_phase_reward(
        best_simulation[0], best_simulation[1], target[None], config)
    best_error_deg = float(maximum_phase_error_deg(
        aligned, designer.make_relative_to_laser_1(target))[0])
    print(f"Fixed policy candidate {policy_index}: reward={policy_best_reward:.4f}")
    print(f"Phase-only search reward: {best_reward[0]:.4f}; "
          f"maximum phase error: {best_error_deg:.2f} deg; "
          f"total coupling: {best_kappa.sum():.3f} ns^-1")
    print("Fixed policy kappa (source rows, receiver columns):\n", best_kappa[0].T)
    print("Optimized phi_p (source rows, receiver columns):\n", best_phi[0].T)
    fixed_topology_search_results = dict(
        optimizer=search, kappa_per_ns=best_kappa[0], phi_p_rad=best_phi[0],
        phase_reward=float(best_reward[0]), maximum_phase_error_deg=best_error_deg,
        policy_candidate_index=policy_index,
        progress_evaluations=progress_evaluations,
        progress_best_rewards=progress_best_rewards,
    )


#%%
"""Simulate and plot the final direct-search design using the inspection layout."""

if __name__ == "__main__":
    if "fixed_topology_search_results" not in globals():
        raise RuntimeError("Run the direct-search cell before plotting its design")
    inspection = bernoulli_sparse_inspection_results
    optimized = fixed_topology_search_results
    target = inspection["inference"]["target_phases_rad"]
    detunings = inspection["inference"]["detuning_distribution_ghz"]
    config = replace(inspection["config"], noise_amplitude=0.0)
    validation_config = replace(
        config, simulation_time_seconds=VALIDATION_TIME_NS * 1e-9)
    kappa = optimized["kappa_per_ns"]
    phi = optimized["phi_p_rad"]
    simulation = simulate_with_vcsel(
        kappa[None], phi[None], validation_config,
        progress=True, smooth_frequencies=True,
        detuning_distribution_ghz=detunings[None],
    )
    phase_reward, _, _, _, aligned, _ = calculate_phase_reward(
        simulation[0], simulation[1], target[None], validation_config,
    )
    optimized_design = dict(
        kappa_per_ns=kappa, phi_p_rad=phi,
        target_phases_rad=target,
        detuning_distribution_ghz=detunings,
        aligned_achieved_phases_rad=aligned[0],
    )
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    label = f"direct_search_{FIGURE_SUFFIX}"
    with plt.rc_context(PLOT_STYLE):
        optimized_design_figure = plot_selected_design(
            optimized_design, config, desired_state=label,
            kappa_vmax_per_ns=max(float(kappa.max()), 1e-12),
            topology_backbone_network=True,
            topology_backbone_edge_limit_factor=max(
                np.count_nonzero(kappa) / len(target), (len(target) - 1) / len(target)),
            topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
            topology_edge_selection=NETWORK_EDGE_SELECTION,
            topology_detuning_arc_layout=True,
            show_total_coupling_budget=True, disabled_links_black=True,
            allowable_rms_phase_error_deg=DISPLAY_PHASE_ERROR_DEG,
            allowable_phase_error_metric="maximum",
        )
        plt.show()
        optimized_dynamics = simulate_and_plot_best_design(
            optimized_design, config,
            validation_delay_count=VALIDATION_TIME_NS / (config.delay_seconds * 1e9),
            desired_state=label, detuning_distribution_ghz=detunings,
            legend_fontsize=PLOT_STYLE["legend.fontsize"],
            dual_axis_power_panel=True, far_field_panel=SHOW_FAR_FIELD,
            emitter_spacing_m=EMITTER_SPACING_UM * 1e-6,
            far_field_theta_range_deg=FAR_FIELD_THETA_RANGE_DEG,
            far_field_n_theta=FAR_FIELD_N_THETA,
            far_field_max_time_points=FAR_FIELD_MAX_TIME_POINTS,
            far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
            far_field_final_slice_panel=SHOW_FAR_FIELD_FINAL_SLICE,
            far_field_slice_convolution_fwhm_deg=FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG,
            precomputed_simulation=simulation,
            allowable_phase_error_deg=DISPLAY_PHASE_ERROR_DEG,
        )
    print(f"Optimized design: phase reward={phase_reward[0]:.4f}, "
          f"total coupling={kappa.sum():.3f} ns^-1")


# %%
"""Compare policy and searched phases on the same fixed topology."""

if __name__ == "__main__":
    if ("bernoulli_sparse_inspection_results" not in globals()
            or "fixed_topology_search_results" not in globals()):
        raise RuntimeError("Run the main inspection and direct-search cells first")

    inspection = bernoulli_sparse_inspection_results
    inferred = inspection["inference"]
    optimized = fixed_topology_search_results
    rewards = inspection["candidate_phase_rewards"]
    if rewards is None:
        raise RuntimeError("Run the main inspection cell with RUN_VALIDATION=True")
    policy_index = int(np.argmax(rewards))
    policy_kappa = inferred["kappa_per_ns"][policy_index]
    policy_phi = inferred["phi_p_rad"][policy_index]
    optimized_kappa = optimized["kappa_per_ns"]
    optimized_phi = optimized["phi_p_rad"]
    if not np.array_equal(policy_kappa > 0, optimized_kappa > 0):
        raise RuntimeError("Policy and optimized designs must have the same active links")

    # Rows choose strengths; columns choose phases. All four cases share the
    # target, detunings, topology, simulation time, and noise-free conditions.
    strengths = (policy_kappa, optimized_kappa)
    phases = (policy_phi, optimized_phi)
    case_kappa = np.stack([strengths[row] for row in range(2) for _ in range(2)])
    case_phi = np.stack([phases[col] for _ in range(2) for col in range(2)])
    target = inferred["target_phases_rad"]
    detunings = inferred["detuning_distribution_ghz"]
    comparison_config = replace(
        inspection["config"], simulation_time_seconds=VALIDATION_TIME_NS * 1e-9,
        noise_amplitude=0.0,
    )
    simulation = simulate_with_vcsel(
        case_kappa, case_phi, comparison_config, progress=True,
        detuning_distribution_ghz=np.broadcast_to(detunings, (4, len(target))),
    )
    phase_reward, _, _, _, aligned, _ = calculate_phase_reward(
        simulation[0], simulation[1],
        np.broadcast_to(target, (4, len(target))), comparison_config,
    )
    error_deg = maximum_phase_error_deg(
        aligned, designer.make_relative_to_laser_1(target),
    )
    comparison_reward = phase_reward.reshape(2, 2)
    comparison_error_deg = error_deg.reshape(2, 2)
    same_strengths = np.array_equal(policy_kappa, optimized_kappa)
    if same_strengths:
        print("Phase-only search: both strength rows are identical by design.")
    strength_names = ("policy", "same fixed" if same_strengths else "optimized")
    for row, strength_name in enumerate(strength_names):
        for col, phase_name in enumerate(("policy", "optimized")):
            print(f"{strength_name:9s} strengths + {phase_name:9s} phases: "
                  f"reward={comparison_reward[row, col]:.4f}, "
                  f"max error={comparison_error_deg[row, col]:.2f} deg, "
                  f"total coupling={strengths[row].sum():.3f} ns^-1")

    figure, axis = plt.subplots(figsize=(5.5, 4))
    image = axis.imshow(comparison_reward, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks((0, 1), ("policy", "optimized"))
    axis.set_yticks((0, 1), ("policy", "same fixed" if same_strengths else "optimized"))
    axis.set(xlabel="coupling phases", ylabel="coupling strengths",
             title=("Phase-only search with fixed strengths" if same_strengths else
                    "Phase reward for strength/phase swaps"))
    for row in range(2):
        for col in range(2):
            value = comparison_reward[row, col]
            axis.text(col, row, f"{value:.3f}", ha="center", va="center",
                      color="black" if value > 0.7 else "white")
    figure.colorbar(image, ax=axis, label="phase reward")
    figure.tight_layout()
    plt.show()
    strength_phase_swap_results = dict(
        policy_candidate_index=policy_index, reward=comparison_reward,
        maximum_phase_error_deg=comparison_error_deg,
        simulation=simulation,
    )


# %%
"""Compare phase-only search results with the policy's phase Gaussians."""

if __name__ == "__main__":
    if ("bernoulli_sparse_inspection_results" not in globals()
            or "fixed_topology_search_results" not in globals()):
        raise RuntimeError("Run the main inspection and phase-only search cells first")

    inspection = bernoulli_sparse_inspection_results
    inferred = inspection["inference"]
    optimized = fixed_topology_search_results
    if "policy_candidate_index" not in optimized:
        raise RuntimeError("Rerun the phase-only search cell with fixed policy strengths")
    policy_index = optimized["policy_candidate_index"]
    policy_kappa = inferred["kappa_per_ns"][policy_index]
    if not np.array_equal(policy_kappa, optimized["kappa_per_ns"]):
        raise RuntimeError("The search result does not match this inspection's policy design")

    policy = inspection["policy"]
    if not policy.config.topology_input_mode:
        raise RuntimeError("This diagnostic requires a topology-input policy")
    target = inferred["target_phases_rad"]
    detunings = inferred["detuning_distribution_ghz"]
    features = encode_context(target, detunings, policy)
    mask = torch.as_tensor(inferred["gates"][policy_index][None],
                           dtype=features.dtype, device=features.device)
    checkpoint_iteration = int(inspection["metadata"]["iteration"])
    if checkpoint_iteration < 1:
        raise RuntimeError("Checkpoint has no completed training iteration")
    # Checkpoints save a one-based count; the training batch used its zero-based
    # index when it set the scheduled log-SD bounds.
    training_iteration = checkpoint_iteration - 1
    minimum_log_std = designer.minimum_log_std_at(training_iteration, inspection["config"])
    maximum_log_std = designer.maximum_log_std_at(training_iteration, inspection["config"])
    previous_bounds = (policy.minimum_log_std, policy.maximum_log_std)
    policy.eval()
    try:
        policy.minimum_log_std = minimum_log_std
        policy.maximum_log_std = maximum_log_std
        with torch.inference_mode():
            action_mean, action_log_std, _ = policy.parameters_for_edges(
                features, topology_masks=mask)
    finally:
        policy.minimum_log_std, policy.maximum_log_std = previous_bounds

    n_lasers = len(target)
    n_edges = n_lasers * (n_lasers - 1)
    receivers, sources = designer.directed_link_indices(n_lasers)
    active = mask[0].cpu().numpy().astype(bool)
    phase_mean = np.angle(np.exp(1j * action_mean[0, n_edges:].cpu().numpy()))
    phase_std = action_log_std[0, n_edges:].exp().cpu().numpy()
    optimized_phase = optimized["phi_p_rad"][receivers, sources]
    policy_phase = inferred["phi_p_rad"][policy_index, receivers, sources]
    # Phases are wrapped after Gaussian sampling, so use the shortest circular
    # distance to the mean, including near the -pi/pi boundary.
    phase_delta = np.angle(np.exp(1j * (optimized_phase - phase_mean)))
    distance_in_sigma = np.abs(phase_delta) / phase_std
    if not np.array_equal(active, policy_kappa[receivers, sources] > 0):
        raise RuntimeError("Policy gate mask and coupling matrix disagree")

    active_ids = np.flatnonzero(active)
    labels = [f"{sources[i] + 1} → {receivers[i] + 1}" for i in active_ids]
    print(f"Checkpoint iteration {checkpoint_iteration}: training phase-SD bounds "
          f"{np.exp(minimum_log_std):.3f}–{np.exp(maximum_log_std):.3f} rad")
    print("Active link | policy mean | bounded SD | inspect sample | optimized | "
          "circular gap | gap / SD")
    print("            (all angles in degrees)")
    for i, label in zip(active_ids, labels):
        print(f"{label:>11s} | {np.degrees(phase_mean[i]):11.1f} | "
              f"{np.degrees(phase_std[i]):10.1f} | "
              f"{np.degrees(policy_phase[i]):13.1f} | "
              f"{np.degrees(optimized_phase[i]):9.1f} | "
              f"{np.degrees(phase_delta[i]):12.1f} | "
              f"{distance_in_sigma[i]:8.2f}")
    active_distances = distance_in_sigma[active]
    print(f"Median/max circular distance: {np.median(active_distances):.2f}/"
          f"{active_distances.max():.2f} training-time SD; "
          f">2 SD: {np.count_nonzero(active_distances > 2)}/{len(active_distances)}, "
          f">3 SD: {np.count_nonzero(active_distances > 3)}/{len(active_distances)}")
    print("These are per-link nearest-wrap distances, not joint probabilities.")
    print("The inspect sample was drawn earlier; this cell corrects the reported SD only.")

    figure, axis = plt.subplots(figsize=(7, max(3, 0.38 * len(active_ids) + 1.5)))
    axis.barh(labels, active_distances, color="tab:blue")
    for threshold in (1, 2, 3):
        axis.axvline(threshold, color="gray", ls="--", alpha=0.6)
    axis.invert_yaxis()
    axis.set(xlabel="optimized phase distance from policy mean / training-time SD",
             title="Phase-only solution versus policy exploration")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    plt.show()
    phase_exploration_diagnostic = dict(
        policy_candidate_index=policy_index,
        checkpoint_iteration=checkpoint_iteration,
        training_log_std_bounds=(minimum_log_std, maximum_log_std),
        active_links_source_receiver=np.column_stack((sources[active], receivers[active])),
        policy_phase_mean_rad=phase_mean[active],
        policy_phase_std_rad=phase_std[active],
        optimized_phase_rad=optimized_phase[active],
        circular_difference_rad=phase_delta[active],
        distance_in_std=active_distances,
    )


# %%
"""Test broad phase exploration without DE, keeping policy strengths fixed."""

RANDOM_PHASE_SAMPLES = 512
RANDOM_PHASE_BATCH_SIZE = 64
RANDOM_PHASE_SEED = 12345

if __name__ == "__main__":
    if "bernoulli_sparse_inspection_results" not in globals():
        raise RuntimeError("Run the main inspection cell before this test")
    inspection = bernoulli_sparse_inspection_results
    rewards = inspection["candidate_phase_rewards"]
    if rewards is None:
        raise RuntimeError("Run the main inspection cell with RUN_VALIDATION=True")
    if RANDOM_PHASE_SAMPLES < 1 or RANDOM_PHASE_BATCH_SIZE < 1:
        raise ValueError("Random phase sample and batch counts must be positive")

    inferred = inspection["inference"]
    policy_index = int(np.argmax(rewards))
    fixed_kappa = inferred["kappa_per_ns"][policy_index].copy()
    policy_phi = inferred["phi_p_rad"][policy_index].copy()
    receivers, sources = np.nonzero(fixed_kappa > 0)
    if len(receivers) == 0:
        raise RuntimeError("The selected policy design has no active links")

    target = inferred["target_phases_rad"]
    detunings = inferred["detuning_distribution_ghz"]
    n_lasers = len(target)
    test_config = replace(
        inspection["config"], simulation_time_seconds=VALIDATION_TIME_NS * 1e-9,
        noise_amplitude=0.0,
    )

    def evaluate_phases(phase_matrices):
        cases = len(phase_matrices)
        simulation = simulate_with_vcsel(
            np.repeat(fixed_kappa[None], cases, axis=0), phase_matrices,
            test_config,
            detuning_distribution_ghz=np.broadcast_to(detunings, (cases, n_lasers)),
        )
        return calculate_phase_reward(
            simulation[0], simulation[1],
            np.broadcast_to(target, (cases, n_lasers)), test_config,
        )[0]

    policy_reward = float(evaluate_phases(policy_phi[None])[0])
    rng = np.random.default_rng(RANDOM_PHASE_SEED)
    all_rewards = []
    best_reward = -np.inf
    best_phi = None
    progress_counts = []
    progress_best = []
    for start in range(0, RANDOM_PHASE_SAMPLES, RANDOM_PHASE_BATCH_SIZE):
        cases = min(RANDOM_PHASE_BATCH_SIZE, RANDOM_PHASE_SAMPLES - start)
        phases = np.zeros((cases, n_lasers, n_lasers))
        phases[:, receivers, sources] = rng.uniform(
            -np.pi, np.pi, size=(cases, len(receivers)))
        batch_rewards = evaluate_phases(phases)
        batch_scores = np.where(np.isfinite(batch_rewards), batch_rewards, -np.inf)
        batch_best = int(np.argmax(batch_scores))
        if batch_scores[batch_best] > best_reward:
            best_reward = float(batch_scores[batch_best])
            best_phi = phases[batch_best].copy()
        all_rewards.extend(batch_rewards)
        progress_counts.append(start + cases)
        progress_best.append(best_reward)

    all_rewards = np.asarray(all_rewards)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(progress_counts, progress_best, "o-", label="best uniform-phase sample")
    axis.axhline(policy_reward, color="tab:orange", ls="--",
                 label="selected policy sample")
    axis.axhline(0.98, color="black", ls=":", label="0.98 reference")
    axis.set(xlabel="random phase designs simulated", ylabel="phase reward",
             xlim=(0, RANDOM_PHASE_SAMPLES), ylim=(0, 1.02),
             title="Fixed policy strengths, uniformly sampled link phases")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    plt.show()
    print(f"Fixed policy candidate {policy_index}: reward={policy_reward:.4f}; "
          f"best of {RANDOM_PHASE_SAMPLES} uniform-phase samples={best_reward:.4f}; "
          f"samples at or above 0.98={np.count_nonzero(all_rewards >= 0.98)}")
    random_phase_exploration_results = dict(
        policy_candidate_index=policy_index, fixed_kappa_per_ns=fixed_kappa,
        policy_phase_reward=policy_reward, sampled_phase_rewards=all_rewards,
        best_phase_reward=best_reward, best_phi_p_rad=best_phi,
        progress_counts=progress_counts, progress_best_rewards=progress_best,
    )
