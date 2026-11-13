#%%
"""Inspect the selector-free tolerance-conditioned sparse GNN policy."""

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_designer_variable_m as designer
from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.archive import conditional_coupling_sparse_tolerance as sparse_policy
from rl.archive.conditional_coupling_sparse_tolerance import (
    encode_sparse_tolerance_context,
    load_sparse_checkpoint,
    q_to_active_link_counts,
)
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design,
    simulate_and_plot_best_design,
)
from rl.archive.inspect_saved_coupling_model_variable_m_learned_tree_scale_aware import (
    plot_inference_summary,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "current"  # "current", "best", or "final"
TRAINING_N_LASERS = tuple(range(3, 11))
CHECKPOINT_SIZE_LABEL = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
MODEL_SUFFIX = "_gnn_sparse_tol_scale_aware_span4ghz"


# "_gnn_sparse_tol_scale_aware_span4ghz"
N_LASERS = 3

N_CLUSTERS = 1
ALLOWABLE_MAX_PHASE_ERROR_DEG = 60.0
DETUNING_SPAN_GHZ = 0.1
DETUNING_MODE = 'manual'


MANUAL_DETUNINGS_GHZ = np.array([ 1.5, -0.5, -1.5])
DETUNING_DISTRIBUTION = {
    'equal': np.linspace(-0.5, 0.5, N_LASERS),
    'random': np.sort(np.random.uniform(-0.5, 0.5, N_LASERS)),
    'manual': MANUAL_DETUNINGS_GHZ/DETUNING_SPAN_GHZ
}


# The automatic mode uses the deterministic mean of the learned q policy.
# Manual mode bypasses that one output while retaining the same GNN edge
# policy, connected pruning, and simulation.
Q_MODE = "automatic"  # "automatic" or "manual"
MANUAL_Q = 3.0/9.0
NUMBER_OF_CANDIDATES = 1
INCLUDE_DETERMINISTIC_CANDIDATE = True

# Optional inspection-only symmetry operation.
MIRROR_IF_LOWER_TRIANGLE_DISABLED = False
# In inference-only mode this caps all drawn network edges; omitted links are
# not shown as gray context.  The validation plot uses the same overlay cap.
NETWORK_MAX_DISPLAY_EDGES = 5000
RUN_VALIDATION = True
TOPOLOGY_BINS = N_LASERS

VALIDATION_TIME_NS = 500.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None

OVERRIDE_COUPLING_RAMP = True
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0

SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-5.0,5.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 5.0
SHOW_FAR_FIELD_FINAL_SLICE = True
# Display-only Gaussian angular blur, applied to every far-field time slice.
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = None

FIGURE_DIRECTORY = RESULTS_DIR / "inference" / "sparse_tolerance_original"
FIGURE_SUFFIX = (
    f"{N_LASERS}_{N_CLUSTERS}_clusters_tol{ALLOWABLE_MAX_PHASE_ERROR_DEG:g}_"
    f"q{MANUAL_Q:.1f}".replace(".", "p")
)

cluster_phases_pi = 2.0 * np.arange(N_CLUSTERS) / N_CLUSTERS
base_count, remainder = divmod(N_LASERS, N_CLUSTERS)
cluster_counts = [
    base_count + (cluster < remainder) for cluster in range(N_CLUSTERS)
]
MANUAL_TARGET_PHASES_PI = np.repeat(
    cluster_phases_pi, cluster_counts
).tolist()
# Inspect the original run at its full trained 4-GHz peak-to-peak span.

MANUAL_DETUNINGS_GHZ = DETUNING_SPAN_GHZ*DETUNING_DISTRIBUTION[DETUNING_MODE]

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


def _mirror_dominant_triangle(kappa, phi):
    """Mirror one candidate before its only simulation."""
    if np.sum(np.triu(kappa, 1)) >= np.sum(np.tril(kappa, -1)):
        triangle_kappa = np.triu(kappa, 1)
        triangle_phi = np.triu(phi, 1)
    else:
        triangle_kappa = np.tril(kappa, -1)
        triangle_phi = np.tril(phi, -1)
    return triangle_kappa + triangle_kappa.T, triangle_phi + triangle_phi.T


def _infer_candidates(policy, target, detunings, config, q_override):
    """Reuse the trained policy's candidate rules without running physics."""
    target, detunings = designer.sort_by_detuning(target, detunings)
    features = encode_sparse_tolerance_context(
        target, detunings, ALLOWABLE_MAX_PHASE_ERROR_DEG, policy, config
    )
    policy.eval()
    with torch.no_grad():
        if q_override is not None:
            q = torch.full(
                (1, NUMBER_OF_CANDIDATES), float(q_override),
                device=features.device, dtype=features.dtype,
            )
            repeated = sparse_policy._repeat_contexts(features, NUMBER_OF_CANDIDATES)
            actions = policy.action_mean(repeated, q.reshape(-1)).reshape(
                1, NUMBER_OF_CANDIDATES, -1
            )
        elif NUMBER_OF_CANDIDATES == 1:
            q_mean = torch.sigmoid(policy.density_distribution(features).mean[:, 0])
            q = q_mean[:, None]
            actions = policy.action_mean(features, q_mean)[:, None, :]
        else:
            actions, q, _ = sparse_policy.sample_joint_sparse_candidates(
                policy, features, NUMBER_OF_CANDIDATES, 0.0
            )
        if INCLUDE_DETERMINISTIC_CANDIDATE and NUMBER_OF_CANDIDATES > 1:
            q_mean = torch.sigmoid(policy.density_distribution(features).mean[:, 0])
            q[0, 0] = q_mean[0]
            actions[0, 0] = policy.action_mean(features, q_mean)[0]
    q_values = q.detach().cpu().numpy()[0]
    kappa, phi, _, _ = sparse_policy._decode_and_prune(
        actions, q.detach().cpu().numpy(), config
    )
    kappa, phi = kappa[0], phi[0]
    if MIRROR_IF_LOWER_TRIANGLE_DISABLED:
        for index in range(len(kappa)):
            kappa[index], phi[index] = _mirror_dominant_triangle(
                kappa[index], phi[index]
            )
    return target, detunings, q_values, kappa, phi


def main() -> dict:
    if CHECKPOINT_KIND not in {"current", "best", "final"}:
        raise ValueError('CHECKPOINT_KIND must be "current", "best", or "final"')
    checkpoint = MODEL_DIR / (
        "conditional_coupling_variable_m_"
        f"{CHECKPOINT_SIZE_LABEL}_lasers_{CHECKPOINT_KIND}"
        f"{MODEL_SUFFIX}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} does not exist. Start the sparse training runner first."
        )
    policy, _, config, metadata = load_sparse_checkpoint(
        checkpoint, device="cpu"
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

    target = np.asarray(MANUAL_TARGET_PHASES_PI, dtype=float) * np.pi
    detunings = np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    if target.shape != (N_LASERS,) or detunings.shape != (N_LASERS,):
        raise ValueError("manual targets and detunings must each have length N_LASERS")
    detunings = detunings - np.mean(detunings)
    if Q_MODE == "automatic":
        q_override = None
    elif Q_MODE == "manual":
        if not 0.0 <= MANUAL_Q <= 1.0:
            raise ValueError("MANUAL_Q must lie in [0, 1]")
        q_override = MANUAL_Q
    else:
        raise ValueError('Q_MODE must be "automatic" or "manual"')

    if NUMBER_OF_CANDIDATES < 1:
        raise ValueError("NUMBER_OF_CANDIDATES must be at least 1")
    target, detunings, q_values, kappa, phi = _infer_candidates(
        policy, target, detunings, config, q_override
    )

    if not RUN_VALIDATION:
        # With no physical ranking, show the deterministic first candidate.
        selected = 0
        receivers, sources = np.nonzero(kappa[selected] > 0.0)
        design = dict(
            target_phases_rad=target,
            detuning_distribution_ghz=detunings,
            q=float(q_values[selected]),
            kappa_per_ns=kappa[selected],
            phi_p_rad=phi[selected],
            receiver_indices=receivers,
            source_indices=sources,
        )
        topology_design = {
            **design,
            "kappa_per_ns": kappa[selected][receivers, sources],
            "phi_p_rad": phi[selected][receivers, sources],
        }
        print(
            f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
            f"{metadata['iteration']}; inference only (no validation simulation)."
        )
        print(
            f"Predicted q: {design['q']:.4f}; active directed links: "
            f"{len(receivers)}/{N_LASERS * (N_LASERS - 1)}"
        )
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        suffix = f"_{FIGURE_SUFFIX}" if FIGURE_SUFFIX else ""
        figure_label = f"sparse_tolerance{suffix}"
        with plt.rc_context(PLOT_STYLE):
            figure = plot_inference_summary(
                topology_design, kappa[selected], phi[selected],
                bins=TOPOLOGY_BINS,
                max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
            )
            figure.savefig(
                FIGURE_DIRECTORY / f"inferred_summary_{figure_label}.png",
                dpi=350,
            )
            plt.show()
        return dict(
            policy=policy, config=config, metadata=metadata,
            design=design, simulation=None,
        )

    # Simulate every candidate in one validation batch, then select the best.
    # The selected trajectory is reused by the dynamics plot below.
    validation_config = replace(
        config,
        simulation_time_seconds=VALIDATION_TIME_NS * 1.0e-9,
        noise_amplitude=NOISE_AMPLITUDE,
        n_jobs=1,
    )
    np.random.seed(NOISE_RANDOM_SEED)
    all_simulations = designer.simulate_with_vcsel(
        np.repeat(kappa, NOISE_ITERATIONS, axis=0),
        np.repeat(phi, NOISE_ITERATIONS, axis=0),
        validation_config,
        progress=True,
        smooth_frequencies=True,
        detuning_distribution_ghz=np.repeat(
            detunings[None, :],
            len(kappa) * NOISE_ITERATIONS,
            axis=0,
        ),
    )
    phase_reward, _, _, achieved, aligned, orientation = (
        designer.calculate_phase_reward(
            all_simulations[0],
            all_simulations[1],
            np.repeat(target[None, :], len(kappa) * NOISE_ITERATIONS, axis=0),
            validation_config,
        )
    )
    case_errors = sparse_policy.maximum_circular_phase_error_deg(
        np.repeat(target[None, :], len(aligned), axis=0), aligned[:, None, :]
    )[:, 0]
    q_cases = np.repeat(q_values, NOISE_ITERATIONS)
    candidate_rewards, _ = sparse_policy.tiered_sparse_reward(
        phase_reward[:, None], case_errors[:, None],
        np.full(len(q_cases), ALLOWABLE_MAX_PHASE_ERROR_DEG),
        q_cases[:, None], config.infeasible_reward_tau_deg,
    )
    candidate_scores = candidate_rewards.reshape(
        len(kappa), NOISE_ITERATIONS
    ).mean(axis=1)
    selected = int(np.argmax(candidate_scores))
    chosen_cases = slice(
        selected * NOISE_ITERATIONS, (selected + 1) * NOISE_ITERATIONS
    )
    precomputed_simulation = (
        all_simulations[0],
        *(part[chosen_cases] for part in all_simulations[1:]),
    )
    aligned_mean = np.angle(np.mean(np.exp(1j * aligned[chosen_cases]), axis=0))
    target_relative = designer.make_relative_to_laser_1(target)
    phase_error = np.angle(np.exp(1j * (aligned_mean - target_relative)))
    active_links = int(np.count_nonzero(kappa[selected] > 0.0))
    maximum_links = N_LASERS * (N_LASERS - 1)
    design = dict(
        rank=1,
        target_phases_rad=target,
        detuning_distribution_ghz=detunings,
        allowable_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
        q=float(q_values[selected]),
        predicted_q=float(q_values[selected]),
        active_link_count=active_links,
        active_connection_fraction=active_links / maximum_links,
        realized_rho=active_links / maximum_links,
        kappa_per_ns=kappa[selected],
        phi_p_rad=phi[selected],
        phase_reward=float(np.mean(phase_reward[chosen_cases])),
        ranking_reward=float(candidate_scores[selected]),
        achieved_phases_rad=np.angle(
            np.mean(np.exp(1j * achieved[chosen_cases]), axis=0)
        ),
        aligned_achieved_phases_rad=aligned_mean,
        orientation=float(orientation[chosen_cases][0]),
    )
    design["maximum_phase_error_deg"] = float(
        np.rad2deg(np.max(np.abs(phase_error[1:])))
    )
    design["constraint_satisfied"] = bool(
        design["maximum_phase_error_deg"] <= ALLOWABLE_MAX_PHASE_ERROR_DEG
    )

    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
        f"{metadata['iteration']} (dense held-out={metadata['held_out_reward']:.3f})"
    )
    print(f"Training sizes: {TRAINING_N_LASERS}; evaluating M={N_LASERS}")
    print(
        "Scale-aware detuning features: "
        f"{config.scale_aware_detuning_features}"
    )
    print(f"Allowable maximum phase error: {ALLOWABLE_MAX_PHASE_ERROR_DEG:.3f} deg")
    print(f"Predicted q: {design['q']:.4f}")
    print(
        f"Active directed links: {design['active_link_count']}/{maximum_links} "
        f"(rho={design['active_connection_fraction']:.4f})"
    )
    print(f"Phase reward: {design['phase_reward']:.4f}")
    print(
        "Maximum relative phase error: "
        f"{design['maximum_phase_error_deg']:.3f} deg"
    )
    print(
        "Maximum-error constraint: "
        + ("satisfied" if design["constraint_satisfied"] else "violated")
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    suffix = f"_{FIGURE_SUFFIX}" if FIGURE_SUFFIX else ""
    figure_label = f"sparse_tolerance{suffix}"
    topology_edge_limit_factor = max(
        design["active_link_count"] / N_LASERS,
        (N_LASERS - 1) / N_LASERS,
    )
    plotted_kappa_max = max(float(np.max(design["kappa_per_ns"])), 1.0e-12)

    # Both figures now use the same completed validation trajectory.
    with plt.rc_context(PLOT_STYLE):
        plot_selected_design(
            design,
            config,
            desired_state=figure_label,
            vertical_layout=False,
            kappa_vmax_per_ns=plotted_kappa_max,
            topology_backbone_network=True,
            topology_backbone_edge_limit_factor=topology_edge_limit_factor,
            topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
            topology_sparsity_mask=False,
            topology_detuning_arc_layout=True,
            show_total_coupling_budget=True,
            disabled_links_black=True,
            allowable_rms_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
            allowable_phase_error_metric="maximum",
        )
        plt.show()

        simulation = simulate_and_plot_best_design(
            design,
            replace(config, noise_amplitude=NOISE_AMPLITUDE),
            validation_delay_count=(
                VALIDATION_TIME_NS / (config.delay_seconds * 1.0e9)
            ),
            desired_state=figure_label,
            detuning_distribution_ghz=design["detuning_distribution_ghz"],
            legend_fontsize=PLOT_STYLE["legend.fontsize"],
            noise_iterations=NOISE_ITERATIONS,
            dual_axis_power_panel=True,
            far_field_panel=SHOW_FAR_FIELD,
            emitter_spacing_m=EMITTER_SPACING_UM * 1.0e-6,
            far_field_theta_range_deg=FAR_FIELD_THETA_RANGE_DEG,
            far_field_n_theta=FAR_FIELD_N_THETA,
            far_field_max_time_points=FAR_FIELD_MAX_TIME_POINTS,
            far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
            far_field_final_slice_panel=SHOW_FAR_FIELD_FINAL_SLICE,
            far_field_slice_convolution_fwhm_deg=(
                FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG
            ),
            precomputed_simulation=precomputed_simulation,
            allowable_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
        )
    return {
        "policy": policy,
        "config": config,
        "metadata": metadata,
        "design": design,
        "simulation": simulation,
    }


# Run directly as a notebook-style inspection cell.
results = main()

#%% Inspect deterministic q versus allowable error without additional simulation
ERROR_SWEEP_DEG = np.asarray([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0])
print("\nDeterministic density-policy sweep (no VCSEL simulations)")
print("error (deg) |    q    | links |  rho")
print("------------+---------+-------+-------")
for tolerance_deg in ERROR_SWEEP_DEG:
    target = np.asarray(results["design"]["target_phases_rad"])
    detunings = np.asarray(results["design"]["detuning_distribution_ghz"])
    features = encode_sparse_tolerance_context(
        target,
        detunings,
        tolerance_deg,
        results["policy"],
        results["config"],
    )
    with torch.no_grad():
        q_value = float(
            torch.sigmoid(
                results["policy"].density_distribution(features).mean[0, 0]
            )
        )
    links = int(q_to_active_link_counts(q_value, N_LASERS))
    print(
        f"{tolerance_deg:11.3f} | {q_value:7.4f} | "
        f"{links:5d} | {links / (N_LASERS * (N_LASERS - 1)):5.3f}"
    )










#%% Evaluate passive optical realizability of the current coupling matrices
# These values define the LK-rate-to-field-transmission calibration.  Change
# them to match the effective output/coupling mirror in the physical device.
PASSIVITY_INTERNAL_REFLECTIVITY = 0.999
PASSIVITY_ROUND_TRIP_TIME_PS = 0.08
PASSIVITY_NUMERICAL_TOLERANCE = 1.0e-10

if not 0.0 < PASSIVITY_INTERNAL_REFLECTIVITY < 1.0:
    raise ValueError("PASSIVITY_INTERNAL_REFLECTIVITY must lie in (0, 1)")
if PASSIVITY_ROUND_TRIP_TIME_PS <= 0.0:
    raise ValueError("PASSIVITY_ROUND_TRIP_TIME_PS must be positive")

# Matrix convention: kappa[receiver, source].  Since kappa is stored in
# ns^-1 and tau_rt is entered in ps, their dimensionless product contains
# the conversion factor 10^-3.
passivity_kappa_per_ns = np.asarray(
    results["design"]["kappa_per_ns"], dtype=float
)
passivity_phi_p_rad = np.asarray(
    results["design"]["phi_p_rad"], dtype=float
)
if not np.all(np.isfinite(passivity_kappa_per_ns)):
    raise ValueError("The inspected kappa matrix contains NaN or infinity")
if np.any(passivity_kappa_per_ns < 0.0):
    raise ValueError("The inspected kappa matrix contains negative rates")
if not np.all(np.isfinite(passivity_phi_p_rad)):
    raise ValueError("The inspected phi_p matrix contains NaN or infinity")
passivity_field_magnitude = (
    passivity_kappa_per_ns
    * PASSIVITY_ROUND_TRIP_TIME_PS
    * 1.0e-3
    * np.sqrt(PASSIVITY_INTERNAL_REFLECTIVITY)
    / (1.0 - PASSIVITY_INTERNAL_REFLECTIVITY)
)
passivity_transfer_matrix = passivity_field_magnitude * np.exp(
    -1.0j * passivity_phi_p_rad
)
np.fill_diagonal(passivity_transfer_matrix, 0.0)
if not np.all(np.isfinite(passivity_transfer_matrix)):
    raise ValueError(
        "The LK-to-transmission conversion produced NaN or infinity. "
        "Check R, tau_rt, and the coupling-rate units."
    )

# |T_ij|^2 is the power fraction sent from source j to receiver i.  Column
# sums therefore give the power fraction committed by each source laser when
# that source is excited alone.  Scale T before squaring or taking its SVD so
# even an obviously nonpassive matrix is diagnosed without overflow warnings.
passivity_matrix_scale = float(np.max(np.abs(passivity_transfer_matrix)))
if passivity_matrix_scale == 0.0:
    passivity_scaled_transfer = passivity_transfer_matrix.copy()
    passivity_scaled_source_power = np.zeros(N_LASERS, dtype=float)
    passivity_scaled_sigma_max = 0.0
else:
    passivity_scaled_transfer = (
        passivity_transfer_matrix / passivity_matrix_scale
    )
    passivity_scaled_source_power = np.sum(
        np.abs(passivity_scaled_transfer) ** 2,
        axis=0,
    )
    passivity_scaled_sigma_max = float(
        np.linalg.svd(passivity_scaled_transfer, compute_uv=False)[0]
    )

passivity_log_float_max = float(np.log(np.finfo(float).max))


def _scaled_square_to_float(scale: float, scaled_value: float) -> float:
    """Return scale**2 * scaled_value without emitting overflow warnings."""
    if scale == 0.0 or scaled_value == 0.0:
        return 0.0
    log_value = 2.0 * np.log(scale) + np.log(scaled_value)
    if log_value > passivity_log_float_max:
        return float("inf")
    return float(np.exp(log_value))


passivity_source_fractions = np.asarray(
    [
        _scaled_square_to_float(
            passivity_matrix_scale,
            float(scaled_source_power),
        )
        for scaled_source_power in passivity_scaled_source_power
    ]
)

# A passive coherent multiport must be a contraction: T^dagger T <= I.
# lambda_max(T^dagger T) = sigma_max(T)^2, so the SVD gives the same test
# without explicitly forming a potentially overflowing matrix product.
passivity_largest_gram_eigenvalue = _scaled_square_to_float(
    passivity_matrix_scale,
    passivity_scaled_sigma_max**2,
)
if passivity_matrix_scale == 0.0:
    passivity_largest_singular_value = 0.0
else:
    passivity_log_sigma_max = (
        np.log(passivity_matrix_scale)
        + np.log(passivity_scaled_sigma_max)
    )
    passivity_largest_singular_value = (
        float("inf")
        if passivity_log_sigma_max > passivity_log_float_max
        else float(np.exp(passivity_log_sigma_max))
    )
passivity_margin = 1.0 - passivity_largest_gram_eigenvalue
passivity_is_physical = passivity_margin >= -PASSIVITY_NUMERICAL_TOLERANCE

print("\nPassive coupling-network check")
print(
    "Assumed R="
    f"{PASSIVITY_INTERNAL_REFLECTIVITY:.6f}, "
    f"tau_rt={PASSIVITY_ROUND_TRIP_TIME_PS:g} ps"
)
print(f"Reporting all {len(passivity_source_fractions)} source lasers")
print("laser | coupled fraction | remaining fraction | individual-source test")
print("------+------------------+--------------------+-----------------------")
# Intentionally do not slice this array: report every laser in the inspected
# coupling matrix, including zero-shot arrays larger than the training sizes.
for source_index, coupled_fraction in enumerate(
    passivity_source_fractions, start=1
):
    remaining_fraction = 1.0 - coupled_fraction
    source_passes = coupled_fraction <= 1.0 + PASSIVITY_NUMERICAL_TOLERANCE
    print(
        f"{source_index:5d} | {coupled_fraction:16.6f} | "
        f"{remaining_fraction:18.6f} | "
        f"{'PASS' if source_passes else 'FAIL'}"
    )
print(
    "largest eigenvalue of T^dagger T: "
    f"{passivity_largest_gram_eigenvalue:.6f}"
)
print(
    "smallest eigenvalue of I - T^dagger T: "
    f"{passivity_margin:.6e}"
)
print(
    "largest singular value of T: "
    f"{passivity_largest_singular_value:.6f}"
)
if passivity_is_physical:
    print(
        "RESULT: PASS -- compatible with a passive coupling network under "
        "the stated R and tau_rt assumptions."
    )
else:
    required_amplitude_scale = 1.0 / passivity_largest_singular_value
    required_power_scale = required_amplitude_scale**2
    print(
        "RESULT: FAIL -- not realizable as a passive coupling network under "
        "the stated R and tau_rt assumptions."
    )
    print(
        "A uniform kappa amplitude scale no larger than "
        f"{required_amplitude_scale:.6f} (power scale "
        f"{required_power_scale:.6f}) would reach the passivity boundary."
    )


#%% Time complete coupling-matrix inference on a log-M axis (no simulation)
BENCHMARK_M_VALUES = np.unique(
    np.rint(np.geomspace(2, 2_000, 20)).astype(int)
)
BENCHMARK_WARMUPS = 2
BENCHMARK_REPEATS = 10
# The dense policy creates multiple O(M^2) arrays. At M=10,000, just the
# returned float64 kappa and phi matrices occupy about 1.5 GiB per candidate,
# before the policy's pairwise tensors and other temporary arrays.
BENCHMARK_MAX_SAFE_M = 2_000

benchmark_checkpoint = MODEL_DIR / (
    "conditional_coupling_variable_m_"
    f"{CHECKPOINT_SIZE_LABEL}_lasers_{CHECKPOINT_KIND}{MODEL_SUFFIX}.pt"
)
benchmark_policy, _, benchmark_config, benchmark_metadata = load_sparse_checkpoint(
    benchmark_checkpoint, device="cpu"
)
benchmark_policy.eval()
benchmark_span_ghz = float(np.ptp(MANUAL_DETUNINGS_GHZ))
if Q_MODE == "automatic":
    benchmark_q_override = None
elif Q_MODE == "manual" and 0.0 <= MANUAL_Q <= 1.0:
    benchmark_q_override = MANUAL_Q
else:
    raise ValueError('Q_MODE must be "automatic" or a valid manual q in [0, 1]')


def _benchmark_one_inference(m):
    m = int(m)
    benchmark_clusters = min(N_CLUSTERS, m)
    counts = np.full(benchmark_clusters, m // benchmark_clusters, dtype=int)
    counts[:m % benchmark_clusters] += 1
    target = np.repeat(
        2.0 * np.pi * np.arange(benchmark_clusters) / benchmark_clusters,
        counts,
    )
    detunings = np.linspace(-benchmark_span_ghz / 2, benchmark_span_ghz / 2, m)
    local_config = replace(benchmark_config, n_lasers=m)
    start = perf_counter()
    _, _, q_values, kappa_matrices, phi_matrices = _infer_candidates(
        benchmark_policy, target, detunings, local_config,
        benchmark_q_override,
    )
    elapsed = perf_counter() - start
    expected_shape = (NUMBER_OF_CANDIDATES, m, m)
    if kappa_matrices.shape != expected_shape or phi_matrices.shape != expected_shape:
        raise RuntimeError("Inference did not return complete coupling matrices")
    if q_values.shape != (NUMBER_OF_CANDIDATES,):
        raise RuntimeError("Inference did not return one q per candidate")
    return elapsed


benchmark_mean_ms = []
benchmark_std_ms = []
benchmark_measured_m = []
for benchmark_m in BENCHMARK_M_VALUES:
    if benchmark_m > BENCHMARK_MAX_SAFE_M:
        print(f"M={benchmark_m:5d}: skipped (dense-inference memory guard)")
        continue
    for _ in range(BENCHMARK_WARMUPS):
        _benchmark_one_inference(benchmark_m)
    timings = [
        _benchmark_one_inference(benchmark_m)
        for _ in range(BENCHMARK_REPEATS)
    ]
    benchmark_measured_m.append(benchmark_m)
    benchmark_mean_ms.append(1.0e3 * np.mean(timings))
    benchmark_std_ms.append(1.0e3 * np.std(timings, ddof=1))
    print(
        f"M={benchmark_m:5d}: {benchmark_mean_ms[-1]:.2f} "
        f"± {benchmark_std_ms[-1]:.2f} ms (mean ± SD)"
    )

print(
    f"Checkpoint iteration {benchmark_metadata['iteration']}; "
    f"scale-aware={benchmark_config.scale_aware_detuning_features}; "
    f"inference device=CPU; candidates={NUMBER_OF_CANDIDATES}; "
    "full kappa and phi matrices timed; VCSEL simulation excluded."
)
with plt.rc_context(PLOT_STYLE):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        benchmark_measured_m, benchmark_mean_ms, yerr=benchmark_std_ms,
        fmt="-o", capsize=3,
    )
    axis.axvline(BENCHMARK_MAX_SAFE_M, color="0.5", linestyle="--", linewidth=1)
    axis.set(
        xlabel="number of lasers M",
        ylabel=r"inference time (ms, mean $\pm$ SD)",
        title=r"Full $\kappa$ and $\phi_p$ matrix inference time (CPU)",
        xscale="log",
        yscale="log",
        xlim=(2, 2_000),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plt.show()


#%% Predict the automatic q budget versus allowable phase error T (no simulation)
# The checkpoint and maximum detuning span match the inspector settings above.
# Ten random phase/detuning cases per M are reused across T. No coupling
# matrices are decoded and no VCSEL dynamics are simulated.
Q_SWEEP_M_VALUES = (3, 5, 10, 20, 50, 100, 1000)
Q_SWEEP_T_VALUES_DEG = np.linspace(1.0, 60.0, 16)
Q_SWEEP_DETUNING_SPAN_GHZ = float(np.ptp(MANUAL_DETUNINGS_GHZ))
Q_SWEEP_CASES = 10
Q_SWEEP_RANDOM_SEED = 20261007
# Avoid constructing a 10-case, million-edge GNN batch at M=1000.
Q_SWEEP_MAX_PAIRWISE_EDGES_PER_BATCH = 100_000

q_sweep_checkpoint = MODEL_DIR / (
    "conditional_coupling_variable_m_"
    f"{CHECKPOINT_SIZE_LABEL}_lasers_{CHECKPOINT_KIND}{MODEL_SUFFIX}.pt"
)
q_sweep_policy, _, q_sweep_config, q_sweep_metadata = load_sparse_checkpoint(
    q_sweep_checkpoint, device="cpu"
)
q_sweep_policy.eval()
q_sweep_values = {}
q_sweep_mean = {}
q_sweep_sd = {}
q_sweep_rng = np.random.default_rng(Q_SWEEP_RANDOM_SEED)

with torch.inference_mode():
    for q_sweep_m in Q_SWEEP_M_VALUES:
        q_sweep_targets = q_sweep_rng.uniform(
            -np.pi, np.pi, size=(Q_SWEEP_CASES, q_sweep_m)
        )
        q_sweep_detunings = q_sweep_rng.uniform(
            -0.5, 0.5, size=(Q_SWEEP_CASES, q_sweep_m)
        )
        q_sweep_detunings[:, 0] = -0.5
        q_sweep_detunings[:, -1] = 0.5
        q_sweep_detunings *= Q_SWEEP_DETUNING_SPAN_GHZ
        q_sweep_detunings -= q_sweep_detunings.mean(axis=1, keepdims=True)
        q_sweep_targets, q_sweep_detunings = designer.sort_by_detuning(
            q_sweep_targets, q_sweep_detunings
        )
        q_sweep_local_config = replace(q_sweep_config, n_lasers=q_sweep_m)
        q_sweep_batch_size = max(
            1,
            min(
                Q_SWEEP_CASES,
                Q_SWEEP_MAX_PAIRWISE_EDGES_PER_BATCH
                // (q_sweep_m * (q_sweep_m - 1)),
            ),
        )
        q_sweep_by_t = []
        for q_sweep_t in Q_SWEEP_T_VALUES_DEG:
            q_sweep_case_values = np.empty(Q_SWEEP_CASES, dtype=float)
            for q_sweep_start in range(0, Q_SWEEP_CASES, q_sweep_batch_size):
                q_sweep_stop = min(
                    q_sweep_start + q_sweep_batch_size, Q_SWEEP_CASES
                )
                q_sweep_features = encode_sparse_tolerance_context(
                    q_sweep_targets[q_sweep_start:q_sweep_stop],
                    q_sweep_detunings[q_sweep_start:q_sweep_stop],
                    float(q_sweep_t),
                    q_sweep_policy,
                    q_sweep_local_config,
                )
                q_sweep_case_values[q_sweep_start:q_sweep_stop] = (
                    torch.sigmoid(
                        q_sweep_policy.density_distribution(q_sweep_features)
                        .mean[:, 0]
                    ).cpu().numpy()
                )
            q_sweep_by_t.append(q_sweep_case_values)
        q_sweep_values[q_sweep_m] = np.stack(q_sweep_by_t)
        q_sweep_mean[q_sweep_m] = q_sweep_values[q_sweep_m].mean(axis=1)
        q_sweep_sd[q_sweep_m] = q_sweep_values[q_sweep_m].std(axis=1, ddof=1)
        print(f"Predicted q across T for M={q_sweep_m}: {Q_SWEEP_CASES} cases")

q_sweep_stage = sparse_policy.training_stage_at(
    int(q_sweep_metadata["iteration"]), q_sweep_config
)
print(
    f"Checkpoint iteration {q_sweep_metadata['iteration']} "
    f"({q_sweep_stage}); detuning span={Q_SWEEP_DETUNING_SPAN_GHZ:g} GHz."
)
if q_sweep_stage == "dense":
    print("Warning: the q head has not been trained in the dense stage.")
print("M > 10 is extrapolation beyond the training array sizes.")

with plt.rc_context(PLOT_STYLE):
    q_sweep_figure, q_sweep_axis = plt.subplots(figsize=(9, 5.5))
    for q_sweep_m in Q_SWEEP_M_VALUES:
        q_sweep_axis.errorbar(
            Q_SWEEP_T_VALUES_DEG,
            q_sweep_mean[q_sweep_m],
            yerr=q_sweep_sd[q_sweep_m],
            fmt="-o", markersize=3, capsize=2, label=f"M={q_sweep_m}",
        )
    q_sweep_axis.set(
        xlabel=r"allowable maximum phase error $T$ (degrees)",
        ylabel=r"predicted link budget $q$ (mean $\pm$ SD)",
        title=r"Automatic $q$ versus allowable phase error (10 random cases)",
        xlim=(Q_SWEEP_T_VALUES_DEG[0], Q_SWEEP_T_VALUES_DEG[-1]),
        ylim=(0.0, 1.02),
    )
    q_sweep_axis.grid(alpha=0.25)
    q_sweep_axis.legend(ncol=2)
    q_sweep_figure.tight_layout()
    plt.show()
