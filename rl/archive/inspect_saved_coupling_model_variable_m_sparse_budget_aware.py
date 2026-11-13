#%%
"""Inspect the budget-aware sparse policy with the standard sparse-inspector controls."""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_designer_variable_m as designer
from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.archive import conditional_coupling_sparse_tolerance as sparse_policy
from rl.conditional_coupling_designer_variable_m import (
    calculate_phase_reward, simulate_with_vcsel,
)
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design, simulate_and_plot_best_design,
)
from rl.archive.conditional_coupling_sparse_tolerance import (
    _decode_and_prune, encode_sparse_tolerance_context,
    load_sparse_checkpoint, maximum_circular_phase_error_deg,
    tiered_sparse_budget_reward,
)
from rl.archive.inspect_saved_coupling_model_variable_m_learned_tree_scale_aware import (
    plot_inference_summary,
)
from rl.paths import RESULTS_DIR
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_budget_aware import (
    config as training_config,
)


CHECKPOINT_KIND = "current"  # "current", "best", or "final"
N_LASERS = 3
N_CLUSTERS = 1
# This must match the fixed training threshold; it is not a sparsity dial.
ALLOWABLE_MAX_PHASE_ERROR_DEG = 3.0
DETUNING_SPAN_GHZ = 0.1
DETUNING_MODE = "manual"  # "equal", "random", or "manual"
MANUAL_DETUNINGS_GHZ = np.array([1.5, -0.5, -1.5])

Q_MODE = "automatic"  # "automatic" or "manual"
MANUAL_Q = 1.0 / 9.0
NUMBER_OF_CANDIDATES = 1
INCLUDE_DETERMINISTIC_CANDIDATE = True
MIRROR_IF_LOWER_TRIANGLE_DISABLED = False
INFERENCE_DEVICE = "auto"
RUN_VALIDATION = True
# True also simulates a q=1 reference for the checkpoint's accuracy gate.
# False simulates only NUMBER_OF_CANDIDATES designs from Q_MODE.
SIMULATE_DENSE_REFERENCE = False
VALIDATION_TIME_NS = 500.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None
NETWORK_MAX_DISPLAY_EDGES = 2_000
# Display-only: when capped, show edges across the full coupling-strength range.
NETWORK_EDGE_SELECTION = "strength_spread"
TOPOLOGY_BINS = N_LASERS

OVERRIDE_COUPLING_RAMP = True
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
FIGURE_DIRECTORY = RESULTS_DIR / "inference" / "sparse_budget_aware"
FIGURE_SUFFIX = (
    f"{N_LASERS}_{N_CLUSTERS}_clusters_tol{ALLOWABLE_MAX_PHASE_ERROR_DEG:g}_"
    + (f"q{MANUAL_Q:.1f}" if Q_MODE == "manual" else "qauto")
).replace(".", "p")

cluster_phases_pi = 2.0 * np.arange(N_CLUSTERS) / N_CLUSTERS
base_count, remainder = divmod(N_LASERS, N_CLUSTERS)
cluster_counts = [base_count + (cluster < remainder) for cluster in range(N_CLUSTERS)]
MANUAL_TARGET_PHASES_PI = np.repeat(cluster_phases_pi, cluster_counts).tolist()

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
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("INFERENCE_DEVICE must be auto, cuda, mps, or cpu")
    return requested


def _stack_validation_matrices(
    sparse_kappa: np.ndarray,
    sparse_phi: np.ndarray,
    dense_kappa: np.ndarray,
    dense_phi: np.ndarray,
    noise_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the one-candidate validation batch shape compatible with earlier runs."""
    if noise_iterations < 1:
        raise ValueError("noise_iterations must be positive")
    sparse_kappa = np.asarray(sparse_kappa)
    sparse_phi = np.asarray(sparse_phi)
    dense_kappa = np.asarray(dense_kappa)
    dense_phi = np.asarray(dense_phi)
    if sparse_kappa.ndim != 2 or sparse_kappa.shape[0] != sparse_kappa.shape[1]:
        raise ValueError("sparse coupling must be one square matrix")
    if sparse_phi.shape != sparse_kappa.shape:
        raise ValueError("sparse phase matrix must match sparse coupling")
    if dense_kappa.shape != (1, 1, *sparse_kappa.shape) or dense_phi.shape != dense_kappa.shape:
        raise ValueError("dense reference must have shape (1, 1, M, M)")
    return (
        np.concatenate((
            np.repeat(sparse_kappa[None], noise_iterations, axis=0),
            np.repeat(dense_kappa[0, 0][None], noise_iterations, axis=0),
        )),
        np.concatenate((
            np.repeat(sparse_phi[None], noise_iterations, axis=0),
            np.repeat(dense_phi[0, 0][None], noise_iterations, axis=0),
        )),
    )


def _mirror_dominant_triangle(kappa, phi):
    """Match the original inspector's optional display/validation edit."""
    if np.sum(np.triu(kappa, 1)) >= np.sum(np.tril(kappa, -1)):
        triangle_kappa, triangle_phi = np.triu(kappa, 1), np.triu(phi, 1)
    else:
        triangle_kappa, triangle_phi = np.tril(kappa, -1), np.tril(phi, -1)
    return triangle_kappa + triangle_kappa.T, triangle_phi + triangle_phi.T


def _inspection_detunings():
    if DETUNING_MODE == "equal":
        return DETUNING_SPAN_GHZ * np.linspace(-0.5, 0.5, N_LASERS)
    if DETUNING_MODE == "random":
        return DETUNING_SPAN_GHZ * np.sort(np.random.uniform(-0.5, 0.5, N_LASERS))
    if DETUNING_MODE == "manual":
        return np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    raise ValueError('DETUNING_MODE must be "equal", "random", or "manual"')


def _infer_candidates(
    policy, target, detunings, config, q_override, include_dense_reference=True
):
    """Decode candidates without running physics, as in the original inspector."""
    target, detunings = designer.sort_by_detuning(target, detunings)
    features = encode_sparse_tolerance_context(
        target, detunings, ALLOWABLE_MAX_PHASE_ERROR_DEG, policy, config
    )
    policy.eval()
    with torch.inference_mode():
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
        if include_dense_reference:
            dense_q = torch.ones_like(q[:, :1])
            dense_actions = policy.action_mean(features, dense_q[:, 0])[:, None, :]
    kappa, phi, _, _ = _decode_and_prune(
        actions, q.detach().cpu().numpy(), config
    )
    dense_kappa = dense_phi = None
    if include_dense_reference:
        dense_kappa, dense_phi, _, _ = _decode_and_prune(
            dense_actions, np.ones((1, 1)), config
        )
        dense_kappa, dense_phi = dense_kappa[0, 0], dense_phi[0, 0]
    kappa, phi = kappa[0], phi[0]
    if MIRROR_IF_LOWER_TRIANGLE_DISABLED:
        for index in range(len(kappa)):
            kappa[index], phi[index] = _mirror_dominant_triangle(
                kappa[index], phi[index]
            )
    return (
        target, detunings, q.detach().cpu().numpy()[0], kappa, phi,
        dense_kappa, dense_phi,
    )


def main() -> dict:
    if CHECKPOINT_KIND not in {"current", "best", "final"}:
        raise ValueError("CHECKPOINT_KIND must be current, best, or final")
    checkpoint = Path(getattr(training_config, f"{CHECKPOINT_KIND}_checkpoint_file"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} does not exist. Start training first.")
    policy, _, saved_config, metadata = load_sparse_checkpoint(
        checkpoint, device=_device(INFERENCE_DEVICE)
    )
    if (
        not saved_config.redistribute_pruned_coupling
        or saved_config.coupling_budget_reward_weight <= 0
    ):
        raise ValueError("This checkpoint is not from the budget-aware run")
    if saved_config.dense_reference_error_slack_deg is None:
        raise ValueError("Checkpoint does not use the dense-reference reward")
    if saved_config.fixed_tolerance_deg is None or not np.isclose(
        ALLOWABLE_MAX_PHASE_ERROR_DEG, saved_config.fixed_tolerance_deg
    ):
        raise ValueError("Inspection tolerance must match the checkpoint's fixed training tolerance")
    if not 1 <= N_CLUSTERS <= N_LASERS:
        raise ValueError("N_CLUSTERS must be between 1 and N_LASERS")
    if NUMBER_OF_CANDIDATES < 1 or NOISE_ITERATIONS < 1:
        raise ValueError("candidate and noise counts must be positive")
    if Q_MODE == "automatic":
        q_override = None
    elif Q_MODE == "manual" and 0.0 <= MANUAL_Q <= 1.0:
        q_override = MANUAL_Q
    else:
        raise ValueError('Q_MODE must be "automatic" or a valid manual q in [0, 1]')

    target = np.asarray(MANUAL_TARGET_PHASES_PI, dtype=float) * np.pi
    detunings = _inspection_detunings()
    if target.shape != (N_LASERS,) or detunings.shape != (N_LASERS,):
        raise ValueError("manual targets and detunings must each have length N_LASERS")
    detunings = detunings - np.mean(detunings)
    config = replace(
        saved_config, n_lasers=N_LASERS, n_jobs=1,
        jupyter_mode=False, noise_amplitude=0.0,
    )
    if OVERRIDE_COUPLING_RAMP:
        config = replace(
            config,
            coupling_ramp_start_delays=COUPLING_RAMP_START_DELAYS,
            coupling_ramp_rise_delays=COUPLING_RAMP_RISE_DELAYS,
        )
    target, detunings, q_values, kappas, phis, dense_kappa, dense_phi = (
        _infer_candidates(
            policy, target, detunings, config, q_override,
            include_dense_reference=RUN_VALIDATION and SIMULATE_DENSE_REFERENCE,
        )
    )
    max_links = N_LASERS * (N_LASERS - 1)
    selected = 0
    precomputed_simulation = None
    validation = None
    if RUN_VALIDATION:
        validation_config = replace(
            config,
            simulation_time_seconds=VALIDATION_TIME_NS * 1.0e-9,
            noise_amplitude=NOISE_AMPLITUDE,
        )
        np.random.seed(NOISE_RANDOM_SEED)
        # Simulate Q_MODE candidates, optionally with one q=1 reference.
        if SIMULATE_DENSE_REFERENCE:
            case_kappa = np.concatenate((kappas, dense_kappa[None]), axis=0)
            case_phi = np.concatenate((phis, dense_phi[None]), axis=0)
        else:
            case_kappa, case_phi = kappas, phis
        case_count = len(kappas) * NOISE_ITERATIONS
        total_cases = len(case_kappa) * NOISE_ITERATIONS
        all_simulations = simulate_with_vcsel(
            np.repeat(case_kappa, NOISE_ITERATIONS, axis=0),
            np.repeat(case_phi, NOISE_ITERATIONS, axis=0),
            validation_config, progress=True, smooth_frequencies=True,
            detuning_distribution_ghz=np.repeat(
                detunings[None], total_cases, axis=0
            ),
        )
        phase_reward, _, _, achieved, aligned, orientation = calculate_phase_reward(
            all_simulations[0], all_simulations[1],
            np.repeat(target[None], total_cases, axis=0),
            validation_config,
        )
        case_errors = maximum_circular_phase_error_deg(
            np.repeat(target[None], len(aligned), axis=0), aligned[:, None, :]
        )[:, 0]
        reference_kwargs = {}
        if SIMULATE_DENSE_REFERENCE:
            dense_errors = case_errors[case_count:]
            reference_kwargs = dict(
                reference_error_deg=np.tile(dense_errors, len(kappas)),
                reference_slack_deg=config.dense_reference_error_slack_deg,
            )
        candidate_rewards, _ = tiered_sparse_budget_reward(
            phase_reward[:case_count, None],
            case_errors[:case_count, None],
            np.full(case_count, ALLOWABLE_MAX_PHASE_ERROR_DEG),
            np.repeat(q_values, NOISE_ITERATIONS)[:, None],
            np.repeat(kappas, NOISE_ITERATIONS, axis=0)[:, None],
            designer.effective_maximum_kappa_per_link(
                config.maximum_kappa_per_ns, N_LASERS,
                config.normalize_incoming_coupling_by_degree,
            ),
            config.infeasible_reward_tau_deg,
            config.coupling_budget_reward_weight,
            config.feasible_phase_accuracy_weight,
            **reference_kwargs,
        )
        scores = candidate_rewards.reshape(
            len(kappas), NOISE_ITERATIONS
        ).mean(axis=1)
        selected = int(np.argmax(scores))
        chosen = slice(
            selected * NOISE_ITERATIONS, (selected + 1) * NOISE_ITERATIONS
        )
        precomputed_simulation = (
            all_simulations[0], *(part[chosen] for part in all_simulations[1:])
        )
        aligned_mean = np.angle(np.mean(np.exp(1j * aligned[chosen]), axis=0))
        target_relative = designer.make_relative_to_laser_1(target)
        mean_error = np.angle(np.exp(1j * (aligned_mean - target_relative)))
        allowed_error = ALLOWABLE_MAX_PHASE_ERROR_DEG
        if SIMULATE_DENSE_REFERENCE:
            dense_error = float(np.mean(dense_errors))
            allowed_error = min(
                allowed_error,
                dense_error + config.dense_reference_error_slack_deg,
            )
        validation = dict(
            phase_reward=float(np.mean(phase_reward[chosen])),
            ranking_reward=float(scores[selected]),
            achieved_phases_rad=np.angle(
                np.mean(np.exp(1j * achieved[chosen]), axis=0)
            ),
            aligned_achieved_phases_rad=aligned_mean,
            orientation=float(orientation[chosen][0]),
            maximum_phase_error_deg=float(np.rad2deg(np.max(np.abs(mean_error[1:])))),
        )
        if SIMULATE_DENSE_REFERENCE:
            validation.update(
                dense_reference_error_deg=dense_error,
                dense_reference_phase_reward=float(np.mean(phase_reward[case_count:])),
            )
        validation["constraint_satisfied"] = bool(
            validation["maximum_phase_error_deg"] <= allowed_error
        )

    kappa, phi = kappas[selected], phis[selected]
    receivers, sources = np.nonzero(kappa > 0.0)
    design = dict(
        rank=1,
        target_phases_rad=target,
        detuning_distribution_ghz=detunings,
        allowable_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
        q=float(q_values[selected]),
        predicted_q=float(q_values[selected]),
        active_link_count=len(receivers),
        active_connection_fraction=len(receivers) / max_links,
        realized_rho=len(receivers) / max_links,
        kappa_per_ns=kappa, phi_p_rad=phi,
        receiver_indices=receivers, source_indices=sources,
    )
    if validation is not None:
        design.update(validation)
    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration {metadata['iteration']}; "
        f"device={next(policy.parameters()).device}; M={N_LASERS}"
    )
    print(
        f"Predicted q: {design['q']:.4f}; active directed links: "
        f"{len(receivers)}/{max_links}; total kappa={kappa.sum():.4g} ns^-1"
    )
    if validation is not None:
        reference_text = (
            f"dense-reference error={dense_error:.3f} deg; "
            if SIMULATE_DENSE_REFERENCE else ""
        )
        print(
            f"Phase reward={validation['phase_reward']:.4f}; "
            f"maximum error={validation['maximum_phase_error_deg']:.3f} deg; "
            f"{reference_text}accuracy gate={allowed_error:.3f} deg; "
            f"constraint={'satisfied' if validation['constraint_satisfied'] else 'violated'}"
        )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    suffix = f"_{FIGURE_SUFFIX}" if FIGURE_SUFFIX else ""
    label = f"budget_aware{suffix}"
    with plt.rc_context(PLOT_STYLE):
        if RUN_VALIDATION:
            figure = plot_selected_design(
                design, config, desired_state=label,
                vertical_layout=False,
                kappa_vmax_per_ns=max(float(kappa.max()), 1.0e-12),
                topology_backbone_network=True,
                topology_backbone_edge_limit_factor=max(
                    design['active_link_count']/N_LASERS,
                    (N_LASERS - 1)/N_LASERS,
                ),
                topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                topology_edge_selection=NETWORK_EDGE_SELECTION,
                topology_sparsity_mask=False,
                topology_detuning_arc_layout=True,
                show_total_coupling_budget=True, disabled_links_black=True,
                allowable_rms_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
                allowable_phase_error_metric="maximum",
            )
            figure.savefig(FIGURE_DIRECTORY / f"selected_design_{label}.png", dpi=200)
            plt.show()
            dynamics = simulate_and_plot_best_design(
                design, config,
                validation_delay_count=VALIDATION_TIME_NS / (config.delay_seconds * 1e9),
                desired_state=label, detuning_distribution_ghz=detunings,
                legend_fontsize=PLOT_STYLE['legend.fontsize'],
                noise_iterations=NOISE_ITERATIONS, dual_axis_power_panel=True,
                far_field_panel=SHOW_FAR_FIELD,
                emitter_spacing_m=EMITTER_SPACING_UM * 1e-6,
                far_field_theta_range_deg=FAR_FIELD_THETA_RANGE_DEG,
                far_field_n_theta=FAR_FIELD_N_THETA,
                far_field_max_time_points=FAR_FIELD_MAX_TIME_POINTS,
                far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
                far_field_final_slice_panel=SHOW_FAR_FIELD_FINAL_SLICE,
                far_field_slice_convolution_fwhm_deg=FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG,
                precomputed_simulation=precomputed_simulation,
                allowable_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
            )
        else:
            print("Inference only; no dynamics were simulated.")
            topology_design = {
                **design,
                "kappa_per_ns": kappa[receivers, sources],
                "phi_p_rad": phi[receivers, sources],
            }
            figure = plot_inference_summary(
                topology_design, kappa, phi, bins=TOPOLOGY_BINS,
                max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                edge_selection=NETWORK_EDGE_SELECTION,
            )
            figure.savefig(FIGURE_DIRECTORY / f"inferred_summary_{label}.png", dpi=350)
            plt.show()
            dynamics = None
    return dict(policy=policy, config=config, metadata=metadata,
                design=design, simulation=dynamics)


if __name__ == "__main__":
    budget_aware_sparse_inspection_results = main()


#%% Optional passivity check for the inspected coupling matrix.
# Set these from the actual device before interpreting the result physically.
PASSIVITY_INTERNAL_REFLECTIVITY = 0.9959
PASSIVITY_ROUND_TRIP_TIME_FS = 28.8

if __name__ == "__main__":
    _inspected = globals().get("budget_aware_sparse_inspection_results")
    if _inspected is None:
        raise RuntimeError("Run the inspection cell first")
    if not 0.0 < PASSIVITY_INTERNAL_REFLECTIVITY < 1.0:
        raise ValueError("reflectivity must lie in (0, 1)")
    if PASSIVITY_ROUND_TRIP_TIME_FS <= 0.0:
        raise ValueError("round-trip time must be positive")
    _design = _inspected["design"]
    _transmission = (
        _design["kappa_per_ns"]
        * PASSIVITY_ROUND_TRIP_TIME_FS * 1.0e-6
        * np.sqrt(PASSIVITY_INTERNAL_REFLECTIVITY)
        / (1.0 - PASSIVITY_INTERNAL_REFLECTIVITY)
        * np.exp(-1.0j * _design["phi_p_rad"])
    )
    np.fill_diagonal(_transmission, 0.0)
    _source_fractions = np.sum(np.abs(_transmission) ** 2, axis=0)
    _largest_singular_value = float(np.linalg.svd(_transmission, compute_uv=False)[0])
    print("Passivity check (using the assumed mirror reflectivity and round-trip time):")
    print(f"Maximum single-source coupled power fraction: {_source_fractions.max():.6g}")
    print(f"Largest singular value of field transfer: {_largest_singular_value:.6g}")
    print(f"Coherent multiport passive: {_largest_singular_value <= 1.0 + 1.0e-10}")
