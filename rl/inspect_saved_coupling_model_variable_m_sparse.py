#%%
"""Inspect the selector-free tolerance-conditioned sparse GNN policy."""

from dataclasses import replace
from pathlib import Path
import sys

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
from rl.conditional_coupling_sparse_tolerance import (
    design_sparse_for_target,
    encode_sparse_tolerance_context,
    load_sparse_checkpoint,
    q_to_active_link_counts,
)
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design,
    simulate_and_plot_best_design,
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
MODEL_SUFFIX = "_gnn_sparse_tol_pcgrad_refine5000_exp_lr1e-4to1e-6"

N_LASERS = 25
N_CLUSTERS = 1
ALLOWABLE_MAX_PHASE_ERROR_DEG = 30.0

# The automatic mode uses the deterministic mean of the learned q policy.
# Manual mode bypasses that one output while retaining the same GNN edge
# policy, connected pruning, and simulation.
Q_MODE = "automatic"  # "automatic" or "manual"
MANUAL_Q = 1.0
NUMBER_OF_CANDIDATES = 1
INCLUDE_DETERMINISTIC_CANDIDATE = True

# Optional inspection-only symmetry operation.
MIRROR_IF_LOWER_TRIANGLE_DISABLED = False
# All links remain visible as thin gray context.  Limit only the colored
# coupling overlay so dense arrays do not become an opaque phase-color mass.
NETWORK_MAX_DISPLAY_EDGES = 600

VALIDATION_TIME_NS = 500.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None

OVERRIDE_COUPLING_RAMP = True
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0

SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-20.0, 20.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 20.0
SHOW_FAR_FIELD_FINAL_SLICE = True
# Display-only Gaussian angular blur, applied to every far-field time slice.
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = 1.0

FIGURE_DIRECTORY = RESULTS_DIR / "inference" / "sparse_tolerance_current"
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
MANUAL_DETUNINGS_GHZ =  1.0 * 0.1 * np.linspace(-5.0, 5.0, N_LASERS)

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


def _mirror_dominant_triangle(design: dict, config) -> dict:
    """Mirror the stronger triangle and update the one-candidate simulation."""
    kappa = np.asarray(design["kappa_per_ns"], dtype=float)
    phi = np.asarray(design["phi_p_rad"], dtype=float)
    if np.sum(np.triu(kappa, 1)) >= np.sum(np.tril(kappa, -1)):
        triangle_kappa = np.triu(kappa, 1)
        triangle_phi = np.triu(phi, 1)
    else:
        triangle_kappa = np.tril(kappa, -1)
        triangle_phi = np.tril(phi, -1)
    mirrored_kappa = triangle_kappa + triangle_kappa.T
    mirrored_phi = triangle_phi + triangle_phi.T
    simulation = designer.simulate_candidates_parallel(
        np.asarray(design["target_phases_rad"])[None, :],
        mirrored_kappa[None, None, :, :],
        mirrored_phi[None, None, :, :],
        config,
        detuning_distributions_ghz=np.asarray(
            design["detuning_distribution_ghz"]
        )[None, :],
    )
    updated = dict(design)
    updated["kappa_per_ns"] = mirrored_kappa
    updated["phi_p_rad"] = mirrored_phi
    updated["phase_reward"] = float(simulation["phase_reward"][0, 0])
    updated["achieved_phases_rad"] = simulation["achieved_phases_rad"][0, 0]
    updated["aligned_achieved_phases_rad"] = simulation[
        "aligned_achieved_phases_rad"
    ][0, 0]
    updated["orientation"] = float(simulation["orientation"][0, 0])
    mask = mirrored_kappa > 0.0
    updated["active_link_count"] = int(np.count_nonzero(mask))
    updated["active_connection_fraction"] = float(
        updated["active_link_count"] / (config.n_lasers * (config.n_lasers - 1))
    )
    updated["realized_rho"] = updated["active_connection_fraction"]
    errors = np.angle(
        np.exp(
            1.0j
            * (
                np.asarray(updated["aligned_achieved_phases_rad"])
                - designer.make_relative_to_laser_1(
                    updated["target_phases_rad"]
                )
            )
        )
    )
    updated["maximum_phase_error_deg"] = float(
        np.rad2deg(np.max(np.abs(errors[1:])))
    )
    updated["constraint_satisfied"] = bool(
        updated["maximum_phase_error_deg"]
        <= updated["allowable_error_deg"]
    )
    return updated


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

    design = design_sparse_for_target(
        policy,
        target,
        detunings,
        ALLOWABLE_MAX_PHASE_ERROR_DEG,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        q_override=q_override,
        include_deterministic_candidate=INCLUDE_DETERMINISTIC_CANDIDATE,
    )[0]
    if MIRROR_IF_LOWER_TRIANGLE_DISABLED:
        design = _mirror_dominant_triangle(design, config)

    maximum_links = N_LASERS * (N_LASERS - 1)
    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
        f"{metadata['iteration']} (dense held-out={metadata['held_out_reward']:.3f})"
    )
    print(f"Training sizes: {TRAINING_N_LASERS}; evaluating M={N_LASERS}")
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

    # Plot the realized coupling matrices and topology before the dynamics.
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

        np.random.seed(NOISE_RANDOM_SEED)
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
