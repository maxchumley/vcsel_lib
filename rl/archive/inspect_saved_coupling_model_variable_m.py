#%%
"""Inspect a variable-size pooled, BiGRU, or GNN policy."""

from dataclasses import replace
from pathlib import Path
import sys

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
    predict_conditional_link_budget,
)
from rl.conditional_coupling_run_variable_m import (
    apply_relative_coupling_backbone,
    plot_selected_design,
    simulate_and_plot_best_design,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "current"  # "best" or "current"
# These identify which multi-size training run produced the checkpoint.
TRAINING_N_LASERS = (range(3, 11))  # The sizes used in the training mixture.
# Match the compact checkpoint label used by the selected training sizes.
# It updates automatically when TRAINING_N_LASERS is changed.
CHECKPOINT_SIZE_LABEL = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
# This is the size to inspect. It may be a trained or unseen array size.
N_LASERS = 3
MODEL_SUFFIX = (
    "_gnn_M3-10_span4_maxerr_selector_pcgrad"
)

# "_bigru_M2-9_span5_edge_std_degree_scaled_stratified_span_M_modulated"

# "auto" detects the architecture stored in the checkpoint. Use "bigru" or
# "pooled" to require a particular architecture and catch filename mistakes.
MODEL_ARCHITECTURE = "auto"

# Output location and optional suffix for both saved figures.
FIGURE_DIRECTORY = (
    RESULTS_DIR / "inference" / "deterministic_max_error_selector_current"
)


# Enter target phases as multiples of pi. The first entry is the reference.
MANUAL_TARGET_PHASES_PI = 1.0*(
    2.0 * np.arange(N_LASERS) / N_LASERS
)





N_CLUSTERS = 3
# N_CLUSTERS = np.copy(N_LASERS)




FIGURE_SUFFIX = f"{N_LASERS}_{N_CLUSTERS}_clusters"  # For example: "splay_noise"

# N_CLUSTERS = np.copy(N_LASERS)

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




# Enter one detuning per laser in GHz. They are centered about 0 GHz below.
MANUAL_DETUNINGS_GHZ = 0.1*0.1*np.linspace(-5.0,5.0, N_LASERS)


# *5.0*np.random.uniform(-1,1, N_LASERS)


# np.linspace(-5.0, 5.0, N_LASERS)

# 5.0*np.random.uniform(-1,1, N_LASERS)

# np.array(
#     [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0]
# )

NUMBER_OF_CANDIDATES = 1
# Maximum directed-link fraction supplied as an input to the conditional
# policy. Values below 1/M still realize the connected M-1-link minimum.
RHO_MODE = "automatic"  # "manual" or "automatic"
REQUESTED_RHO = 1.0
ALLOWABLE_MAX_PHASE_ERROR_DEG = 1.0
# When enabled, choose the triangle with the greater total kappa and mirror
# it exactly into the reverse directions before simulation and plotting.
# This remains reliable when the connectivity tree leaves a few weak links
# in the otherwise sparse triangle.
MIRROR_IF_LOWER_TRIANGLE_DISABLED = False
# False preserves a sampled training-style connected mask. If True,
# candidate zero uses the thresholded deterministic gate policy instead.
INCLUDE_DETERMINISTIC_CANDIDATE = True
# Learned-rho checkpoints already return their predicted connected backbone.
# Keep this False to inspect that learned topology without pruning it again.
# True remains available only as an optional manual second pruning pass.
USE_BACKBONE_COUPLING = False
BACKBONE_EDGE_LIMIT_FACTOR = 50.0
NETWORK_MAX_DISPLAY_EDGES = None
VALIDATION_TIME_NS = 500.0

# Optional inspection-only override for the coupling ramp.  These values are
# measured in multiples of the physical delay tau (not nanoseconds directly).
# With tau=1 ns, start=5 and rise=100 means the ramp starts at 5 ns and the
# full cosine ramp reaches its final value at about 130 ns.
OVERRIDE_COUPLING_RAMP = True
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0

# Noise is applied only during the final validation, not candidate selection.
# Set NOISE_AMPLITUDE=0.0 and NOISE_ITERATIONS=1 for a deterministic run.
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1

NOISE_RANDOM_SEED = None

# Time-resolved one-dimensional far-field array geometry.
SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-20.0, 20.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 20.0
# Show the unfiltered final angular-intensity slice beside the time-resolved
# map. Set a positive FWHM here only if angular smoothing is desired.
SHOW_FAR_FIELD_FINAL_SLICE = True
# Display-only Gaussian angular blur, applied to every far-field time slice.
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = 1.0

# Match the TeX/serif typography used by simple_example.py while keeping
# these settings local to figures produced by this inspection script.
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


def main() -> dict:
    """Load the selected checkpoint, design couplings, and plot validation."""
    if CHECKPOINT_KIND not in {"best", "current"}:
        raise ValueError('CHECKPOINT_KIND must be "best" or "current"')

    suffix = MODEL_SUFFIX
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    compact_checkpoint = MODEL_DIR / (
        f"conditional_coupling_variable_m_{CHECKPOINT_SIZE_LABEL}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )
    # The first GNN runner release accidentally used the dataclass's verbose
    # default size label. Keep it readable while an already-running process
    # continues to update that checkpoint. New GNN runs use the compact name.
    verbose_size_label = "-".join(
        str(n_lasers) for n_lasers in TRAINING_N_LASERS
    )
    legacy_checkpoint = MODEL_DIR / (
        f"conditional_coupling_variable_m_{verbose_size_label}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )
    if compact_checkpoint.exists():
        checkpoint = compact_checkpoint
    elif legacy_checkpoint.exists():
        checkpoint = legacy_checkpoint
        print(
            "Using legacy verbose-size checkpoint name from the active "
            f"GNN run: {checkpoint.name}"
        )
    else:
        raise FileNotFoundError(
            "Checkpoint not found. Checked:\n"
            f"  {compact_checkpoint}\n"
            f"  {legacy_checkpoint}"
        )

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
        raise ValueError(f"MANUAL_TARGET_PHASES_PI must have shape {expected_shape}")
    if detunings_ghz.shape != expected_shape:
        raise ValueError(f"MANUAL_DETUNINGS_GHZ must have shape {expected_shape}")

    # Detunings are defined relative to their mean optical frequency (0 GHz).
    detunings_ghz = detunings_ghz - np.mean(detunings_ghz)

    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
        f"{metadata['iteration']} (held-out={metadata['held_out_reward']:.3f})"
    )
    print(
        f"Training sizes: {TRAINING_N_LASERS}; evaluating M={N_LASERS}"
    )
    print("Centered detunings (GHz):", np.round(detunings_ghz, 4))

    if RHO_MODE == "manual":
        requested_active_link_count = max(
            N_LASERS - 1,
            int(np.ceil(REQUESTED_RHO * N_LASERS * (N_LASERS - 1))),
        )
        print(f"Manual requested rho: {REQUESTED_RHO:.4f}")
    elif RHO_MODE == "automatic":
        selection = predict_conditional_link_budget(
            policy,
            target_phases_rad,
            detunings_ghz,
            config,
            ALLOWABLE_MAX_PHASE_ERROR_DEG,
        )
        requested_active_link_count = int(selection["active_link_count"])
        print(
            "Automatic budget | allowable maximum phase error="
            f"{ALLOWABLE_MAX_PHASE_ERROR_DEG:.3f} deg | "
            f"predicted rho={selection['rho']:.4f} | "
            f"sparsity score={selection['normalized_sparsity_score']:.4f}"
        )
    else:
        raise ValueError('RHO_MODE must be "manual" or "automatic"')

    design = design_for_target(
        policy,
        target_phases_rad,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        top_k=1,
        detuning_distribution_ghz=detunings_ghz,
        active_link_count=requested_active_link_count,
        include_deterministic_candidate=(
            INCLUDE_DETERMINISTIC_CANDIDATE
        ),
        mirror_if_lower_triangle_disabled=(
            MIRROR_IF_LOWER_TRIANGLE_DISABLED
        ),
    )[0]
    if design.get("mirrored_about_diagonal", False):
        print(
            "Mirrored the dominant-triangle kappa and phi_p values "
            "about the diagonal."
        )
    if design.get("realized_rho") is not None:
        print(
            "Requested rho: "
            f"{requested_active_link_count / (N_LASERS * (N_LASERS - 1)):.4f}"
        )
        print(f"Realized rho:  {design['realized_rho']:.4f}")
    if USE_BACKBONE_COUPLING:
        design = apply_relative_coupling_backbone(
            design,
            config,
            edge_limit_factor=BACKBONE_EDGE_LIMIT_FACTOR,
        )

    phase_reward_label = (
        "Original full-matrix phase reward"
        if USE_BACKBONE_COUPLING
        else "Phase reward"
    )
    print(f"{phase_reward_label}: {design['phase_reward']:.4f}")
    target_relative = np.angle(
        np.exp(
            1.0j
            * (
                np.asarray(design["target_phases_rad"])
                - float(design["target_phases_rad"][0])
            )
        )
    )
    phase_errors = np.angle(
        np.exp(
            1.0j
            * (
                np.asarray(design["aligned_achieved_phases_rad"])
                - target_relative
            )
        )
    )
    maximum_phase_error_deg = float(
        np.rad2deg(np.max(np.abs(phase_errors[1:])))
    )
    print(
        "Maximum relative phase error: "
        f"{maximum_phase_error_deg:.3f} deg"
    )
    if RHO_MODE == "automatic":
        print(
            "Maximum-error constraint: "
            + (
                "satisfied"
                if maximum_phase_error_deg
                <= ALLOWABLE_MAX_PHASE_ERROR_DEG
                else "violated"
            )
        )
    maximum_link_count = config.n_lasers * (config.n_lasers - 1)
    active_link_count = int(design["active_link_count"])
    print(
        "Active directed links: "
        f"{active_link_count}/{maximum_link_count} "
        f"({active_link_count / maximum_link_count:.1%} active, "
        f"{1.0 - active_link_count / maximum_link_count:.1%} disabled)"
    )
    topology_edge_limit_factor = max(
        active_link_count / config.n_lasers,
        (config.n_lasers - 1) / config.n_lasers,
    )
    print(
        "Magnitude symmetry penalty: "
        f"{design['magnitude_symmetry_penalty']:.4f}"
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    figure_suffix = FIGURE_SUFFIX
    if NOISE_AMPLITUDE > 0:
        figure_suffix = f"{figure_suffix}_noise" if figure_suffix else "_noise"
    if figure_suffix and not figure_suffix.startswith("_"):
        figure_suffix = f"_{figure_suffix}"
    if USE_BACKBONE_COUPLING:
        figure_suffix = f"{figure_suffix}_backbone"
    figure_label = f"manual_target{figure_suffix}"

    # Scale this script's kappa heatmap to the selected physical per-link
    # matrix instead of the policy's full aggregate coupling budget.
    plotted_kappa_max = max(
        float(np.max(design["kappa_per_ns"])),
        1.0e-12,
    )
    with plt.rc_context(PLOT_STYLE):
        if not USE_BACKBONE_COUPLING:
            plot_selected_design(
                design,
                config,
                desired_state=figure_label,
                vertical_layout=False,
                kappa_vmax_per_ns=plotted_kappa_max,
                topology_backbone_network=True,
                topology_backbone_edge_limit_factor=(
                    topology_edge_limit_factor
                ),
                topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                show_total_coupling_budget=True,
                disabled_links_black=True,
                allowable_rms_phase_error_deg=(
                    ALLOWABLE_MAX_PHASE_ERROR_DEG
                    if RHO_MODE == "automatic"
                    else None
                ),
                allowable_phase_error_metric="maximum",
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
            # Use the sorted detuning order associated with the matrices.
            detuning_distribution_ghz=design[
                "detuning_distribution_ghz"
            ],
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
        )
        if USE_BACKBONE_COUPLING:
            design["phase_reward"] = float(
                np.mean(simulation["phase_rewards"])
            )
            design["aligned_achieved_phases_rad"] = (
                simulation["phase_mean_rad"][:, -1].copy()
            )
            print(
                "Backbone simulation phase reward: "
                f"{design['phase_reward']:.4f}"
            )
            plot_selected_design(
                design,
                config,
                desired_state=figure_label,
                vertical_layout=False,
                kappa_vmax_per_ns=plotted_kappa_max,
                topology_backbone_network=True,
                topology_backbone_edge_limit_factor=(
                    BACKBONE_EDGE_LIMIT_FACTOR
                ),
                topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                show_total_coupling_budget=True,
                disabled_links_black=True,
                allowable_rms_phase_error_deg=(
                    ALLOWABLE_MAX_PHASE_ERROR_DEG
                    if RHO_MODE == "automatic"
                    else None
                ),
                allowable_phase_error_metric="maximum",
            )
            plt.show()
    print(
        "Saved figures:\n"
        f"  {FIGURE_DIRECTORY / f'selected_design_{figure_label}.png'}\n"
        f"  {FIGURE_DIRECTORY / f'selected_design_dynamics_{figure_label}.png'}"
    )
    return {
        "policy": policy,
        "config": config,
        "metadata": metadata,
        "design": design,
        "simulation": simulation,
    }


# if __name__ == "__main__":
results = main()


#%% Inspect the learned allowable-error -> link-budget selector (no simulation)
SELECTOR_ERROR_SWEEP_DEG = np.unique(
    np.asarray(
        [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0,
         ALLOWABLE_MAX_PHASE_ERROR_DEG],
        dtype=float,
    )
)

print("\nSelector sweep for the current target and detunings")
print("error (deg) |    q    | links |  rho   |   S")
print("------------+---------+-------+--------+--------")
for allowable_error_deg in SELECTOR_ERROR_SWEEP_DEG:
    selector_output = predict_conditional_link_budget(
        results["policy"],
        results["design"]["target_phases_rad"],
        results["design"]["detuning_distribution_ghz"],
        results["config"],
        float(allowable_error_deg),
    )
    print(
        f"{allowable_error_deg:11.3f} | "
        f"{selector_output['normalized_budget_q']:7.4f} | "
        f"{selector_output['active_link_count']:5d} | "
        f"{selector_output['rho']:6.4f} | "
        f"{selector_output['normalized_sparsity_score']:6.4f}"
    )
