#%%
"""Inspect a learned-tree design with the standard four-panel and dynamics plots."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.archive.conditional_coupling_learned_tree_tolerance import (
    infer_learned_tree,
    load_checkpoint, topology_summary,
)
from rl.conditional_coupling_designer_variable_m import (
    calculate_phase_reward, simulate_with_vcsel,
)
from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design, simulate_and_plot_best_design,
)
from rl.archive.conditional_coupling_sparse_tolerance import maximum_circular_phase_error_deg
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_learned_tree_scale_aware import make_config
from rl.archive.inspect_saved_coupling_model_variable_m_sparse_graph_scale_aware import plot_edge_list
from rl.paths import RESULTS_DIR

CHECKPOINT = Path(make_config().current_checkpoint_file)
N_LASERS = 5
N_CLUSTERS = 5
DETUNING_SPAN_GHZ = 1.0
MANUAL_DETUNINGS_GHZ = None
ALLOWABLE_MAX_PHASE_ERROR_DEG = 1.0
MANUAL_Q = 0.9  # None uses the model's deterministic q; 0 uses a learned tree.
INFERENCE_DEVICE = "auto"
RUN_VALIDATION = True
VALIDATION_TIME_NS = 500.0
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None
OVERRIDE_COUPLING_RAMP = True
COUPLING_RAMP_START_DELAYS = 5.0
COUPLING_RAMP_RISE_DELAYS = 200.0
NETWORK_MAX_DISPLAY_EDGES = 6000
SHOW_DETUNING_BINNED_SUMMARY = False
DETUNING_SUMMARY_BINS = 5
DETUNING_SUMMARY_MAX_EDGES = None
SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-30.0, 30.0)
FAR_FIELD_N_THETA = 2400
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 20.0
SHOW_FAR_FIELD_FINAL_SLICE = True
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = None
TOPOLOGY_BINS = np.copy(N_LASERS)
FIGURE_DIRECTORY = RESULTS_DIR / "inference" / "learned_tree_scale_aware"
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


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def materialize_coupling_matrices(design, n_lasers):
    """Build dense matrices only for plotting or physical validation."""
    receivers = np.asarray(design["receiver_indices"], dtype=int)
    sources = np.asarray(design["source_indices"], dtype=int)
    kappa = np.zeros((n_lasers, n_lasers))
    phi = np.zeros_like(kappa)
    kappa[receivers, sources] = design["kappa_per_ns"]
    phi[receivers, sources] = design["phi_p_rad"]
    return kappa, phi


def plot_inference_topology(
    ax, design, *, bins, max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
    edge_selection="strongest",
):
    """Use the validated plot's detuning-ordered, two-sided arc convention."""
    m = len(design["target_phases_rad"])
    bins = min(m, bins) if m > 20 else m
    mapping = np.minimum(np.arange(m) * bins // m, bins - 1)
    source = mapping[np.asarray(design["source_indices"], dtype=int)]
    receiver = mapping[np.asarray(design["receiver_indices"], dtype=int)]
    strength = np.zeros((bins, bins))
    phase_sum = np.zeros((bins, bins), dtype=complex)
    np.add.at(strength, (receiver, source), design["kappa_per_ns"])
    np.add.at(phase_sum, (receiver, source),
              design["kappa_per_ns"] * np.exp(1j * design["phi_p_rad"]))
    detuning = np.array([
        np.mean(design["detuning_distribution_ghz"][mapping == index])
        for index in range(bins)
    ])
    curvature = np.clip((bins - 3) / 17, 0, 1)
    angles = np.linspace(np.pi, 2 * np.pi, bins)
    positions = np.column_stack((
        (1 - curvature) * np.linspace(-1, 1, bins) + curvature * np.cos(angles),
        0.78 * curvature * np.sin(angles),
    ))
    edges = [
        (int(s), int(r), float(strength[r, s]), float(np.angle(phase_sum[r, s])))
        for r, s in zip(*np.nonzero(strength)) if r != s
    ]
    if max_display_edges < 1:
        raise ValueError("max_display_edges must be positive")
    if edge_selection == "strength_spread":
        displayed = coupling_plots.sample_edges_across_strengths(
            edges, max_display_edges, lambda edge: edge[2]
        )
    elif edge_selection == "strongest":
        displayed = sorted(edges, key=lambda edge: edge[2], reverse=True)[
            :max_display_edges
        ]
    else:
        raise ValueError("edge_selection must be strongest or strength_spread")
    values = [edge[2] for edge in edges]
    minimum, maximum = (min(values), max(values)) if values else (0.0, 1.0)
    span = max(maximum - minimum, 1e-12)
    curve_t = np.linspace(0, 1, 48)[:, None]

    def draw_arcs(selected, *, background):
        segments, colors, widths = [], [], []
        arrow_stride = max(1, int(np.ceil(len(selected) / 200)))
        for index, (s, r, weight, phase) in enumerate(selected):
            start, end = positions[s], positions[r]
            sign = 1.0 if end[0] >= start[0] else -1.0
            control = np.array([
                0.5 * (start[0] + end[0]),
                sign * (0.035 + 0.65 * abs(end[0] - start[0])),
            ])
            curve = ((1 - curve_t) ** 2 * start
                     + 2 * (1 - curve_t) * curve_t * control
                     + curve_t**2 * end)
            segments.append(curve)
            color = "0.48" if background else plt.cm.twilight((phase + np.pi) / (2 * np.pi))
            colors.append(color)
            widths.append((0.035 + 0.68 * (weight - minimum) / span) if background
                          else (0.10 + 1.65 * (weight - minimum) / span))
            if not background and index % arrow_stride == 0:
                ax.add_patch(FancyArrowPatch(
                    curve[34], curve[39], arrowstyle="-|>", mutation_scale=14,
                    linewidth=0, color=color, alpha=0.78, zorder=2,
                ))
        if segments:
            ax.add_collection(LineCollection(
                segments, colors=colors, linewidths=widths,
                alpha=0.055 if background else 0.78,
                capstyle="round", zorder=1,
            ))

    # When capped, do not draw the omitted links as gray context: that would
    # make a 100-edge display limit appear ineffective on dense networks.
    if len(edges) <= max_display_edges:
        draw_arcs(edges, background=True)
    draw_arcs(displayed, background=False)
    nodes = ax.scatter(positions[:, 0], positions[:, 1], c=detuning,
                       cmap="bwr", vmin=float(detuning.min()),
                       vmax=float(detuning.max()), s=300 * min(1, 10 / bins),
                       edgecolors="0.35", linewidths=0.7, zorder=3)
    label_stride = max(1, int(np.ceil(bins / 20)))
    for index, (x, y) in enumerate(positions):
        if index % label_stride == 0 or index == bins - 1:
            ax.text(x, y, str(index + 1), ha="center", va="center",
                    fontsize=9 if bins > 20 else 11, zorder=4)
    ax.set(xlim=(-1.12, 1.12), ylim=(-0.97, 0.94),
           title="Detuning-ordered\ncoupling arcs")
    # The network owns the width of two matrix panels. Do not preserve equal
    # data-unit scaling here, which would compress the arcs into a small oval.
    ax.set_aspect("auto")
    ax.axis("off")
    colorbar = ax.figure.colorbar(nodes, ax=ax, shrink=0.82)
    colorbar.set_label("detuning (GHz)")
    ax.text(0.5, -0.05,
            f"showing {len(displayed)} of {len(edges)} retained directed links"
            + (" (strength-spread sample)\n" if edge_selection == "strength_spread" else "\n")
            + "upper/lower = direction; width = normalized coupling; arrows = overlay",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)


def plot_inference_summary(
    design, kappa, phi, *, bins, max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
    edge_selection="strongest",
):
    """Plot inferred matrices and topology without a settled simulation."""
    n_lasers = len(kappa)
    disabled = kappa <= 0.0
    kappa_cmap = plt.get_cmap("Blues").copy()
    phase_cmap = plt.get_cmap("twilight").copy()
    kappa_cmap.set_bad("black")
    phase_cmap.set_bad("black")
    figure, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True,
                                gridspec_kw={"width_ratios": (1, 1, 2.2)})
    magnitude = axes[0].imshow(
        np.ma.masked_where(disabled.T, kappa.T), origin="upper",
        cmap=kappa_cmap, vmin=0.0, vmax=max(float(kappa.max()), 1e-12),
    )
    phase = axes[1].imshow(
        np.ma.masked_where(disabled.T, phi.T), origin="upper",
        cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
    )
    total_coupling = float(np.sum(kappa))
    axes[0].set_title(
        r"$\kappa$ (ns$^{-1}$), $K_{\mathrm{total}}="
        f"{total_coupling:.3g}"
        r"\ \mathrm{ns}^{-1}$"
        "\nblack = disabled"
    )
    axes[1].set_title(r"$\phi_p$" + "\nblack = disabled")
    ticks = np.unique(np.rint(np.linspace(0, n_lasers - 1,
                                        min(n_lasers, 6))).astype(int))
    for ax in axes:
        ax.set(xlabel="receiver", ylabel="source")
        ax.set_xticks(ticks, ticks + 1)
        ax.set_yticks(ticks, ticks + 1)
    figure.colorbar(magnitude, ax=axes[0], shrink=0.82)
    phase_bar = figure.colorbar(phase, ax=axes[1], shrink=0.82)
    phase_bar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    phase_bar.set_ticklabels(
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )
    plot_inference_topology(
        axes[2], design, bins=bins, max_display_edges=max_display_edges,
        edge_selection=edge_selection,
    )
    return figure


def main(checkpoint=CHECKPOINT, n_lasers=N_LASERS, run_validation=RUN_VALIDATION):
    policy, _, metadata = load_checkpoint(checkpoint, device=resolve_device(INFERENCE_DEVICE))
    if not 1 <= N_CLUSTERS <= n_lasers:
        raise ValueError("Require 1 <= N_CLUSTERS <= M")
    counts = np.full(N_CLUSTERS, n_lasers // N_CLUSTERS)
    counts[:n_lasers % N_CLUSTERS] += 1
    target = np.repeat(2 * np.pi * np.arange(N_CLUSTERS) / N_CLUSTERS, counts)
    detunings = (np.linspace(-DETUNING_SPAN_GHZ / 2, DETUNING_SPAN_GHZ / 2, n_lasers)
                 if MANUAL_DETUNINGS_GHZ is None else np.asarray(MANUAL_DETUNINGS_GHZ))
    design = infer_learned_tree(policy, target, detunings,
                                ALLOWABLE_MAX_PHASE_ERROR_DEG, q_override=MANUAL_Q)
    print(f"Checkpoint iteration {metadata['iteration']}; device={next(policy.parameters()).device}")
    print(json.dumps({**topology_summary(design), "root_index": int(design["root_index"]),
                      "root_original_index": int(design["sorted_to_original"][design["root_index"]])}, indent=2))
    print("Single deterministic edge-list inference; q=0 is a learned M-1 tree, q=1 is all-to-all.")
    simulation = None
    if run_validation:
        config = replace(policy.config, n_lasers=n_lasers, n_jobs=1,
                         jupyter_mode=False, noise_amplitude=NOISE_AMPLITUDE)
        if OVERRIDE_COUPLING_RAMP:
            config = replace(config,
                coupling_ramp_start_delays=COUPLING_RAMP_START_DELAYS,
                coupling_ramp_rise_delays=COUPLING_RAMP_RISE_DELAYS)
        kappa, phi = materialize_coupling_matrices(design, n_lasers)
        validation_config = replace(
            config, simulation_time_seconds=VALIDATION_TIME_NS * 1e-9,
        )
        np.random.seed(NOISE_RANDOM_SEED)
        precomputed_simulation = simulate_with_vcsel(
            np.repeat(kappa[None, :, :], NOISE_ITERATIONS, axis=0),
            np.repeat(phi[None, :, :], NOISE_ITERATIONS, axis=0),
            validation_config,
            progress=True,
            smooth_frequencies=True,
            detuning_distribution_ghz=np.repeat(
                design["detuning_distribution_ghz"][None, :],
                NOISE_ITERATIONS, axis=0,
            ),
        )
        reward, _, _, _, aligned, _ = calculate_phase_reward(
            precomputed_simulation[0], precomputed_simulation[1],
            np.repeat(design["target_phases_rad"][None, :],
                      NOISE_ITERATIONS, axis=0),
            validation_config,
        )
        achieved = np.angle(np.mean(np.exp(1j * aligned), axis=0))
        error_deg = float(maximum_circular_phase_error_deg(
            design["target_phases_rad"], achieved[None, None, :])[0, 0])
        maximum_links = n_lasers * (n_lasers - 1)
        plotted_design = {**design, "kappa_per_ns": kappa, "phi_p_rad": phi,
            "aligned_achieved_phases_rad": achieved,
            "phase_reward": float(np.mean(reward)),
            "maximum_phase_error_deg": error_deg,
            "active_link_count": len(design["source_indices"]),
            "active_connection_fraction": len(design["source_indices"]) / maximum_links}
        print(f"Phase reward={plotted_design['phase_reward']:.4f}; "
              f"maximum phase error={error_deg:.3f} deg; "
              f"constraint={'satisfied' if error_deg <= ALLOWABLE_MAX_PHASE_ERROR_DEG else 'violated'}")
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        coupling_plots.figure_directory = FIGURE_DIRECTORY
        figure_label = (
            f"learned_tree_M{n_lasers}_span"
            f"{np.ptp(design['detuning_distribution_ghz']):g}GHz_"
            f"tol{ALLOWABLE_MAX_PHASE_ERROR_DEG:g}deg"
        )
        topology_edge_limit_factor = max(
            len(design["source_indices"]) / n_lasers,
            (n_lasers - 1) / n_lasers,
        )
        with plt.rc_context(PLOT_STYLE):
            design_figure = plot_selected_design(
                plotted_design, config, desired_state=figure_label,
                vertical_layout=False,
                kappa_vmax_per_ns=max(float(np.max(kappa)), 1.0e-12),
                topology_backbone_network=True,
                topology_backbone_edge_limit_factor=topology_edge_limit_factor,
                topology_max_display_edges=NETWORK_MAX_DISPLAY_EDGES,
                topology_sparsity_mask=False,
                topology_detuning_arc_layout=True,
                topology_detuning_summary=SHOW_DETUNING_BINNED_SUMMARY,
                topology_detuning_summary_bins=DETUNING_SUMMARY_BINS,
                topology_detuning_summary_max_edges=DETUNING_SUMMARY_MAX_EDGES,
                show_total_coupling_budget=True,
                disabled_links_black=True,
                allowable_rms_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
                allowable_phase_error_metric="maximum",
            )
            # The final title can collide with the nearby colorbar in the
            # compact four-panel layout, especially with non-TeX fonts.
            design_figure.axes[3].set_title(
                "Detuning-binned\ncoupling summary" if SHOW_DETUNING_BINNED_SUMMARY
                else "Detuning-ordered\ncoupling arcs"
            )
            design_figure.savefig(
                FIGURE_DIRECTORY / f"selected_design_{figure_label}.png"
            )
            plt.show()
            simulation = simulate_and_plot_best_design(
                plotted_design, config,
                validation_delay_count=VALIDATION_TIME_NS / (config.delay_seconds * 1e9),
                desired_state=figure_label,
                detuning_distribution_ghz=design["detuning_distribution_ghz"],
                legend_fontsize=PLOT_STYLE["legend.fontsize"],
                noise_iterations=NOISE_ITERATIONS,
                dual_axis_power_panel=True,
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
        print("Inference-only: no settled phases or dynamics were simulated.")
        FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        figure_label = (
            f"learned_tree_M{n_lasers}_span"
            f"{np.ptp(design['detuning_distribution_ghz']):g}GHz_"
            f"tol{ALLOWABLE_MAX_PHASE_ERROR_DEG:g}deg"
        )
        with plt.rc_context(PLOT_STYLE):
            kappa, phi = materialize_coupling_matrices(design, n_lasers)
            summary_figure = plot_inference_summary(
                design, kappa, phi, bins=TOPOLOGY_BINS,
            )
            summary_figure.savefig(
                FIGURE_DIRECTORY / f"inferred_summary_{figure_label}.png", dpi=350,
            )
            plt.show()
    return dict(policy=policy, design=design, metadata=metadata,
                simulation=simulation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--m", type=int, default=N_LASERS)
    parser.add_argument("--inference-only", action="store_true")
    args, _ = parser.parse_known_args()
    learned_tree_inspection_results = main(args.checkpoint, args.m,
                                            RUN_VALIDATION and not args.inference_only)
