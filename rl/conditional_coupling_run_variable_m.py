#%%
"""Run the variable-size BiGRU conditional coupling designer.

All executable work is inside :func:`main`, which is required for spawn-safe
multiprocessing on macOS and Windows. Importing this module never trains a
model or creates worker processes.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import torch
from scipy.constants import c, hbar
import os

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    calculate_phase_reward,
    DesignerConfig,
    PHASE_TICK_LABELS,
    PHASE_TICKS,
    PolicyNetwork,
    design_for_target,
    design_sparsest_for_target,
    directed_link_indices,
    effective_maximum_kappa_per_link,
    load_checkpoint,
    make_policy,
    make_relative_to_laser_1,
    named_phase_targets,
    normalized_magnitude_symmetry_penalty,
    normalized_squared_coupling_cost,
    plot_training_history,
    sample_detuning_distributions,
    run_architecture_sanity_checks,
    simulate_with_vcsel,
    train_reinforce,
    validate_policy,
    validate_policy_sizes,
)
from rl.paths import MODEL_DIR, RESULTS_DIR

figure_directory = RESULTS_DIR

#  1. Settings

# Train one shared policy over M=2,...,9 from the first update.  This run
# Train every array size from the beginning. The decoder's hidden features are
# modulated by M so the shared policy can represent a genuinely different
# mapping for the M=2 boundary case while retaining shared weights.
TRAINING_N_LASERS = tuple(range(2, 10))
# Inspect the same trained size in the post-training example below.
EVALUATION_N_LASERS = 3
# Optional Stage-3 validation on unseen sizes using the same loaded policy.
RUN_ZERO_SHOT_VALIDATION = False
ZERO_SHOT_N_LASERS = (2,)

# Multiprocessing is target-level. Each worker receives one target and keeps
# all 64 candidates for that target in one vectorized integration.
config = DesignerConfig(
    n_lasers=EVALUATION_N_LASERS,
    training_n_lasers=TRAINING_N_LASERS,
    training_n_laser_weights=None,
    training_iterations=2500,
    enable_array_size_curriculum=False,
    array_size_curriculum_initial_count=1,
    array_size_curriculum_start_iteration=500,
    array_size_curriculum_add_interval=100,
    targets_per_batch=14,
    candidates_per_target=64,
    candidates_per_worker_task=64,
    node_hidden_sizes=(64,),
    node_embedding_dim=64,
    # Train the recurrent architecture for this run. The pooled implementation
    # remains available only for loading older pooled checkpoints.
    encoder_architecture="bigru",
    gru_hidden_size=64,
    edge_hidden_sizes=(128, 128),
    # Condition the edge means on M and give each directed kappa/phi action
    # its own decoder-predicted exploration width, as in the fixed-M policy.
    condition_on_n_lasers=True,
    condition_log_std_on_n_lasers=False,
    modulate_edge_decoder_by_n_lasers=True,
    edge_conditioned_log_std=True,
    smooth_edge_log_std_bounds=False,
    learning_rate=1.0e-3,
    # Keep the learning rate at 1e-3 throughout training.
    learning_rate_switch_iteration=1500,
    learning_rate_after_switch=1.0e-3,
    # Symmetry can be selected independently for magnitudes and phases.
    force_symmetric_kappa=False,
    force_symmetric_phi_p=False,
    # Leave all coupling directions independent for this architecture test.
    # In particular, balancing the two triangular sums would force
    # kappa[0, 1] == kappa[1, 0] when M=2.
    balanced_coupling=False,
    # Keep each receiver's maximum total incoming coupling independent of M.
    # This leaves M=2 unchanged and divides each M>2 edge by M-1.
    normalize_incoming_coupling_by_degree=True,
    # Set the switch True to lower the learned log-std ceiling on this schedule.
    initial_log_std=-0.25,
    minimum_log_std=-3.0,
    maximum_log_std=0.75,
    # Preserve broad exploration through iteration 1500, then lower both the
    # floor and ceiling gradually over the final 1000 iterations.
    enable_minimum_log_std_annealing=True,
    initial_minimum_log_std=-1.5,
    minimum_log_std_anneal_start_iteration=1500,
    minimum_log_std_anneal_iterations=1000,
    enable_exploration_annealing=True,
    exploration_anneal_start_iteration=1500,
    exploration_anneal_iterations=1000,
    final_maximum_log_std=-1.5,
    maximum_kappa_per_ns=100.0,
    # Disable symmetry preference so this test matches the fully directed
    # fixed-size baseline as closely as possible.
    magnitude_symmetry_weight=0.0,
    held_out_target_count=64,
    validation_interval=10,
    # Evaluate the same held-out targets and normalized detuning patterns at
    # the maximum physical span currently allowed by the curriculum.
    validation_follows_detuning_curriculum=True,
    random_seed=7,
    device="cpu",
    n_jobs=os.cpu_count(),
    jupyter_mode=True,
    initial_kappa_fraction = 0.05,
    # Keep every sampled detuning distribution within a total 5.0 GHz span.
    detuning_span_ghz=5.0,
    # Hold a 0.1 GHz maximum peak-to-peak span through iteration 200, then
    # increase it linearly to 5 GHz over the following 800 iterations.
    detuning_curriculum_initial_half_span_ghz=0.05,
    detuning_curriculum_warmup_iterations=200,
    detuning_curriculum_iterations=1000,
    # Give every M the same randomized, stratified distribution of realized
    # peak-to-peak detuning spans within the current curriculum limit.
    detuning_span_sampling_mode="stratified_span",
    simulation_time_seconds=500e-9,
    coupling_ramp_start_delays = 5.0,
    coupling_ramp_rise_delays = 100.0
)

TRAIN_NEW_MODEL = True
RUN_ARCHITECTURE_SANITY_CHECKS = True
# Optional label appended before ``.pt`` (for example, ``"_wide_kappa"``).
# The training-size range is already included compactly as ``firsttolast``.
# Keep this label matched to encoder_architecture so runs cannot overwrite
# one another (for example, use "_bigru" for a recurrent run).
FILE_SUFFIX = (
    "_bigru_M2-9_span5_edge_std_degree_scaled_stratified_span_"
    "M_modulated"
)
# Optional fixed detuning in GHz for the final post-training trajectory.
# Use one value per laser, e.g. [0.0, -2.0, 1.0, 2.0, -1.0], or leave None
# to retain the uniformly random detuning used during training.
MANUAL_DETUNING_DISTRIBUTION_GHZ = None#np.linspace(-5.0, 5.0, config.n_lasers) if config.n_lasers > 1 else None


def _apply_checkpoint_suffix(
    run_config: DesignerConfig, suffix: str
) -> DesignerConfig:
    """Append a user label to both checkpoint filenames."""
    if not suffix:
        return run_config
    normalized = suffix if suffix.startswith("_") else f"_{suffix}"

    def add_suffix(filename: str | None) -> str | None:
        if filename is None:
            return None
        path = Path(filename)
        return str(path.with_name(f"{path.stem}{normalized}{path.suffix}"))

    return replace(
        run_config,
        best_checkpoint_file=add_suffix(run_config.best_checkpoint_file),
        current_checkpoint_file=add_suffix(
            run_config.current_checkpoint_file
        ),
        final_checkpoint_file=add_suffix(run_config.final_checkpoint_file),
    )


# Use a compact range label instead of embedding every size in each name.
# Changing TRAINING_N_LASERS automatically changes this to (for example)
# ``3to10`` or ``5to20``. ``FILE_SUFFIX`` is applied below.
training_size_label = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
config = replace(
    config,
    best_checkpoint_file=str(
        MODEL_DIR
        / f"conditional_coupling_variable_m_{training_size_label}_lasers_best.pt"
    ),
    current_checkpoint_file=str(
        MODEL_DIR
        / f"conditional_coupling_variable_m_{training_size_label}_lasers_current.pt"
    ),
    final_checkpoint_file=str(
        MODEL_DIR
        / f"conditional_coupling_variable_m_{training_size_label}_lasers_final.pt"
    ),
)
config = _apply_checkpoint_suffix(config, FILE_SUFFIX)


#  2. Matrix and settled-phase summary

def _relative_coupling_backbone(
    kappa_per_ns: np.ndarray,
    *,
    edge_limit_factor: float = 1.5,
) -> tuple[set[tuple[int, int]], nx.Graph, np.ndarray]:
    """Build a sparse display-only backbone from receiver-normalized links.

    Matrix entry ``[receiver, source]`` is normalized by that receiver's
    total incoming coupling.  A maximum-spanning forest keeps the visible
    graph connected where possible.  Strongest incoming/outgoing candidates
    and then globally strongest remaining directions fill a limit of roughly
    ``edge_limit_factor * M`` directed links.
    """
    kappa_per_ns = np.asarray(kappa_per_ns, dtype=float)
    if (
        kappa_per_ns.ndim != 2
        or kappa_per_ns.shape[0] != kappa_per_ns.shape[1]
    ):
        raise ValueError("kappa_per_ns must be a square matrix")
    n_lasers = kappa_per_ns.shape[0]
    if n_lasers < 2:
        raise ValueError("kappa_per_ns must contain at least two lasers")

    incoming_totals = np.sum(kappa_per_ns, axis=1, keepdims=True)
    relative_coupling = np.divide(
        kappa_per_ns,
        incoming_totals,
        out=np.zeros_like(kappa_per_ns, dtype=float),
        where=incoming_totals > 1.0e-15,
    )
    np.fill_diagonal(relative_coupling, 0.0)

    reciprocal_graph = nx.Graph()
    reciprocal_graph.add_nodes_from(range(n_lasers))
    for first in range(n_lasers):
        for second in range(first + 1, n_lasers):
            reciprocal_strength = max(
                float(relative_coupling[second, first]),
                float(relative_coupling[first, second]),
            )
            if reciprocal_strength > 0.0:
                reciprocal_graph.add_edge(
                    first,
                    second,
                    weight=reciprocal_strength,
                )

    selected_directions: set[tuple[int, int]] = set()
    maximum_spanning_forest = nx.maximum_spanning_tree(
        reciprocal_graph,
        weight="weight",
    )
    for first, second in maximum_spanning_forest.edges():
        first_to_second = float(relative_coupling[second, first])
        second_to_first = float(relative_coupling[first, second])
        selected_directions.add(
            (first, second)
            if first_to_second >= second_to_first
            else (second, first)
        )

    # Prioritize each node's strongest incoming and outgoing direction.
    node_candidates: dict[tuple[int, int], float] = {}
    for receiver in range(n_lasers):
        strengths = relative_coupling[receiver].copy()
        strengths[receiver] = -np.inf
        source = int(np.argmax(strengths))
        if strengths[source] > 0.0:
            node_candidates[(source, receiver)] = float(strengths[source])
    for source in range(n_lasers):
        strengths = relative_coupling[:, source].copy()
        strengths[source] = -np.inf
        receiver = int(np.argmax(strengths))
        if strengths[receiver] > 0.0:
            node_candidates[(source, receiver)] = float(strengths[receiver])

    edge_limit = max(
        len(selected_directions),
        int(np.ceil(edge_limit_factor * n_lasers)),
    )
    for direction, _ in sorted(
        node_candidates.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if len(selected_directions) >= edge_limit:
            break
        selected_directions.add(direction)

    # Fill any remaining slots using the strongest relative influences.
    if len(selected_directions) < edge_limit:
        remaining_directions = sorted(
            (
                (
                    float(relative_coupling[receiver, source]),
                    source,
                    receiver,
                )
                for source in range(n_lasers)
                for receiver in range(n_lasers)
                if source != receiver
                and relative_coupling[receiver, source] > 0.0
                and (source, receiver) not in selected_directions
            ),
            reverse=True,
        )
        for _, source, receiver in remaining_directions:
            if len(selected_directions) >= edge_limit:
                break
            selected_directions.add((source, receiver))

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(range(n_lasers))
    for source, receiver in selected_directions:
        pair_strength = max(
            float(relative_coupling[receiver, source]),
            float(relative_coupling[source, receiver]),
        )
        if layout_graph.has_edge(source, receiver):
            layout_graph[source][receiver]["weight"] = max(
                float(layout_graph[source][receiver]["weight"]),
                pair_strength,
            )
        else:
            layout_graph.add_edge(source, receiver, weight=pair_strength)

    maximum_layout_strength = max(
        (
            float(data["weight"])
            for _, _, data in layout_graph.edges(data=True)
        ),
        default=1.0,
    )
    for _, _, data in layout_graph.edges(data=True):
        normalized_strength = (
            float(data["weight"])
            / max(maximum_layout_strength, 1.0e-15)
        )
        # Squaring expands the contrast between strong and weak attractions.
        data["layout_weight"] = max(normalized_strength**2, 1.0e-8)

    return selected_directions, layout_graph, relative_coupling


def apply_relative_coupling_backbone(
    design: dict,
    run_config: DesignerConfig,
    *,
    edge_limit_factor: float = 1.5,
) -> dict:
    """Return a design pruned to the links shown by the backbone panel.

    Matrix entry ``[receiver, source]`` represents ``source -> receiver``.
    Coupling magnitudes and phases outside the selected directed backbone are
    set to zero, so downstream plotting and simulation use the same topology.
    The input design is not modified.
    """
    kappa_per_ns = np.asarray(design["kappa_per_ns"], dtype=float)
    phi_p_rad = np.asarray(design["phi_p_rad"], dtype=float)
    expected_shape = (run_config.n_lasers, run_config.n_lasers)
    if kappa_per_ns.shape != expected_shape or phi_p_rad.shape != expected_shape:
        raise ValueError(
            "design coupling matrices must both have shape "
            f"{expected_shape}"
        )

    selected_directions, _, _ = _relative_coupling_backbone(
        kappa_per_ns,
        edge_limit_factor=edge_limit_factor,
    )
    active_mask = np.zeros(expected_shape, dtype=bool)
    for source, receiver in selected_directions:
        active_mask[receiver, source] = True

    backbone_kappa = np.where(active_mask, kappa_per_ns, 0.0)
    backbone_phi = np.where(active_mask, phi_p_rad, 0.0)
    active_link_count = int(np.count_nonzero(backbone_kappa))
    maximum_link_count = run_config.n_lasers * (run_config.n_lasers - 1)
    maximum_kappa_per_link = effective_maximum_kappa_per_link(
        run_config.maximum_kappa_per_ns,
        run_config.n_lasers,
        run_config.normalize_incoming_coupling_by_degree,
    )

    backbone_design = dict(design)
    backbone_design["kappa_per_ns"] = backbone_kappa
    backbone_design["phi_p_rad"] = backbone_phi
    backbone_design["active_link_count"] = active_link_count
    backbone_design["active_connection_fraction"] = (
        active_link_count / maximum_link_count
    )
    backbone_design["normalized_coupling_budget"] = float(
        np.sum(backbone_kappa)
        / (maximum_link_count * maximum_kappa_per_link)
    )
    backbone_design["normalized_squared_coupling_cost"] = float(
        normalized_squared_coupling_cost(
            backbone_kappa[None, :, :], maximum_kappa_per_link
        )[0]
    )
    backbone_design["magnitude_symmetry_penalty"] = float(
        normalized_magnitude_symmetry_penalty(
            backbone_kappa[None, :, :], maximum_kappa_per_link
        )[0]
    )
    magnitude_gates = design.get("magnitude_gates")
    if magnitude_gates is not None:
        receivers, sources = directed_link_indices(run_config.n_lasers)
        if np.asarray(magnitude_gates).shape == receivers.shape:
            backbone_design["magnitude_gates"] = active_mask[
                receivers, sources
            ].astype(float)
    backbone_design["backbone_directions"] = tuple(
        sorted(selected_directions)
    )
    return backbone_design


def plot_selected_design(
    design: dict,
    run_config: DesignerConfig,
    desired_state: str = "splay",
    *,
    vertical_layout: bool = False,
    kappa_vmax_per_ns: float | None = None,
    topology_backbone_network: bool = False,
    topology_backbone_edge_limit_factor: float = 1.5,
    topology_max_display_edges: int | None = None,
    topology_sparsity_mask: bool = False,
    topology_detuning_arc_layout: bool = False,
    topology_detuning_summary: bool = False,
    topology_detuning_summary_bins: int = 5,
    topology_detuning_summary_max_edges: int | None = None,
    show_total_coupling_budget: bool = False,
    disabled_links_black: bool = False,
    allowable_rms_phase_error_deg: float | None = None,
    allowable_phase_error_metric: str = "rms",
) -> plt.Figure:
    """Plot one selected design and its settled phase relationship.

    ``kappa_vmax_per_ns`` is the physical per-link color limit.  When it is
    omitted, derive the limit from the configured aggregate coupling budget.
    ``topology_backbone_network`` changes only the fourth, network panel.  It
    never changes the plotted matrices or the design used by simulation.
    ``topology_backbone_edge_limit_factor`` controls the approximate number
    of displayed directed links as a multiple of the array size.
    ``topology_max_display_edges`` optionally caps only the network panel at
    the strongest displayed coupling magnitudes. It does not prune the
    matrices or simulation design.
    ``topology_sparsity_mask`` shows every possible directed link: retained
    links in gray and zeroed links in red. It is intended for diagnosing the
    learned pruning pattern rather than coupling magnitude or phase.
    ``topology_detuning_arc_layout`` places nodes left-to-right by detuning
    and renders retained links as arcs. Reciprocal directions occupy opposite
    sides of the baseline.
    ``topology_detuning_summary`` replaces the detailed topology panel with
    a small directed quotient network over ordered detuning bins.
    ``topology_detuning_summary_max_edges`` caps the displayed aggregate
    arrows; by default it shows the strongest three arrows per bin.
    ``show_total_coupling_budget`` appends the directed off-diagonal sum to
    the coupling-matrix title.
    ``disabled_links_black`` masks zero-magnitude links in both coupling
    heatmaps, including the no-self-feedback diagonal, and draws them black.
    ``allowable_rms_phase_error_deg`` draws an angular guide of
    ``+/- allowable_rms_phase_error_deg`` around each distinct target phase.
    ``allowable_phase_error_metric`` controls whether the legend describes
    that guide as an RMS visualization or a maximum per-laser constraint.
    Laser labels are one-based for presentation; array indices remain
    zero-based internally.
    """
    if allowable_rms_phase_error_deg is not None and not (
        0.0 <= allowable_rms_phase_error_deg <= 180.0
    ):
        raise ValueError(
            "allowable_rms_phase_error_deg must lie in [0, 180]"
        )
    if allowable_phase_error_metric not in {"rms", "maximum"}:
        raise ValueError(
            "allowable_phase_error_metric must be 'rms' or 'maximum'"
        )
    if topology_detuning_summary_bins < 2:
        raise ValueError("topology_detuning_summary_bins must be at least 2")
    if (
        topology_detuning_summary_max_edges is not None
        and topology_detuning_summary_max_edges < 1
    ):
        raise ValueError("topology_detuning_summary_max_edges must be positive")
    if vertical_layout:
        # Reserve the same colorbar column beside every row so the four main
        # plotting axes have identical left and right boundaries.
        figure = plt.figure(figsize=(7, 21), constrained_layout=True)
        grid = figure.add_gridspec(
            4,
            2,
            width_ratios=(1.0, 0.055),
            height_ratios=(1.0, 1.0, 1.0, 1.0),
        )
        axes = np.asarray(
            [figure.add_subplot(grid[row, 0]) for row in range(4)]
        )
        colorbar_axes = (
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, 1]),
        )
        colorbar_spacer = figure.add_subplot(grid[2, 1])
        colorbar_spacer.set_axis_off()
        detuning_colorbar_axis = figure.add_subplot(grid[3, 1])
    else:
        figure, axes = plt.subplots(
            1,
            4,
            figsize=(18, 4),
            constrained_layout=True,
        )
        colorbar_axes = (None, None)
        detuning_colorbar_axis = None

    # Coupling-magnitude matrix
    if kappa_vmax_per_ns is None:
        kappa_vmax_per_ns = effective_maximum_kappa_per_link(
            run_config.maximum_kappa_per_ns,
            run_config.n_lasers,
            run_config.normalize_incoming_coupling_by_degree,
        )
    plotted_kappa_per_ns = np.asarray(design["kappa_per_ns"], dtype=float)
    disabled_links = plotted_kappa_per_ns <= 0.0
    plotted_kappa_heatmap = plotted_kappa_per_ns.T
    kappa_colormap = "Blues"
    if disabled_links_black:
        kappa_colormap = plt.get_cmap("Blues").copy()
        kappa_colormap.set_bad("black")
        plotted_kappa_heatmap = np.ma.masked_where(
            disabled_links.T,
            plotted_kappa_heatmap,
        )
    kappa_image = axes[0].imshow(
        plotted_kappa_heatmap,
        origin="upper",
        cmap=kappa_colormap,
        vmin=0.0,
        vmax=kappa_vmax_per_ns,
    )
    kappa_title = r"$\kappa$ (ns$^{-1}$)"
    if show_total_coupling_budget:
        off_diagonal = ~np.eye(
            plotted_kappa_per_ns.shape[0],
            dtype=bool,
        )
        total_coupling_per_ns = float(
            np.sum(plotted_kappa_per_ns[off_diagonal])
        )
        kappa_title = (
            r"$\kappa$ (ns$^{-1}$), "
            r"$K_{\mathrm{total}}="
            f"{total_coupling_per_ns:.3g}"
            r"\ \mathrm{ns}^{-1}$"
        )
    if disabled_links_black:
        kappa_title += "\nblack = disabled"
    axes[0].set(
        title=kappa_title,
        xlabel="receiver",
        ylabel="source",
    )
    kappa_colorbar_kwargs = (
        {"cax": colorbar_axes[0]} if vertical_layout else {"ax": axes[0]}
    )
    figure.colorbar(kappa_image, **kappa_colorbar_kwargs)

    # Coupling-phase matrix
    plotted_phi_p_rad = np.asarray(design["phi_p_rad"], dtype=float).T
    phase_colormap = "twilight"
    if disabled_links_black:
        phase_colormap = plt.get_cmap("twilight").copy()
        phase_colormap.set_bad("black")
        plotted_phi_p_rad = np.ma.masked_where(
            disabled_links.T,
            plotted_phi_p_rad,
        )
    phase_image = axes[1].imshow(
        plotted_phi_p_rad,
        origin="upper",
        cmap=phase_colormap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set(
        title=(
            r"$\phi_p$" + ("\nblack = disabled" if disabled_links_black else "")
        ),
        xlabel="receiver",
        ylabel="source",
    )
    if run_config.n_lasers <= 10:
        laser_tick_positions = np.arange(run_config.n_lasers)
    else:
        laser_tick_positions = np.unique(
            np.rint(
                np.linspace(0, run_config.n_lasers - 1, 6)
            ).astype(int)
        )
    laser_tick_labels = laser_tick_positions + 1
    for matrix_axis in axes[:2]:
        matrix_axis.set_xticks(laser_tick_positions, laser_tick_labels)
        matrix_axis.set_yticks(laser_tick_positions, laser_tick_labels)
    phase_colorbar_kwargs = (
        {"cax": colorbar_axes[1]} if vertical_layout else {"ax": axes[1]}
    )
    phase_colorbar = figure.colorbar(
        phase_image,
        ticks=PHASE_TICKS,
        **phase_colorbar_kwargs,
    )
    phase_colorbar.ax.set_yticklabels(PHASE_TICK_LABELS)

    # Relative target and achieved phases
    target_relative = make_relative_to_laser_1(
        np.asarray(design["target_phases_rad"])
    )
    achieved_aligned = make_relative_to_laser_1(
        np.asarray(design["aligned_achieved_phases_rad"])
    )

    target_x = np.cos(target_relative)
    target_y = np.sin(target_relative)

    achieved_x = np.cos(achieved_aligned)
    achieved_y = np.sin(achieved_aligned)

    # Draw the unit circle.
    circle_angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[2].plot(
        np.cos(circle_angle),
        np.sin(circle_angle),
        color="0.55",
        linewidth=1.2,
        zorder=1,
    )

    # Draw one tolerance arc per distinct target phase. Multiple lasers in
    # the same phase cluster intentionally share one guide instead of
    # repeatedly darkening the same arc.
    if (
        allowable_rms_phase_error_deg is not None
        and allowable_rms_phase_error_deg > 0.0
    ):
        tolerance_rad = np.deg2rad(allowable_rms_phase_error_deg)
        distinct_targets = np.unique(np.round(target_relative, decimals=12))
        for target_index, target_angle in enumerate(distinct_targets):
            tolerance_angles = np.linspace(
                target_angle - tolerance_rad,
                target_angle + tolerance_rad,
                101,
            )
            axes[2].plot(
                np.cos(tolerance_angles),
                np.sin(tolerance_angles),
                color="tab:blue",
                alpha=0.25,
                linewidth=7.0,
                solid_capstyle="round",
                label=(
                    (
                        rf"max $\pm {allowable_rms_phase_error_deg:g}^\circ$"
                        if allowable_phase_error_metric == "maximum"
                        else rf"RMS $\pm {allowable_rms_phase_error_deg:g}^\circ$"
                    )
                    if target_index == 0
                    else None
                ),
                zorder=2,
            )

    # Target phases: solid blue circles.
    axes[2].scatter(
        target_x,
        target_y,
        s=55,
        marker="o",
        color="tab:blue",
        label="target",
        zorder=3,
    )

    # Achieved phases: orange squares.
    axes[2].scatter(
        achieved_x,
        achieved_y,
        s=45,
        marker="s",
        color="tab:orange",
        label="achieved",
        zorder=4,
    )

    axes[2].axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
    axes[2].axvline(0.0, color="0.8", linewidth=0.7, zorder=0)

    axes[2].set(
        title="Settled phase comparison",
        xlabel=r"$\cos(\Delta\phi)$",
        ylabel=r"$\sin(\Delta\phi)$",
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        aspect="equal",
    )

    axes[2].set_xticks([-1.0, 0.0, 1.0])
    axes[2].set_yticks([-1.0, 0.0, 1.0])
    # All phase markers and tolerance arcs lie on the unit circle, leaving
    # its center open for the legend without hiding the phase data.
    axes[2].legend(loc="center")
    axes[2].grid(alpha=0.2)

    # Coupling network. Matrix entry [receiver, source] represents the edge
    # source -> receiver, and edge width is proportional to kappa.
    kappa_per_ns = plotted_kappa_per_ns
    phi_p_rad = np.asarray(design["phi_p_rad"], dtype=float)
    node_detunings_ghz = np.asarray(
        design.get(
            "detuning_distribution_ghz",
            np.zeros(run_config.n_lasers),
        ),
        dtype=float,
    )
    if node_detunings_ghz.shape != (run_config.n_lasers,):
        raise ValueError(
            "design detuning_distribution_ghz must have shape "
            f"({run_config.n_lasers},)"
        )
    if topology_detuning_summary:
        bin_count = min(topology_detuning_summary_bins, run_config.n_lasers)
        ordered_nodes = np.argsort(node_detunings_ghz)
        detuning_bins = [
            np.asarray(nodes, dtype=int)
            for nodes in np.array_split(ordered_nodes, bin_count)
        ]
        # Lay macro-nodes out in detuning order on one baseline.  The flows
        # then alternate above/below the row instead of being obscured by a
        # curved node arrangement.
        bin_positions = np.column_stack(
            (np.linspace(-1.08, 1.08, bin_count), np.zeros(bin_count))
        )
        bin_detunings = np.asarray(
            [node_detunings_ghz[nodes].mean() for nodes in detuning_bins]
        )
        detuning_minimum = float(np.min(node_detunings_ghz))
        detuning_maximum = float(np.max(node_detunings_ghz))
        if np.isclose(detuning_minimum, detuning_maximum):
            detuning_minimum -= 0.5
            detuning_maximum += 0.5
        detuning_normalization = Normalize(
            vmin=detuning_minimum, vmax=detuning_maximum
        )
        detuning_colormap = plt.get_cmap("bwr")
        phase_colormap = plt.get_cmap("twilight")
        phase_normalization = Normalize(vmin=-np.pi, vmax=np.pi)

        # ``weight[a, b]`` is the mean directed coupling from detuning bin a
        # into bin b.  Divide by possible links so unequal bin sizes do not
        # create a spurious strength trend.
        weight = np.zeros((bin_count, bin_count), dtype=float)
        phase = np.zeros((bin_count, bin_count), dtype=float)
        for source_bin, sources in enumerate(detuning_bins):
            for receiver_bin, receivers in enumerate(detuning_bins):
                block_kappa = kappa_per_ns[np.ix_(receivers, sources)]
                block_phase = phi_p_rad[np.ix_(receivers, sources)]
                possible_links = len(sources) * len(receivers)
                if source_bin == receiver_bin:
                    block_kappa = np.where(
                        np.eye(len(sources), dtype=bool), 0.0, block_kappa
                    )
                    possible_links -= len(sources)
                if possible_links <= 0:
                    continue
                total = float(block_kappa.sum())
                weight[source_bin, receiver_bin] = total / possible_links
                if total > 0.0:
                    phase[source_bin, receiver_bin] = float(
                        np.angle(
                            np.sum(block_kappa * np.exp(1.0j * block_phase))
                        )
                    )

        positive_weight = weight[weight > 0.0]
        minimum_weight = (
            float(positive_weight.min()) if len(positive_weight) else 0.0
        )
        maximum_weight = (
            float(positive_weight.max()) if len(positive_weight) else 1.0
        )
        aggregate_edges = sorted(
            (
                (weight[source_bin, receiver_bin], source_bin, receiver_bin)
                for source_bin in range(bin_count)
                for receiver_bin in range(bin_count)
                if source_bin != receiver_bin
                and weight[source_bin, receiver_bin] > 0.0
            ),
            reverse=True,
        )
        summary_edge_limit = (
            topology_detuning_summary_max_edges
            if topology_detuning_summary_max_edges is not None
            else 3 * bin_count
        )
        displayed_summary_directions = {
            (source_bin, receiver_bin)
            for _, source_bin, receiver_bin in aggregate_edges[:summary_edge_limit]
        }

        def summary_width(value: float) -> float:
            if maximum_weight <= minimum_weight + 1.0e-12:
                return 2.4
            return 0.45 + 4.8 * (value - minimum_weight) / (
                maximum_weight - minimum_weight
            )

        # Alternate unordered bin pairs above/below the baseline.  Keeping
        # the signed curvature fixed for a reciprocal pair puts its reverse
        # direction on the other side of the straight-line layout.
        for source_bin in range(bin_count):
            for receiver_bin in range(bin_count):
                if source_bin == receiver_bin or weight[source_bin, receiver_bin] <= 0.0:
                    continue
                if (source_bin, receiver_bin) not in displayed_summary_directions:
                    continue
                pair_side = (
                    1.0
                    if (min(source_bin, receiver_bin) + max(source_bin, receiver_bin)) % 2 == 0
                    else -1.0
                )
                axes[3].add_patch(
                    FancyArrowPatch(
                        bin_positions[source_bin],
                        bin_positions[receiver_bin],
                        connectionstyle=(
                            # Make aggregate flows visibly distinct even when
                            # their endpoint bins are adjacent.
                            f"arc3,rad={pair_side * (0.25 + 0.10 * abs(receiver_bin - source_bin))}"
                        ),
                        arrowstyle="-|>",
                        mutation_scale=13.0,
                        linewidth=summary_width(weight[source_bin, receiver_bin]),
                        color=phase_colormap(
                            phase_normalization(phase[source_bin, receiver_bin])
                        ),
                        alpha=0.80,
                        zorder=1,
                    )
                )

        diagonal_weight = np.diag(weight)
        diagonal_maximum = max(float(np.max(diagonal_weight)), 1.0e-12)
        node_size_scale = min(1.0, 8.0 / bin_count)
        axes[3].scatter(
            bin_positions[:, 0],
            bin_positions[:, 1],
            s=node_size_scale**2 * (
                260.0 + 340.0 * diagonal_weight / diagonal_maximum
            ),
            c=bin_detunings,
            cmap=detuning_colormap,
            norm=detuning_normalization,
            edgecolors=[
                phase_colormap(phase_normalization(value))
                for value in np.diag(phase)
            ],
            linewidths=2.7,
            zorder=3,
        )
        for bin_index, (position, members) in enumerate(
            zip(bin_positions, detuning_bins)
        ):
            axes[3].text(
                position[0],
                position[1],
                f"{bin_index + 1}\n{len(members)}",
                ha="center",
                va="center",
                fontsize=max(5.5, 8.0 * node_size_scale),
                zorder=4,
            )
        axes[3].set_title("Detuning-binned\ncoupling summary", pad=12)
        axes[3].text(
            0.5,
            -0.055,
            (
                f"{bin_count} detuning bins; "
                f"{len(displayed_summary_directions)}/{len(aggregate_edges)} strongest flows"
                "\n"
                r"width $\propto\overline{\kappa}$; color = circular mean $\overline{\phi_p}$"
                "\n"
                "fill = mean detuning; ring = within-bin phase"
                "\n"
                "reciprocal arrows use opposite sides"
            ),
            transform=axes[3].transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            color="0.35",
        )
        axes[3].set_xlim(-1.22, 1.22)
        axes[3].set_ylim(-1.18, 1.18)
        axes[3].set_aspect("equal")
        axes[3].set_axis_off()
        colorbar_kwargs = (
            {"cax": detuning_colorbar_axis}
            if vertical_layout
            else {"ax": axes[3], "fraction": 0.055, "pad": 0.02}
        )
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(
                norm=detuning_normalization, cmap=detuning_colormap
            ),
            orientation="vertical",
            **colorbar_kwargs,
        )
        colorbar.set_label("detuning (GHz)")
        if not vertical_layout:
            # Give the quotient network the unused gap to its left without
            # changing the geometry of the matrix or phase-comparison axes.
            # Its right edge (and therefore its colorbar) stays fixed.
            figure.get_layout_engine().execute(figure)
            topology_position = axes[3].get_position()
            axes[3].set_position(
                [
                    topology_position.x0 - 0.050,
                    topology_position.y0,
                    topology_position.width + 0.050,
                    topology_position.height,
                ]
            )
        plt.savefig(
            figure_directory / f"selected_design_{desired_state}.png",
            dpi=300,
            bbox_inches="tight",
        )
        return figure
    symmetric_kappa = np.allclose(
        kappa_per_ns,
        kappa_per_ns.T,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    phase_transpose_error = np.angle(
        np.exp(1.0j * (phi_p_rad - phi_p_rad.T))
    )
    symmetric_phi = np.allclose(
        phase_transpose_error,
        0.0,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    graph = nx.Graph() if symmetric_kappa and symmetric_phi else nx.DiGraph()
    graph.add_nodes_from(range(run_config.n_lasers))
    if symmetric_kappa:
        for source in range(run_config.n_lasers):
            for receiver in range(source + 1, run_config.n_lasers):
                coupling = float(kappa_per_ns[receiver, source])
                if coupling > 0.0:
                    graph.add_edge(source, receiver, weight=coupling)
    else:
        for source in range(run_config.n_lasers):
            for receiver in range(run_config.n_lasers):
                if source == receiver:
                    continue
                coupling = float(kappa_per_ns[receiver, source])
                if coupling > 0.0:
                    graph.add_edge(source, receiver, weight=coupling)

    if topology_backbone_network:
        # This branch changes only the network summary.  The physical design
        # matrices above remain untouched and are still passed to simulation.
        topology_directions, layout_graph, _ = (
            _relative_coupling_backbone(
                kappa_per_ns,
                edge_limit_factor=topology_backbone_edge_limit_factor,
            )
        )
        if topology_max_display_edges is not None:
            if topology_max_display_edges < 1:
                raise ValueError("topology_max_display_edges must be positive")
            if len(topology_directions) > topology_max_display_edges:
                topology_directions = set(
                    sorted(
                        topology_directions,
                        key=lambda direction: float(
                            kappa_per_ns[direction[1], direction[0]]
                        ),
                        reverse=True,
                    )[:topology_max_display_edges]
                )
                layout_graph = nx.Graph()
                layout_graph.add_nodes_from(range(run_config.n_lasers))
                for source, receiver in topology_directions:
                    coupling = float(kappa_per_ns[receiver, source])
                    if layout_graph.has_edge(source, receiver):
                        layout_graph[source][receiver]["weight"] = max(
                            float(layout_graph[source][receiver]["weight"]),
                            coupling,
                        )
                    else:
                        layout_graph.add_edge(
                            source, receiver, weight=coupling
                        )
    else:
        topology_directions = set()
        # Use an undirected, reciprocal-strength graph only to determine node
        # positions. Strongly coupled lasers pull together even when the
        # physical coupling matrices themselves are directed.
        layout_graph = nx.Graph()
        layout_graph.add_nodes_from(range(run_config.n_lasers))
        for source in range(run_config.n_lasers):
            for receiver in range(source + 1, run_config.n_lasers):
                reciprocal_strength = 0.5 * (
                    kappa_per_ns[receiver, source]
                    + kappa_per_ns[source, receiver]
                )
                if reciprocal_strength > 0.0:
                    layout_graph.add_edge(
                        source,
                        receiver,
                        weight=float(reciprocal_strength),
                    )

    maximum_spanning_forest = nx.maximum_spanning_tree(
        layout_graph, weight="weight"
    )

    # Start from a detuning-ordered circle, then let reciprocal coupling
    # strength determine an abstract topology layout.  The initialization
    # keeps repeated plots stable while the spring forces pull strongly
    # coupled pairs together.  These display distances are not physical
    # emitter separations.
    node_order = np.argsort(node_detunings_ghz)
    node_angles = np.pi / 2.0 - 2.0 * np.pi * (
        np.arange(run_config.n_lasers) / run_config.n_lasers
    )
    initial_positions = {
        int(node): np.array(
            [np.cos(node_angles[rank]), np.sin(node_angles[rank])]
        )
        for rank, node in enumerate(node_order)
    }
    if not topology_backbone_network:
        maximum_reciprocal_coupling = max(
            (
                float(data["weight"])
                for _, _, data in layout_graph.edges(data=True)
            ),
            default=1.0,
        )
        for _, _, data in layout_graph.edges(data=True):
            # Square-root compression prevents the largest link from
            # collapsing its endpoints while still making stronger links
            # more attractive in the original display mode.
            normalized_weight = (
                float(data["weight"])
                / max(maximum_reciprocal_coupling, 1.0e-12)
            )
            data["layout_weight"] = 0.15 + 3.0 * np.sqrt(
                normalized_weight
            )
    arc_layout = topology_detuning_arc_layout and run_config.n_lasers > 2
    arc_layout_curvature = float(
        np.clip((run_config.n_lasers - 3) / (20 - 3), 0.0, 1.0)
    )
    cluster_layout = (
        topology_backbone_network
        and run_config.n_lasers > 20
        and not arc_layout
    )
    cluster_centers: list[float] = []
    cluster_half_widths: list[float] = []
    if arc_layout:
        detuning_order = np.argsort(node_detunings_ghz)
        row_x = np.linspace(-1.0, 1.0, run_config.n_lasers)
        semicircle_angles = np.linspace(
            np.pi, 2.0 * np.pi, run_config.n_lasers
        )
        semicircle_x = np.cos(semicircle_angles)
        semicircle_y = 0.78 * np.sin(semicircle_angles)
        positions = {
            int(node): np.array(
                [
                    (1.0 - arc_layout_curvature) * row_x[rank]
                    + arc_layout_curvature * semicircle_x[rank],
                    arc_layout_curvature * semicircle_y[rank],
                ]
            )
            for rank, node in enumerate(detuning_order)
        }
    elif cluster_layout:
        # Use the requested phase clusters as the topology geometry: nodes
        # share a wedge only when they share a wrapped target phase, while
        # detuning determines their radius.  This makes the layout stable and
        # interpretable even when thousands of links overlap.
        target_cluster_members: list[list[int]] = []
        for node, target_phase in enumerate(target_relative):
            for cluster_index, center in enumerate(cluster_centers):
                if np.isclose(
                    np.angle(np.exp(1.0j * (target_phase - center))),
                    0.0,
                    atol=1.0e-8,
                ):
                    target_cluster_members[cluster_index].append(node)
                    break
            else:
                cluster_centers.append(float(target_phase))
                target_cluster_members.append([node])
        cluster_count = len(cluster_centers)
        wedge_width = 0.82 * 2.0 * np.pi / max(cluster_count, 1)
        cluster_half_widths = [0.5 * wedge_width] * cluster_count
        detuning_span = max(
            float(np.ptp(node_detunings_ghz)), 1.0e-12
        )
        normalized_detunings = (
            (node_detunings_ghz - np.min(node_detunings_ghz)) / detuning_span
        )
        positions = {}
        for center, members in zip(cluster_centers, target_cluster_members):
            members = sorted(members, key=lambda node: node_detunings_ghz[node])
            offsets = np.linspace(
                -0.42 * wedge_width,
                0.42 * wedge_width,
                len(members),
            )
            for node, offset in zip(members, offsets):
                radius = 0.34 + 0.62 * normalized_detunings[node]
                angle = center + offset
                positions[node] = np.array(
                    [radius * np.cos(angle), radius * np.sin(angle)]
                )
    else:
        positions = nx.spring_layout(
            layout_graph,
            pos=initial_positions,
            weight="layout_weight",
            seed=run_config.random_seed,
            k=(
                2.25 if topology_backbone_network else 1.15
            ) / np.sqrt(run_config.n_lasers),
            iterations=500 if topology_backbone_network else 250,
            threshold=1.0e-5,
            scale=1.0,
        )
    detuning_colorbar_min_ghz = float(np.min(node_detunings_ghz))
    detuning_colorbar_max_ghz = float(np.max(node_detunings_ghz))
    if np.isclose(
        detuning_colorbar_min_ghz, detuning_colorbar_max_ghz
    ):
        detuning_colorbar_min_ghz -= 0.5
        detuning_colorbar_max_ghz += 0.5
    detuning_normalization = Normalize(
        vmin=detuning_colorbar_min_ghz,
        vmax=detuning_colorbar_max_ghz,
    )
    detuning_colormap = plt.get_cmap("bwr")
    edge_phase_colormap = plt.get_cmap("twilight")
    edge_phase_normalization = Normalize(vmin=-np.pi, vmax=np.pi)
    maximum_coupling = max(
        (data["weight"] for _, _, data in graph.edges(data=True)),
        default=1.0,
    )
    minimum_coupling = min(
        (data["weight"] for _, _, data in graph.edges(data=True)),
        default=0.0,
    )
    coupling_span = max(maximum_coupling - minimum_coupling, 1.0e-12)

    def displayed_coupling_fraction(weight: float) -> float:
        """Normalize active link strength for a readable topology stroke."""
        if topology_backbone_network:
            # The topology view benefits from contrast rather than absolute
            # scale: map the active design's weakest and strongest links to
            # the full visual range, even when their physical values differ
            # only slightly.
            return float(
                np.clip((weight - minimum_coupling) / coupling_span, 0.0, 1.0)
            )
        return float(weight / maximum_coupling)
    all_edges = list(graph.edges(data=True))
    edge_data_by_direction = {
        (source, receiver): data
        for source, receiver, data in all_edges
    }
    if topology_backbone_network:
        if graph.is_directed():
            highlighted_edges = [
                (source, receiver, data)
                for source, receiver, data in all_edges
                if (source, receiver) in topology_directions
            ]
        else:
            topology_pairs = {
                frozenset((source, receiver))
                for source, receiver in topology_directions
            }
            highlighted_edges = [
                (source, receiver, data)
                for source, receiver, data in all_edges
                if frozenset((source, receiver)) in topology_pairs
            ]
    elif run_config.n_lasers <= 4:
        # Small networks remain legible when every direction is shown.
        highlighted_edges = all_edges
    elif graph.is_directed():
        # For a large directed network, emphasize at most 2*M links. Begin
        # with one direction from every edge in the maximum reciprocal-
        # strength spanning tree, which keeps the visible summary connected.
        highlighted_directions: set[tuple[int, int]] = set()
        for first, second in maximum_spanning_forest.edges():
            forward = (first, second)
            reverse = (second, first)
            forward_weight = float(
                edge_data_by_direction.get(forward, {}).get("weight", -1.0)
            )
            reverse_weight = float(
                edge_data_by_direction.get(reverse, {}).get("weight", -1.0)
            )
            highlighted_directions.add(
                forward if forward_weight >= reverse_weight else reverse
            )

        # Also retain each laser's strongest outgoing influence so no source
        # disappears from the directed summary.
        for source in range(run_config.n_lasers):
            outgoing = [
                (receiver, float(data["weight"]))
                for (edge_source, receiver), data
                in edge_data_by_direction.items()
                if edge_source == source
            ]
            if outgoing:
                strongest_receiver = max(
                    outgoing, key=lambda item: item[1]
                )[0]
                highlighted_directions.add((source, strongest_receiver))

        visible_edge_limit = max(
            2 * run_config.n_lasers,
            len(highlighted_directions),
        )
        remaining_directions = sorted(
            (
                (float(data["weight"]), source, receiver)
                for (source, receiver), data
                in edge_data_by_direction.items()
                if (source, receiver) not in highlighted_directions
            ),
            reverse=True,
        )
        for _, source, receiver in remaining_directions:
            if len(highlighted_directions) >= visible_edge_limit:
                break
            highlighted_directions.add((source, receiver))
        highlighted_edges = [
            (source, receiver, data)
            for source, receiver, data in all_edges
            if (source, receiver) in highlighted_directions
        ]
    else:
        # For a large symmetric network, retain the connected maximum-
        # strength backbone and then the strongest remaining pairs.
        highlighted_pairs = {
            frozenset((source, receiver))
            for source, receiver in maximum_spanning_forest.edges()
        }
        visible_edge_limit = 2 * run_config.n_lasers
        remaining_pairs = sorted(
            (
                (float(data["weight"]), source, receiver)
                for source, receiver, data in all_edges
                if frozenset((source, receiver))
                not in highlighted_pairs
            ),
            reverse=True,
        )
        for _, source, receiver in remaining_pairs:
            if len(highlighted_pairs) >= visible_edge_limit:
                break
            highlighted_pairs.add(frozenset((source, receiver)))
        highlighted_edges = [
            (source, receiver, data)
            for source, receiver, data in all_edges
            if frozenset((source, receiver)) in highlighted_pairs
        ]
    node_size_scale = min(
        1.0,
        10.0 / run_config.n_lasers,
    )
    # Keep node area fixed so coupling strength has two unambiguous visual
    # encodings: pair separation and directed edge width.
    node_sizes = np.full(
        run_config.n_lasers,
        node_size_scale * 620.0,
    )
    # First draw the complete coupling network as context.
    # Backbone plots can include hundreds of displayed links.  Use a much
    # thinner stroke in that summary view so individual topology and nodes
    # remain visible; the coupling matrices retain the full quantitative view.
    if topology_backbone_network:
        background_width_offset = 0.04
        background_width_scale = 0.86
        foreground_width_offset = 0.12
        foreground_width_scale = 1.85
    else:
        background_width_offset = 0.12
        background_width_scale = 0.55
        foreground_width_offset = 0.65
        foreground_width_scale = 4.85

    if arc_layout:
        def draw_detuning_arcs(
            edges: list[tuple[int, int, dict]],
            *,
            colors: str | list,
            alpha: float,
            width_offset: float,
            width_scale: float,
            draw_arrows: bool = False,
        ) -> None:
            """Draw directed quadratic arcs without thousands of arrowheads."""
            curve_points = np.linspace(0.0, 1.0, 48)
            segments = []
            widths = []
            arrow_stride = max(1, int(np.ceil(len(edges) / 200)))
            for edge_index, (source, receiver, data) in enumerate(edges):
                start = positions[source]
                end = positions[receiver]
                horizontal_span = abs(end[0] - start[0])
                # Reversing source and receiver flips the arc side, making
                # reciprocal directions separable while keeping the matrix
                # panels as the precise directional reference.
                sign = 1.0 if end[0] >= start[0] else -1.0
                control = np.array(
                    [
                        0.5 * (start[0] + end[0]),
                        sign * (0.035 + 0.43 * horizontal_span),
                    ]
                )
                t = curve_points[:, None]
                segments.append(
                    (1.0 - t) ** 2 * start
                    + 2.0 * (1.0 - t) * t * control
                    + t**2 * end
                )
                widths.append(
                    width_offset
                    + width_scale
                    * displayed_coupling_fraction(float(data["weight"]))
                )
                if draw_arrows and edge_index % arrow_stride == 0:
                    tail_time, head_time = 0.72, 0.82
                    tail = (
                        (1.0 - tail_time) ** 2 * start
                        + 2.0 * (1.0 - tail_time) * tail_time * control
                        + tail_time**2 * end
                    )
                    head = (
                        (1.0 - head_time) ** 2 * start
                        + 2.0 * (1.0 - head_time) * head_time * control
                        + head_time**2 * end
                    )
                    arrow_color = (
                        colors[edge_index]
                        if isinstance(colors, list)
                        else colors
                    )
                    axes[3].add_patch(
                        FancyArrowPatch(
                            tail,
                            head,
                            arrowstyle="-|>",
                            mutation_scale=14.0,
                            linewidth=0.0,
                            color=arrow_color,
                            alpha=alpha,
                            zorder=2,
                        )
                    )
            if not segments:
                return
            axes[3].add_collection(
                LineCollection(
                    segments,
                    colors=colors,
                    linewidths=widths,
                    alpha=alpha,
                    capstyle="round",
                    zorder=1,
                )
            )

        draw_detuning_arcs(
            all_edges,
            colors="0.48",
            alpha=0.055,
            width_offset=0.035,
            width_scale=0.68,
        )
        draw_detuning_arcs(
            highlighted_edges,
            colors=[
                edge_phase_colormap(
                    edge_phase_normalization(
                        np.angle(
                            np.exp(1.0j * phi_p_rad[receiver, source])
                        )
                    )
                )
                for source, receiver, _ in highlighted_edges
            ],
            alpha=0.78,
            width_offset=0.10,
            width_scale=1.65,
            draw_arrows=True,
        )
    elif topology_sparsity_mask:
        active_directions = {
            (source, receiver) for source, receiver, _ in all_edges
        }
        all_directions = [
            (source, receiver)
            for source in range(run_config.n_lasers)
            for receiver in range(run_config.n_lasers)
            if source != receiver
        ]
        removed_directions = [
            direction
            for direction in all_directions
            if direction not in active_directions
        ]
        # Draw retained links first, then keep the comparatively rare removed
        # links on top.  Otherwise the retained layer can completely obscure
        # the red mask in near-dense designs.
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axes[3],
            edgelist=list(active_directions),
            width=0.08,
            edge_color="0.48",
            alpha=0.045,
            arrows=False,
            node_size=node_sizes,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axes[3],
            edgelist=removed_directions,
            width=0.42,
            edge_color="tab:red",
            alpha=0.78,
            arrows=False,
            node_size=node_sizes,
        )
    else:
        background_options = {
            "ax": axes[3],
            "edgelist": [
                (source, receiver) for source, receiver, _ in all_edges
            ],
            "width": [
                background_width_offset
                + background_width_scale
                * displayed_coupling_fraction(float(data["weight"]))
                for _, _, data in all_edges
            ],
            "edge_color": (
                "0.30"
                if cluster_layout
                else [
                    edge_phase_colormap(
                        edge_phase_normalization(
                            np.angle(
                                np.exp(
                                    1.0j * phi_p_rad[receiver, source]
                                )
                            )
                        )
                    )
                    for source, receiver, _ in all_edges
                ]
            ),
            "alpha": (
                0.025
                if cluster_layout
                else (0.035 if run_config.n_lasers > 4 else 0.08)
            ),
            "node_size": node_sizes,
            "arrows": False,
        }
        nx.draw_networkx_edges(graph, positions, **background_options)

        # Overlay the strong-coupling backbone using the heatmap's exact colors.
        foreground_options = {
            "ax": axes[3],
            "edgelist": [
                (source, receiver)
                for source, receiver, _ in highlighted_edges
            ],
            "width": [
                foreground_width_offset
                + foreground_width_scale
                * displayed_coupling_fraction(float(data["weight"]))
                for _, _, data in highlighted_edges
            ],
            "edge_color": [
                edge_phase_colormap(
                    edge_phase_normalization(
                        np.angle(
                            np.exp(
                                1.0j * phi_p_rad[receiver, source]
                            )
                        )
                    )
                )
                for source, receiver, _ in highlighted_edges
            ],
            "alpha": 0.70 if cluster_layout else 0.85,
            "node_size": node_sizes,
        }
        if not graph.is_directed() or cluster_layout:
            foreground_options["arrows"] = False
            nx.draw_networkx_edges(graph, positions, **foreground_options)
        else:
            # Draw directed links individually.  Reciprocal arrows use the same
            # signed curvature: reversing their endpoints then places the two
            # arrows on opposite physical sides of the node pair.  The previous
            # opposite signs caused both directions to occupy the same arc.
            highlighted_directions = {
                (source, receiver)
                for source, receiver, _ in highlighted_edges
            }
            for edge_index, (source, receiver, _) in enumerate(
                highlighted_edges
            ):
                reciprocal = (receiver, source) in highlighted_directions
                curvature = (
                    0.24
                    if reciprocal
                    else (0.07 if source < receiver else -0.07)
                )
                directed_options = dict(foreground_options)
                directed_options.update(
                    edgelist=[foreground_options["edgelist"][edge_index]],
                    width=[foreground_options["width"][edge_index]],
                    edge_color=[foreground_options["edge_color"][edge_index]],
                    arrows=True,
                    arrowsize=14,
                    connectionstyle=f"arc3,rad={curvature}",
                    min_source_margin=2,
                    min_target_margin=2,
                )
                nx.draw_networkx_edges(graph, positions, **directed_options)

    if cluster_layout:
        # Delineate the requested phase-cluster wedges without covering any
        # links.  The node radius itself encodes detuning.
        for center, half_width in zip(cluster_centers, cluster_half_widths):
            angles = np.linspace(center - half_width, center + half_width, 96)
            axes[3].plot(
                1.06 * np.cos(angles),
                1.06 * np.sin(angles),
                color="0.55",
                linewidth=0.55,
                linestyle="--",
                alpha=0.65,
                zorder=0,
            )

    # Draw nodes last so that strong edges never obscure their labels.
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes[3],
        node_size=node_sizes,
        node_color=node_detunings_ghz,
        cmap=detuning_colormap,
        vmin=detuning_normalization.vmin,
        vmax=detuning_normalization.vmax,
        edgecolors="0.25",
        linewidths=0.8,
    )
    if topology_backbone_network and run_config.n_lasers > 30:
        # Large arrays become unreadable when every small node carries an
        # index. Label one hub in each of the six largest detected communities
        # plus the minimum- and maximum-detuning nodes.
        labeled_nodes: set[int] = set()
        if layout_graph.number_of_edges() > 0:
            communities = sorted(
                nx.community.greedy_modularity_communities(
                    layout_graph,
                    weight="layout_weight",
                ),
                key=len,
                reverse=True,
            )
            weighted_degree = dict(
                layout_graph.degree(weight="layout_weight")
            )
            for community in communities[:6]:
                labeled_nodes.add(
                    int(
                        max(
                            community,
                            key=lambda node: weighted_degree[node],
                        )
                    )
                )
        labeled_nodes.update(
            {
                int(np.argmin(node_detunings_ghz)),
                int(np.argmax(node_detunings_ghz)),
            }
        )
    else:
        labeled_nodes = set(range(run_config.n_lasers))
    for node in sorted(labeled_nodes):
        detuning_ghz = node_detunings_ghz[node]
        red, green, blue, _ = detuning_colormap(
            detuning_normalization(detuning_ghz)
        )
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axes[3].text(
            positions[node][0],
            positions[node][1],
            str(node + 1),
            ha="center",
            va="center",
            color="white" if luminance < 0.52 else "0.15",
            fontsize=10,
            zorder=4,
        )
    axes[3].set_title(
        "Detuning-ordered coupling arcs"
        if arc_layout
        else (
            "All-to-all sparsity mask"
            if topology_sparsity_mask
            else (
            "Cluster-aware relative-coupling backbone"
            if topology_backbone_network
            else "Coupling-topology layout"
            )
        ),
        pad=18 if arc_layout else None,
    )
    if arc_layout:
        if arc_layout_curvature <= 0.0:
            detuning_geometry_caption = "left-to-right row = detuning order"
        elif arc_layout_curvature >= 1.0:
            detuning_geometry_caption = "lower semicircle = detuning order"
        else:
            detuning_geometry_caption = (
                "row-to-semicircle curve = detuning order"
            )
        network_caption = (
            f"all {len(all_edges)} retained directed links; "
            f"overlay: {len(highlighted_edges)} strongest"
            + "\n"
            + detuning_geometry_caption
            + "\n"
            + r"upper/lower = direction; width = normalized $\kappa_{ij}$; arrows = overlay"
        )
    elif topology_sparsity_mask:
        network_caption = (
            f"retained: {len(all_edges)}/{run_config.n_lasers * (run_config.n_lasers - 1)}; "
            f"removed: {run_config.n_lasers * (run_config.n_lasers - 1) - len(all_edges)}"
            "\n"
            "gray = retained link; red = removed link; "
            "angle = target-phase cluster; radius = detuning"
        )
    elif topology_backbone_network:
        network_caption = (
            f"colored overlay: {len(highlighted_edges)}/"
            f"{len(all_edges)} directed links; all links shown in gray"
            "\n"
            "angle = target-phase cluster; radius = detuning; "
            r"width $\propto\kappa_{ij}$"
        )
    else:
        network_caption = (
            f"showing {len(highlighted_edges)}/{len(all_edges)} strongest "
            "links; "
            r"width $\propto\kappa_{ij}$; color $=\phi_{p,ij}$"
            "\n"
            r"distance $\sim$ reciprocal $\kappa$ "
            "(not physical spacing)"
        )
    axes[3].text(
        0.5,
        -0.055,
        network_caption,
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=6.5 if arc_layout else 9,
        color="0.35",
    )
    # With two lasers the circular layout puts both nodes on a vertical
    # diameter.  Matplotlib otherwise collapses the x limits to a single
    # value, clipping the nodes and making the network panel appear empty.
    if arc_layout:
        # Geometry morphs from a row at M=3 to the lower semicircle at M=20.
        axes[3].set_xlim(-1.16, 1.16)
        axes[3].set_ylim(
            -0.60 - 0.48 * arc_layout_curvature,
            0.60 + 0.10 * arc_layout_curvature,
        )
    else:
        axes[3].set_xlim(-1.2, 1.2)
        axes[3].set_ylim(-1.2, 1.2)
    axes[3].set_aspect("equal")
    axes[3].set_axis_off()
    detuning_colorbar_kwargs = (
        {"cax": detuning_colorbar_axis}
        if vertical_layout
        else {"ax": axes[3], "fraction": 0.055, "pad": 0.02}
    )
    detuning_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=detuning_normalization,
            cmap=detuning_colormap,
        ),
        orientation="vertical",
        **detuning_colorbar_kwargs,
    )
    detuning_colorbar.set_label("detuning (GHz)")

    plt.savefig(
        figure_directory / f"selected_design_{desired_state}.png",
        dpi=300,
        bbox_inches="tight",
    )

    return figure


#  3. Detailed serial validation

def simulate_and_plot_best_design(
    design: dict,
    run_config: DesignerConfig,
    *,
    validation_delay_count: float = 1000.0,
    wavelength_m: float = 910.0e-9,
    desired_state: str = "splay",
    detuning_distribution_ghz: np.ndarray | None = None,
    legend_fontsize: float = 8.0,
    noise_iterations: int = 1,
    separate_total_power_panel: bool = False,
    dual_axis_power_panel: bool = False,
    far_field_panel: bool = False,
    emitter_spacing_m: float = 10.0e-6,
    far_field_theta_range_deg: tuple[float, float] = (-30.0, 30.0),
    far_field_n_theta: int = 801,
    far_field_max_time_points: int = 2000,
    far_field_element_fwhm_deg: float | None = 20.0,
    far_field_final_slice_panel: bool = False,
    far_field_slice_convolution_fwhm_deg: float | None = 0.5,
    precomputed_simulation: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    transition_time_ns: float | tuple[float, ...] | None = None,
    allowable_phase_error_deg: float | None = None,
) -> dict:
    """Run free-running validation cases and plot their mean and spread.

    ``detuning_distribution_ghz`` optionally fixes one detuning value per
    laser.  If omitted, the simulator samples the configured uniform
    distribution. Repeated cases share the same design and detunings but
    receive independent simulator noise.
    """
    if noise_iterations < 1:
        raise ValueError("noise_iterations must be at least 1")
    if (
        allowable_phase_error_deg is not None
        and not 0.0 < allowable_phase_error_deg <= 180.0
    ):
        raise ValueError(
            "allowable_phase_error_deg must lie in (0, 180] when supplied"
        )
    if separate_total_power_panel and dual_axis_power_panel:
        raise ValueError(
            "separate_total_power_panel and dual_axis_power_panel cannot "
            "both be enabled"
        )
    if (
        far_field_slice_convolution_fwhm_deg is not None
        and far_field_slice_convolution_fwhm_deg <= 0.0
    ):
        raise ValueError(
            "far_field_slice_convolution_fwhm_deg must be positive when "
            "supplied"
        )
    validation_config = replace(
        run_config,
        simulation_time_seconds=(
            validation_delay_count * run_config.delay_seconds
        ),
        n_jobs=1,
    )
    kappa_per_ns = np.repeat(
        np.asarray(design["kappa_per_ns"])[None, :, :],
        noise_iterations,
        axis=0,
    )
    phi_p_rad = np.repeat(
        np.asarray(design["phi_p_rad"])[None, :, :],
        noise_iterations,
        axis=0,
    )
    simulation_detunings = detuning_distribution_ghz
    if simulation_detunings is not None:
        simulation_detunings = np.asarray(
            simulation_detunings, dtype=float
        )
        if simulation_detunings.ndim == 1:
            simulation_detunings = np.repeat(
                simulation_detunings[None, :],
                noise_iterations,
                axis=0,
            )

    if precomputed_simulation is None:
        time_seconds, states, frequencies_nd, initial_frequency_ghz = (
            simulate_with_vcsel(
                kappa_per_ns,
                phi_p_rad,
                validation_config,
                progress=True,
                smooth_frequencies=True,
                detuning_distribution_ghz=simulation_detunings,
            )
        )
    else:
        (
            time_seconds,
            states,
            frequencies_nd,
            initial_frequency_ghz,
        ) = precomputed_simulation

    photons = states[:, 1::3]
    phases_rad = states[:, 2::3]
    relative_phases_rad = np.angle(
        np.exp(1j * (phases_rad - phases_rad[:, :1]))
    )
    frequencies_ghz = frequencies_nd / (
        2.0
        * np.pi
        * run_config.photon_lifetime_seconds
        * 1.0e9
    )

    # Restore explicit free-running frequencies over the history segment.
    initial_saved = initial_frequency_ghz[
        :, :, :: validation_config.save_every
    ]
    history_points = min(
        initial_saved.shape[2], frequencies_ghz.shape[2]
    )
    frequencies_ghz[:, :, :history_points] = initial_saved[
        :, :, :history_points
    ]

    requested_targets = np.repeat(
        np.asarray(design["target_phases_rad"])[None, :],
        noise_iterations,
        axis=0,
    )
    reward_result = calculate_phase_reward(
        time_seconds,
        states,
        requested_targets,
        validation_config,
    )
    phase_rewards = reward_result[0]
    orientations = reward_result[5]

    # Align complete conjugate solutions before averaging their phases.
    aligned_relative_phases = np.angle(
        np.exp(
            1j
            * orientations[:, None, None]
            * relative_phases_rad
        )
    )
    phase_mean = np.angle(
        np.mean(np.exp(1j * aligned_relative_phases), axis=0)
    )
    phase_deviation = np.angle(
        np.exp(1j * (aligned_relative_phases - phase_mean[None, :, :]))
    )
    phase_std = np.std(phase_deviation, axis=0)
    frequency_mean = np.mean(frequencies_ghz, axis=0)
    frequency_std = np.std(frequencies_ghz, axis=0)

    optical_angular_frequency = 2.0 * np.pi * c / wavelength_m
    intensity_to_mw = (
        1.0e3
        * hbar
        * optical_angular_frequency
        / (
            run_config.gain_per_second
            * run_config.carrier_lifetime_seconds
            * run_config.photon_lifetime_seconds
        )
    )
    laser_power_mw = photons * intensity_to_mw
    total_field = np.sum(
        np.sqrt(np.maximum(photons, 0.0)) * np.exp(1j * phases_rad),
        axis=1,
    )
    total_power_cases_mw = np.abs(total_field) ** 2 * intensity_to_mw
    laser_power_mean = np.mean(laser_power_mw, axis=0)
    laser_power_std = np.std(laser_power_mw, axis=0)
    total_power_mean = np.mean(total_power_cases_mw, axis=0)
    total_power_std = np.std(total_power_cases_mw, axis=0)
    time_ns = time_seconds * 1.0e9

    matched_target_rad = make_relative_to_laser_1(
        np.asarray(design["target_phases_rad"])
    )
    target_phase_error = (
        aligned_relative_phases - matched_target_rad[None, :, None]
    )
    target_order_parameter_cases = np.abs(
        np.mean(np.exp(1j * target_phase_error), axis=1)
    )
    target_order_parameter_mean = np.mean(
        target_order_parameter_cases, axis=0
    )
    target_order_parameter_std = np.std(
        target_order_parameter_cases, axis=0
    )
    show_uncertainty = noise_iterations > 1

    frequency_axis_index = 0
    phase_axis_index = 1
    order_parameter_axis_index = 2
    individual_power_axis_index = 3
    power_panel_count = 5 if separate_total_power_panel else 4
    panel_count = power_panel_count + int(far_field_panel)
    far_field_axis_index = panel_count - 1 if far_field_panel else None
    panel_height_ratios = [1.0, 1.0, 0.45] + [1.0] * (panel_count - 3)
    far_field_colorbar_axis = None
    far_field_slice_axis = None
    if far_field_panel:
        figure = plt.figure(
            figsize=(
                17 if far_field_final_slice_panel else 14,
                17 if not separate_total_power_panel else 20,
            ),
            constrained_layout=True,
        )
        if far_field_final_slice_panel:
            grid = figure.add_gridspec(
                panel_count,
                3,
                width_ratios=(1.0, 0.22, 0.025),
                height_ratios=panel_height_ratios,
            )
        else:
            grid = figure.add_gridspec(
                panel_count,
                2,
                width_ratios=(1.0, 0.025),
                height_ratios=panel_height_ratios,
            )
        axes_list = []
        for panel_index in range(panel_count):
            shared_axis = axes_list[0] if axes_list else None
            plot_columns = (
                grid[panel_index, 0]
                if not far_field_final_slice_panel
                or panel_index == far_field_axis_index
                else grid[panel_index, :2]
            )
            axes_list.append(
                figure.add_subplot(plot_columns, sharex=shared_axis)
            )
        axes = np.asarray(axes_list)
        if far_field_final_slice_panel:
            far_field_slice_axis = figure.add_subplot(
                grid[-1, 1], sharey=axes[far_field_axis_index]
            )
            far_field_colorbar_axis = figure.add_subplot(grid[-1, 2])
        else:
            far_field_colorbar_axis = figure.add_subplot(grid[-1, 1])
        for axis in axes[:-1]:
            axis.tick_params(labelbottom=False)
    else:
        figure, axes = plt.subplots(
            panel_count,
            1,
            figsize=(14, 16 if separate_total_power_panel else 13),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": panel_height_ratios},
        )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Manual clustered targets assign the same target phase to multiple
    # lasers.  Plot their guide and tolerance region once per distinct wrapped
    # target, rather than stacking identical bands for every member of a
    # cluster.  This also handles an equivalent -pi / +pi representation.
    target_cluster_representatives: list[int] = []
    for laser, target_phase in enumerate(matched_target_rad):
        if all(
            not np.isclose(
                np.angle(
                    np.exp(
                        1.0j
                        * (target_phase - matched_target_rad[representative])
                    )
                ),
                0.0,
                atol=1.0e-8,
            )
            for representative in target_cluster_representatives
        ):
            target_cluster_representatives.append(laser)
    target_cluster_representative_set = set(target_cluster_representatives)

    for laser in range(run_config.n_lasers):
        color = colors[laser % len(colors)]
        axes[frequency_axis_index].plot(
            time_ns,
            frequency_mean[laser],
            color=color,
            label=rf"$\dot{{\phi}}_{{{laser + 1}}}$",
        )
        if show_uncertainty:
            axes[frequency_axis_index].fill_between(
                time_ns,
                frequency_mean[laser] - frequency_std[laser],
                frequency_mean[laser] + frequency_std[laser],
                color=color,
                alpha=0.20,
            )
        axes[phase_axis_index].plot(
            time_ns,
            phase_mean[laser],
            color=color,
            label=rf"$\phi_{{{laser + 1}}}-\phi_1$",
        )
        if (
            allowable_phase_error_deg is not None
            and laser in target_cluster_representative_set
        ):
            half_width = np.deg2rad(allowable_phase_error_deg)
            center = float(matched_target_rad[laser])
            lower = center - half_width
            upper = center + half_width
            label = (
                rf"$\pm {allowable_phase_error_deg:g}^\circ$ target range"
                if laser == target_cluster_representatives[0]
                else "_nolegend_"
            )
            intervals: list[tuple[float, float]]
            if half_width >= np.pi:
                intervals = [(-np.pi, np.pi)]
            elif lower < -np.pi:
                intervals = [(-np.pi, upper), (lower + 2.0 * np.pi, np.pi)]
            elif upper > np.pi:
                intervals = [(lower, np.pi), (-np.pi, upper - 2.0 * np.pi)]
            else:
                intervals = [(lower, upper)]
            for interval_index, (interval_lower, interval_upper) in enumerate(
                intervals
            ):
                axes[phase_axis_index].axhspan(
                    interval_lower,
                    interval_upper,
                    color=color,
                    alpha=0.12,
                    linewidth=0.0,
                    zorder=0,
                    label=label if interval_index == 0 else "_nolegend_",
                )
        if show_uncertainty:
            axes[phase_axis_index].fill_between(
                time_ns,
                phase_mean[laser] - phase_std[laser],
                phase_mean[laser] + phase_std[laser],
                color=color,
                alpha=0.20,
            )
        if laser in target_cluster_representative_set:
            axes[phase_axis_index].axhline(
                matched_target_rad[laser],
                color=color,
                linestyle="--",
                alpha=0.7,
            )
        axes[individual_power_axis_index].plot(
            time_ns,
            laser_power_mean[laser],
            color=color,
            # Keep the power legend readable for large arrays. The individual
            # traces remain visible; only their legend entries are suppressed.
            label=(
                rf"$P_{{{laser + 1}}}$"
                if run_config.n_lasers <= 25
                else "_nolegend_"
            ),
        )
        if show_uncertainty:
            axes[individual_power_axis_index].fill_between(
                time_ns,
                np.maximum(
                    laser_power_mean[laser] - laser_power_std[laser],
                    0.0,
                ),
                laser_power_mean[laser] + laser_power_std[laser],
                color=color,
                alpha=0.20,
            )

    axes[frequency_axis_index].set_ylabel(r"$\dot{\phi}$ (GHz)")
    axes[frequency_axis_index].set_title(
        "Standalone conditional design from a free-running initial state\n"
        f"phase reward={np.mean(phase_rewards):.3f} "
        f"$\\pm$ {np.std(phase_rewards):.3f} over "
        f"{noise_iterations} realization"
        f"{'s' if noise_iterations != 1 else ''}"
    )
    axes[phase_axis_index].set_ylabel(r"wrapped $\phi_i-\phi_1$")
    axes[phase_axis_index].set_ylim(-1.1*np.pi, 1.1*np.pi)
    axes[phase_axis_index].set_yticks(PHASE_TICKS)
    axes[phase_axis_index].set_yticklabels(PHASE_TICK_LABELS)
    axes[order_parameter_axis_index].plot(
        time_ns,
        target_order_parameter_mean,
        color="tab:blue",
        linewidth=2.0,
        label=r"$r_{\mathrm{target}}$",
    )
    if show_uncertainty:
        axes[order_parameter_axis_index].fill_between(
            time_ns,
            np.maximum(
                target_order_parameter_mean - target_order_parameter_std,
                0.0,
            ),
            np.minimum(
                target_order_parameter_mean + target_order_parameter_std,
                1.0,
            ),
            color="tab:blue",
            alpha=0.20,
        )
    axes[order_parameter_axis_index].axhline(
        0.98,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="0.98 threshold",
    )
    axes[order_parameter_axis_index].set(
        ylabel=r"$r_{\mathrm{target}}$",
        ylim=(0.0, 1.02),
        yticks=(0.0, 0.5, 1.0),
    )
    total_power_axis_index = 4 if separate_total_power_panel else 3
    total_power_axis = (
        axes[individual_power_axis_index].twinx()
        if dual_axis_power_panel
        else axes[total_power_axis_index]
    )
    total_power_axis.plot(
        time_ns,
        total_power_mean,
        color="black" if dual_axis_power_panel else "green",
        linewidth=1.5,
        linestyle="--" if dual_axis_power_panel else "-",
        label=r"$P_{\rm total}$"
    )
    if show_uncertainty:
        total_power_axis.fill_between(
            time_ns,
            np.maximum(total_power_mean - total_power_std, 0.0),
            total_power_mean + total_power_std,
            color="black" if dual_axis_power_panel else "green",
            alpha=0.25,
        )
    axes[individual_power_axis_index].set_ylabel(
        "individual output power (mW)"
        if separate_total_power_panel or dual_axis_power_panel
        else "output power (mW)"
    )
    axes[-1].set_xlabel("time (ns)")
    if separate_total_power_panel or dual_axis_power_panel:
        total_power_axis.set_ylabel("total output power (mW)")
    if dual_axis_power_panel:
        individual_power_minimum = float(
            np.min(
                np.maximum(laser_power_mean - laser_power_std, 0.0)
            )
            if show_uncertainty
            else np.min(laser_power_mean)
        )
        individual_power_maximum = float(
            np.max(laser_power_mean + laser_power_std)
            if show_uncertainty
            else np.max(laser_power_mean)
        )
        axes[individual_power_axis_index].set_ylim(
            0.95 * individual_power_minimum,
            1.05 * max(float(individual_power_maximum), 1.0e-12),
        )

    far_field_result = None
    if far_field_panel:
        from examples.visualization.far_field_intensity import (
            far_field_intensity_map,
        )

        theta_deg, far_field_time_indices, intensity_map = (
            far_field_intensity_map(
                S=photons,
                phi=phases_rad,
                lambda0=wavelength_m,
                d=emitter_spacing_m,
                theta_range_deg=far_field_theta_range_deg,
                n_theta=far_field_n_theta,
                max_time_points=far_field_max_time_points,
                element_fwhm_deg=far_field_element_fwhm_deg,
            )
        )
        # Match the display-only angular blur used by injection_steering:
        # convolve every time-resolved far-field slice, after the physical
        # element envelope has been applied.  The global renormalization keeps
        # the existing inferno color scale (0--1) unchanged.
        if far_field_slice_convolution_fwhm_deg is not None:
            angular_step_deg = float(np.median(np.diff(theta_deg)))
            gaussian_sigma_deg = (
                far_field_slice_convolution_fwhm_deg
                / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            )
            kernel_radius = min(
                int(np.ceil(4.0 * gaussian_sigma_deg / angular_step_deg)),
                len(theta_deg) - 1,
            )
            kernel_angles_deg = (
                np.arange(-kernel_radius, kernel_radius + 1)
                * angular_step_deg
            )
            gaussian_kernel = np.exp(
                -0.5 * (kernel_angles_deg / gaussian_sigma_deg) ** 2
            )
            gaussian_kernel /= np.sum(gaussian_kernel)
            padded_map = np.pad(
                intensity_map,
                ((kernel_radius, kernel_radius), (0, 0)),
                mode="edge",
            )
            intensity_map = np.apply_along_axis(
                lambda profile: np.convolve(
                    profile, gaussian_kernel, mode="valid"
                ),
                0,
                padded_map,
            )
            blurred_maximum = float(np.max(intensity_map))
            if blurred_maximum > 0.0:
                intensity_map /= blurred_maximum
        far_field_axis = axes[far_field_axis_index]
        far_field_image = far_field_axis.pcolormesh(
            time_ns[far_field_time_indices],
            theta_deg,
            intensity_map,
            shading="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        far_field_axis.set_ylabel(r"far-field angle $\theta$ (degrees)")
        far_field_axis.set_title(
            "Time-resolved normalized far-field intensity; "
            rf"$d={emitter_spacing_m * 1.0e6:.1f}\,\mu$m"
        )
        far_field_colorbar = figure.colorbar(
            far_field_image,
            cax=far_field_colorbar_axis,
        )
        far_field_colorbar.set_label("normalized intensity")
        final_intensity = intensity_map[:, -1].copy()
        convolved_final_intensity = final_intensity.copy()
        if far_field_slice_axis is not None:
            slice_title = "Final-time slice"
            if far_field_slice_convolution_fwhm_deg is not None:
                slice_title += (
                    "\nGaussian angular convolution, "
                    rf"FWHM={far_field_slice_convolution_fwhm_deg:g}$^\circ$"
                )
            far_field_slice_axis.plot(
                convolved_final_intensity,
                theta_deg,
                color="tab:orange",
                linewidth=2.0,
            )
            far_field_slice_axis.set(
                xlabel="normalized intensity",
                title=slice_title,
                xlim=(0.0, 1.05),
            )
            far_field_slice_axis.tick_params(labelleft=False)
            far_field_slice_axis.grid(alpha=0.25)
        far_field_result = {
            "theta_deg": theta_deg,
            "time_indices": far_field_time_indices,
            "intensity_normalized": intensity_map,
            "final_intensity_normalized": final_intensity,
            "final_intensity_convolved_normalized": (
                convolved_final_intensity
            ),
            "slice_convolution_fwhm_deg": (
                far_field_slice_convolution_fwhm_deg
            ),
        }

    transition_times_ns = (
        ()
        if transition_time_ns is None
        else tuple(np.atleast_1d(transition_time_ns).astype(float))
    )
    for axis_index, axis in enumerate(axes):
        for transition_time in transition_times_ns:
            axis.axvline(
                transition_time,
                color="black",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
            )
        if far_field_panel and axis_index == far_field_axis_index:
            continue
        axis.axvspan(
            0.0,
            run_config.coupling_ramp_start_delays
            * run_config.delay_seconds
            * 1.0e9,
            color="gray",
            alpha=0.2,
        )
        axis.grid(alpha=0.25)
        # Large arrays produce unreadable frequency/phase legends. Keep the
        # total-power legend on the power panel while hiding only those two
        # oversized legends when more than 25 lasers are plotted.
        if dual_axis_power_panel and axis_index == individual_power_axis_index:
            continue
        if (
            run_config.n_lasers <= 25
            or axis_index
            in (order_parameter_axis_index, total_power_axis_index)
        ):
            axis.legend(ncol=3, fontsize=legend_fontsize, loc="upper right")
    if dual_axis_power_panel:
        individual_handles, individual_labels = (
            axes[individual_power_axis_index].get_legend_handles_labels()
        )
        total_handles, total_labels = (
            total_power_axis.get_legend_handles_labels()
        )
        if run_config.n_lasers > 25:
            individual_handles, individual_labels = [], []
        power_legend = total_power_axis.legend(
            individual_handles + total_handles,
            individual_labels + total_labels,
            ncol=3,
            fontsize=legend_fontsize,
            loc="best",
            framealpha=0.8,
            facecolor="white",
        )
        power_legend.set_zorder(100)
    plt.savefig(
        figure_directory / f"selected_design_dynamics_{desired_state}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return {
        "time_seconds": time_seconds,
        "states": states,
        "frequencies_nondimensional": frequencies_nd,
        "phase_rewards": phase_rewards,
        "frequency_mean_ghz": frequency_mean,
        "frequency_std_ghz": frequency_std,
        "phase_mean_rad": phase_mean,
        "phase_std_rad": phase_std,
        "target_order_parameter_cases": target_order_parameter_cases,
        "target_order_parameter_mean": target_order_parameter_mean,
        "target_order_parameter_std": target_order_parameter_std,
        "laser_power_mean_mw": laser_power_mean,
        "laser_power_std_mw": laser_power_std,
        "total_power_mean_mw": total_power_mean,
        "total_power_std_mw": total_power_std,
        "far_field": far_field_result,
        "figure": figure,
    }


#  4. Complete workflow

def main(
    run_config: DesignerConfig | None = None,
    *,
    train_new_model: bool = TRAIN_NEW_MODEL,
) -> dict:
    """Train/load, validate, design, and run one detailed simulation."""
    active_config = config if run_config is None else run_config
    torch.manual_seed(active_config.random_seed)
    policy = make_policy(active_config).to(
        torch.device(active_config.device)
    )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=active_config.learning_rate
    )

    if RUN_ARCHITECTURE_SANITY_CHECKS:
        run_architecture_sanity_checks(active_config)

    if train_new_model:
        policy, optimizer, history = train_reinforce(
            active_config,
            policy,
            optimizer,
            live_plot=True,
        )
        checkpoint_path = Path(active_config.best_checkpoint_file)
    else:
        checkpoint_path = Path(active_config.best_checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_path} does not exist. Set "
                "train_new_model=True and run once."
            )
        history = None

    policy, optimizer, loaded_config, checkpoint_metadata = load_checkpoint(
        checkpoint_path,
        device=active_config.device,
    )
    # n_jobs controls execution only; retain the requested value even when an
    # older checkpoint predates the multiprocessing configuration field.
    active_config = replace(
        loaded_config,
        # This may be a training size or an unseen zero-shot size.
        n_lasers=active_config.n_lasers,
        n_jobs=active_config.n_jobs,
        jupyter_mode=active_config.jupyter_mode,
    )
    print(
        f"Loaded iteration {checkpoint_metadata['iteration']} with held-out "
        f"phase reward {checkpoint_metadata['held_out_reward']:.3f}"
    )

    zero_shot_validation = None
    if RUN_ZERO_SHOT_VALIDATION:
        zero_shot_validation = validate_policy_sizes(
            policy,
            active_config,
            ZERO_SHOT_N_LASERS,
            candidates_per_target=64,
        )

    if history is not None:
        plot_training_history(history)
        plt.show()

    # validation_results = validate_policy(
    #     policy,
    #     active_config,
    #     candidates_per_target=64,
    # )

    # Replace this target with any vector of length active_config.n_lasers.
    desired_state = 'in-phase'
    requested_phases_rad = named_phase_targets(
        active_config.n_lasers
    )[desired_state]
    if MANUAL_DETUNING_DISTRIBUTION_GHZ is None:
        design_detuning_distribution_ghz = sample_detuning_distributions(
            1,
            np.random.default_rng(active_config.random_seed + 30_000),
            active_config,
        )[0]
    else:
        design_detuning_distribution_ghz = np.asarray(
            MANUAL_DETUNING_DISTRIBUTION_GHZ,
            dtype=float,
        )
    if active_config.enable_connected_edge_budgets:
        best_design = design_sparsest_for_target(
            policy,
            requested_phases_rad,
            active_config,
            minimum_phase_reward=0.98,
            number_of_candidates=100,
            detuning_distribution_ghz=design_detuning_distribution_ghz,
        )
        print(
            "Selected active links: "
            f"{best_design['active_link_count']} | "
            "phase target satisfied: "
            f"{best_design['phase_target_satisfied']}"
        )
    else:
        best_design = design_for_target(
            policy,
            requested_phases_rad,
            active_config,
            number_of_candidates=100,
            top_k=1,
            detuning_distribution_ghz=design_detuning_distribution_ghz,
        )[0]
    print(f"Phase reward: {best_design['phase_reward']:.3f}")
    print(
        "Magnitude symmetry penalty: "
        f"{best_design['magnitude_symmetry_penalty']:.3f}"
    )
    print("kappa (ns^-1):")
    print(np.round(best_design["kappa_per_ns"], 3))
    print("phi_p (rad):")
    print(np.round(best_design["phi_p_rad"], 3))

    run_label = FILE_SUFFIX.strip("_")
    output_label = (
        f"{desired_state}_{run_label}" if run_label else desired_state
    )
    plot_selected_design(best_design, active_config, output_label)
    plt.show()
    simulation_result = simulate_and_plot_best_design(
    best_design,
        replace(
        active_config,
        noise_amplitude=0.0,
        coupling_ramp_rise_delays=50.0,
        detuning_span_ghz=0.2,
        time_step_seconds = 1.0e-12
    ),
        validation_delay_count=200.0,
        desired_state=output_label,
        # Use the same detuning ordering that was used to generate the
        # returned coupling matrices.
        detuning_distribution_ghz=best_design[
            "detuning_distribution_ghz"
        ],
    )
    return {
        "policy": policy,
        "optimizer": optimizer,
        "config": active_config,
        "history": history,
        "zero_shot_validation": zero_shot_validation,
        # "validation_results": validation_results,
        "best_design": best_design,
        "simulation_result": simulation_result,
    }


if __name__ == "__main__":
    results = main()
