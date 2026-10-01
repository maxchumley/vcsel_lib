#%%
"""Run the standalone conditional coupling designer.

All executable work is inside :func:`main`, which is required for spawn-safe
multiprocessing on macOS and Windows. Importing this module never trains a
model or creates worker processes.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

from rl.conditional_coupling_designer import (
    calculate_phase_reward,
    DesignerConfig,
    PHASE_TICK_LABELS,
    PHASE_TICKS,
    PolicyNetwork,
    design_for_target,
    load_checkpoint,
    make_relative_to_laser_1,
    named_phase_targets,
    plot_training_history,
    sample_detuning_distributions,
    simulate_with_vcsel,
    train_reinforce,
    validate_policy,
)
from rl.paths import RESULTS_DIR

figure_directory = RESULTS_DIR

#  1. Settings

# Multiprocessing is target-level. Each worker receives one or more targets
# and keeps every candidate for those targets in one vectorized integration.
config = DesignerConfig(
    n_lasers=2,
    training_iterations=2500,
    targets_per_batch=14,
    candidates_per_target=64,
    hidden_sizes=(256, 256, 128),
    detuning_hidden_sizes=(64, 64),
    edge_hidden_sizes=(128, 128),
    learning_rate=1.0e-3,
    # Use larger updates for learning, then smaller updates for fine-tuning.
    learning_rate_switch_iteration=1500,
    learning_rate_after_switch=1.0e-4,
    # Symmetry can be selected independently for magnitudes and phases.
    force_symmetric_kappa=False,
    force_symmetric_phi_p=False,
    # Set the switch True to lower the learned log-std ceiling on this schedule.
    initial_log_std=-0.25,
    minimum_log_std=-3.0,
    maximum_log_std=0.75,
    enable_exploration_annealing=True,
    exploration_anneal_start_iteration=1000,
    exploration_anneal_iterations=500,
    final_maximum_log_std=-1.5,
    maximum_kappa_per_ns=100.0,
    # Softly encourage reciprocal coupling magnitudes from iteration 1.
    magnitude_symmetry_weight=0.005,
    held_out_target_count=64,
    validation_interval=10,
    random_seed=7,
    device="cpu",
    n_jobs=os.cpu_count(),
    jupyter_mode=True,
    initial_kappa_fraction = 0.05,
    detuning_span_ghz=10.0,
    # Hold a narrow random ±0.05 GHz span through iteration 200, then grow
    # the random span to ±5 GHz at iteration 1500.
    detuning_curriculum_initial_half_span_ghz=0.05,
    detuning_curriculum_warmup_iterations=200,
    detuning_curriculum_iterations=1000,
    simulation_time_seconds=500e-9
)

TRAIN_NEW_MODEL = True
# Optional label appended before ``.pt`` (for example, ``"_wide_kappa"``).
# Leave empty to use the standard laser-count checkpoint names.
FILE_SUFFIX = "_asymmetric_kappa_asymmetric_phi"
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


config = _apply_checkpoint_suffix(config, FILE_SUFFIX)


#  2. Matrix and settled-phase summary

def plot_selected_design(
    design: dict,
    run_config: DesignerConfig,
    desired_state: str = "splay",
    *,
    vertical_layout: bool = False,
) -> plt.Figure:
    """Plot one selected design with one-based laser labels."""
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
    kappa_image = axes[0].imshow(
        np.asarray(design["kappa_per_ns"]).T,
        origin="upper",
        cmap="Blues",
        vmin=0.0,
        vmax=run_config.maximum_kappa_per_ns,
    )
    axes[0].set(
        title=r"$\kappa$ (ns$^{-1}$)",
        xlabel="receiver",
        ylabel="source",
    )
    kappa_colorbar_kwargs = (
        {"cax": colorbar_axes[0]} if vertical_layout else {"ax": axes[0]}
    )
    figure.colorbar(kappa_image, **kappa_colorbar_kwargs)

    # Coupling-phase matrix
    phase_image = axes[1].imshow(
        np.asarray(design["phi_p_rad"]).T,
        origin="upper",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set(
        title=r"$\phi_p$",
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
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    # Coupling network. Matrix entry [receiver, source] represents the edge
    # source -> receiver, and edge width is proportional to kappa.
    kappa_per_ns = np.asarray(design["kappa_per_ns"], dtype=float)
    symmetric_kappa = np.allclose(
        kappa_per_ns,
        kappa_per_ns.T,
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    graph = nx.Graph() if symmetric_kappa else nx.DiGraph()
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

    positions = nx.circular_layout(graph)
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
    maximum_coupling = max(
        (data["weight"] for _, _, data in graph.edges(data=True)),
        default=1.0,
    )
    edge_widths = [
        4.0 * data["weight"] / maximum_coupling
        for _, _, data in graph.edges(data=True)
    ]
    # Use the heatmap's exact colormap and normalization so each edge color
    # matches the kappa matrix cell with the same coupling magnitude.
    edge_colors = [
        kappa_image.cmap(kappa_image.norm(data["weight"]))
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axes[3],
        node_size=520,
        node_color=node_detunings_ghz,
        cmap=detuning_colormap,
        vmin=detuning_normalization.vmin,
        vmax=detuning_normalization.vmax,
        edgecolors="0.25",
        linewidths=0.8,
    )
    for node, detuning_ghz in enumerate(node_detunings_ghz):
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
    edge_drawing_options = {
        "ax": axes[3],
        "width": edge_widths,
        "edge_color": edge_colors,
        "alpha": 0.65,
        "node_size": 520,
    }
    if symmetric_kappa:
        edge_drawing_options["arrows"] = False
    else:
        edge_drawing_options.update(
            arrows=True,
            arrowsize=12,
            connectionstyle="arc3,rad=0.08",
        )
    nx.draw_networkx_edges(graph, positions, **edge_drawing_options)
    axes[3].set_title(r"Coupling network (width $\propto\kappa$)")
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
) -> dict:
    """Run free-running validation cases and plot their mean and spread.

    ``detuning_distribution_ghz`` optionally fixes one detuning value per
    laser.  If omitted, the simulator samples the configured uniform
    distribution. Repeated cases share the same design and detunings but
    receive independent simulator noise.
    """
    if noise_iterations < 1:
        raise ValueError("noise_iterations must be at least 1")
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
    show_uncertainty = noise_iterations > 1

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 13),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for laser in range(run_config.n_lasers):
        color = colors[laser % len(colors)]
        axes[0].plot(
            time_ns,
            frequency_mean[laser],
            color=color,
            label=rf"$\dot{{\phi}}_{{{laser + 1}}}$",
        )
        if show_uncertainty:
            axes[0].fill_between(
                time_ns,
                frequency_mean[laser] - frequency_std[laser],
                frequency_mean[laser] + frequency_std[laser],
                color=color,
                alpha=0.20,
            )
        axes[1].plot(
            time_ns,
            phase_mean[laser],
            color=color,
            label=rf"$\phi_{{{laser + 1}}}-\phi_1$",
        )
        if show_uncertainty:
            axes[1].fill_between(
                time_ns,
                phase_mean[laser] - phase_std[laser],
                phase_mean[laser] + phase_std[laser],
                color=color,
                alpha=0.20,
            )
        axes[1].axhline(
            matched_target_rad[laser],
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        axes[2].plot(
            time_ns,
            laser_power_mean[laser],
            color=color,
            label=rf"$P_{{{laser + 1}}}$",
        )
        if show_uncertainty:
            axes[2].fill_between(
                time_ns,
                np.maximum(
                    laser_power_mean[laser] - laser_power_std[laser],
                    0.0,
                ),
                laser_power_mean[laser] + laser_power_std[laser],
                color=color,
                alpha=0.20,
            )

    axes[0].set_ylabel(r"$\dot{\phi}$ (GHz)")
    axes[0].set_title(
        "Standalone conditional design from a free-running initial state\n"
        f"phase reward={np.mean(phase_rewards):.3f} "
        f"$\\pm$ {np.std(phase_rewards):.3f} over "
        f"{noise_iterations} realization"
        f"{'s' if noise_iterations != 1 else ''}"
    )
    axes[1].set_ylabel(r"wrapped $\phi_i-\phi_1$")
    axes[1].set_ylim(-1.1*np.pi, 1.1*np.pi)
    axes[1].set_yticks(PHASE_TICKS)
    axes[1].set_yticklabels(PHASE_TICK_LABELS)
    axes[2].plot(
        time_ns,
        total_power_mean,
        color="green",
        linewidth=2.5,
        label=r"$P_{\rm total}$",
    )
    if show_uncertainty:
        axes[2].fill_between(
            time_ns,
            np.maximum(total_power_mean - total_power_std, 0.0),
            total_power_mean + total_power_std,
            color="green",
            alpha=0.25,
        )
    axes[2].set(
        xlabel="time (ns)",
        ylabel="output power (mW)",
    )

    for axis in axes:
        axis.axvspan(
            0.0,
            run_config.coupling_ramp_start_delays
            * run_config.delay_seconds
            * 1.0e9,
            color="gray",
            alpha=0.2,
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=legend_fontsize)
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
        "laser_power_mean_mw": laser_power_mean,
        "laser_power_std_mw": laser_power_std,
        "total_power_mean_mw": total_power_mean,
        "total_power_std_mw": total_power_std,
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
    policy = PolicyNetwork(active_config).to(
        torch.device(active_config.device)
    )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=active_config.learning_rate
    )

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
        n_jobs=active_config.n_jobs,
        jupyter_mode=active_config.jupyter_mode,
    )
    print(
        f"Loaded iteration {checkpoint_metadata['iteration']} with held-out "
        f"phase reward {checkpoint_metadata['held_out_reward']:.3f}"
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
        detuning_span_ghz=10.0,
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
        # "validation_results": validation_results,
        "best_design": best_design,
        "simulation_result": simulation_result,
    }


if __name__ == "__main__":
    results = main()
