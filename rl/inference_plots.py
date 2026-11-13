"""Plots shared by the current Bernoulli checkpoint inspector."""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np

from rl import conditional_coupling_run_variable_m as coupling_plots

NETWORK_MAX_DISPLAY_EDGES = 6000


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
