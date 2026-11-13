"""Selected and ensemble figures, using already evaluated API data.

The original renderers also save/show through module globals. Bind their
unchanged code to a private copy of those globals so the API owns output and
display without changing the original module or simulating a second time.
"""
from types import FunctionType, SimpleNamespace
from pathlib import Path
import builtins

import numpy as np
import matplotlib.pyplot as plt


def renderer_without_output(function):
    """Reuse a renderer but leave saving/showing to this API call.

    FunctionType keeps the original code, arguments and defaults. Only its
    private plotting namespace differs; numerical helpers remain identical.
    """
    plotting = SimpleNamespace(**vars(plt))
    plotting.savefig = lambda *args, **kwargs: None
    plotting.show = lambda *args, **kwargs: None
    def renderer_import(name, globals=None, locals=None, fromlist=(), level=0):
        # The old dynamics renderer imports a standalone example that changes
        # fonts and imports VCSEL on import. Supply the same pure calculation
        # from our API instead, without importing that example or its globals.
        if name == "examples.visualization.far_field_intensity":
            from .far_field import far_field_intensity_map
            return SimpleNamespace(far_field_intensity_map=far_field_intensity_map)
        return builtins.__import__(name, globals, locals, fromlist, level)
    namespace = {**function.__globals__, "plt": plotting,
                 "__builtins__": {**vars(builtins), "__import__": renderer_import}}
    renderer = FunctionType(function.__code__, namespace, function.__name__,
                            function.__defaults__, function.__closure__)
    renderer.__kwdefaults__ = function.__kwdefaults__
    return renderer


def fix_automatic_legend_positions(figures, location):
    """Avoid Matplotlib scanning millions of vertices to place each legend.

    Output-power legends use center right. Other explicit placements (such as
    the phase-circle center) are kept; automatic legends use the requested location.
    This changes neither plotted data nor any simulation/summary calculation.
    """
    for figure in figures.values():
        for axis in figure.axes:
            legend = axis.get_legend()
            if legend is None:
                continue
            if "output power" in axis.get_ylabel().lower():
                legend.set_loc("center right")
            # Matplotlib's code 0 denotes automatic 'best' placement.
            elif legend._loc == 0:
                legend.set_loc(location)


def average_coupling_matrices(results, valid_topologies):
    """Average each topology once; absent strengths are zero, phases are circular.

    Disabled links have no phase and do not contribute to the phase mean.
    Opposing phases with a vanishing resultant also have no defined mean.
    """
    valid = np.asarray(valid_topologies, dtype=bool)
    if valid.shape != (len(results.designs),) or not valid.any():
        raise ValueError("Ensemble averages need at least one valid topology")
    kappa = results.designs.coupling_strengths_per_ns[valid]
    phi = results.designs.coupling_phases_rad[valid]
    active = kappa > 0
    counts = active.sum(axis=0)
    phasors = np.where(active, np.exp(1j*phi), 0.).sum(axis=0)
    undefined = (counts == 0) | (np.abs(phasors) <= 1e-12*counts)
    return kappa.mean(axis=0), np.ma.array(np.angle(phasors), mask=undefined)


def plot_ensemble_phase_distribution(axis, results, valid_topologies, phase_error_deg):
    """Single-case phase-circle convention with an ensemble-density ring.

    Use the same aligned settled phases as the selected-case renderer. Retain
    all noise realizations in the distribution, rather than collapsing a
    multimodal distribution to one circular mean before binning.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Wedge
    from .._compat import core

    size = results.designs.context.n_lasers
    samples = np.asarray(results.aligned_achieved_phases).reshape(
        len(results.designs), results.noise_realizations, size)[valid_topologies]
    wrapped = (samples + np.pi) % (2*np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, 181)  # Two-degree circular bins.
    counts, _ = np.histogram(wrapped, bins=edges)
    percentages = 100*counts/counts.sum()
    wedges = [Wedge((0., 0.), 1.25, np.rad2deg(a), np.rad2deg(b), width=.16)
              for a, b in zip(edges[:-1], edges[1:])]
    ring = PatchCollection(wedges, cmap="Oranges", edgecolor="none", zorder=1)
    ring.set_array(percentages)
    ring.set_clim(0., max(float(percentages.max()), 1e-12))
    axis.add_collection(ring)

    target = core.make_relative_to_laser_1(results.designs.context.target_phases_rad)
    circle = np.linspace(-np.pi, np.pi, 500)
    for radius in (1.09, 1.25):
        axis.plot(radius*np.cos(circle), radius*np.sin(circle), color=".85", linewidth=.5, zorder=1)
    axis.plot(np.cos(circle), np.sin(circle), color=".55", linewidth=1.2, zorder=1)
    tolerance = np.deg2rad(phase_error_deg)
    for i, angle in enumerate(np.unique(np.round(target, decimals=12))):
        arc = np.linspace(angle-tolerance, angle+tolerance, 101)
        axis.plot(np.cos(arc), np.sin(arc), color="tab:blue", alpha=.25,
                  linewidth=7., solid_capstyle="round", zorder=2,
                  label=rf"max $\pm {phase_error_deg:g}^\circ$" if i == 0 else None)
    axis.scatter(np.cos(target), np.sin(target), s=55, color="tab:blue", marker="o",
                 label="target", zorder=3)
    phasors = np.mean(np.exp(1j*samples), axis=(0, 1))
    defined = np.abs(phasors) > 1e-12
    achieved = np.angle(phasors[defined])
    axis.scatter(np.cos(achieved), np.sin(achieved), s=45, color="tab:orange", marker="s",
                 label="ensemble mean", zorder=4)
    axis.axhline(0., color=".8", linewidth=.7, zorder=0)
    axis.axvline(0., color=".8", linewidth=.7, zorder=0)
    axis.set(title="Settled phase comparison\nring = ensemble distribution",
             xlabel=r"$\cos(\Delta\phi)$", ylabel=r"$\sin(\Delta\phi)$",
             xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal",
             xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    axis.grid(alpha=.2)
    axis.legend(loc="center")
    return ring


def plot_ensemble_couplings(results, valid_topologies, phase_error_deg=3.):
    """Mean coupling matrices and settled phase distribution around a circle."""
    kappa, phi = average_coupling_matrices(results, valid_topologies)
    size = kappa.shape[0]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0., .07, 1., .93))
    kappa_cmap = plt.get_cmap("Blues").copy()
    phase_cmap = plt.get_cmap("twilight").copy()
    kappa_cmap.set_bad("black")
    phase_cmap.set_bad("black")
    magnitude = axes[0].imshow(np.ma.masked_where(kappa.T == 0, kappa.T),
        origin="upper", cmap=kappa_cmap, vmin=0., vmax=max(float(kappa.max()), 1e-12))
    phase = axes[1].imshow(phi.T, origin="upper", cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Mean coupling strength\n" +
        rf"$\langle K_{{\mathrm{{total}}}}\rangle = {kappa.sum():.3g}\ \mathrm{{ns}}^{{-1}}$")
    axes[1].set_title("Circular mean coupling phase\nenabled links only")
    figure.colorbar(magnitude, ax=axes[0], label=r"$\langle\kappa\rangle\ (\mathrm{ns}^{-1})$")
    colorbar = figure.colorbar(phase, ax=axes[1], label=r"$\langle\phi_p\rangle_{\mathrm{circ}}\ (\mathrm{rad})$")
    colorbar.set_ticks([-np.pi, -np.pi/2, 0., np.pi/2, np.pi],
                      labels=[r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    # Match the selected-design renderer: use six labels for larger arrays.
    ticks = np.arange(size) if size <= 10 else np.unique(np.rint(np.linspace(0, size-1, 6)).astype(int))
    for axis in axes[:2]:
        axis.set(xlabel="receiver", ylabel="source", xticks=ticks, yticks=ticks,
                 xticklabels=ticks+1, yticklabels=ticks+1)
    ring = plot_ensemble_phase_distribution(axes[2], results, valid_topologies, phase_error_deg)
    figure.colorbar(ring, ax=axes[2], label=r"Phase samples per $2^\circ$ bin ($\%$)")
    figure.suptitle(f"Ensemble coupling and settled phases ({np.count_nonzero(valid_topologies)}/{len(results.designs)} topologies)")
    figure.text(.5, .02, "Strengths include absent links as zero; phase means use enabled links. The ring includes valid topologies and noise realizations.",
                ha="center", fontsize="small")
    return figure


def selected_inspection_figures(results, chosen, options):
    """Reuse the original selected-design renderers without another integration."""
    from rl.conditional_coupling_run_variable_m import (
        plot_selected_design, simulate_and_plot_best_design)
    from . import selected_simulation

    index, designs = chosen.index, results.designs
    target = designs.context.target_phases_rad
    kappa, phi = chosen.coupling_strengths_per_ns, chosen.coupling_phases_rad
    receivers, sources = np.nonzero(kappa > 0)
    r = results.noise_realizations
    aligned = results.aligned_achieved_phases[index*r:(index+1)*r]
    achieved = results.raw_achieved_phases[index*r:(index+1)*r]
    design = dict(target_phases_rad=target,
        detuning_distribution_ghz=designs.context.detunings_ghz,
        kappa_per_ns=kappa, phi_p_rad=phi, receiver_indices=receivers, source_indices=sources,
        active_link_count=len(receivers),
        achieved_phases_rad=np.angle(np.mean(np.exp(1j*achieved), axis=0)),
        aligned_achieved_phases_rad=np.angle(np.mean(np.exp(1j*aligned), axis=0)),
        phase_reward=float(results.cosine_reward[index]))
    summary_plot = renderer_without_output(plot_selected_design)
    figures = {"design": summary_plot(design, results.config, vertical_layout=False,
            kappa_vmax_per_ns=max(float(kappa.max()), 1e-12),
            topology_backbone_network=True,
            topology_backbone_edge_limit_factor=max(len(receivers)/len(target),
                                                    (len(target)-1)/len(target)),
            topology_max_display_edges=options.max_display_edges,
            topology_edge_selection=options.edge_selection,
            topology_detuning_arc_layout=True, show_total_coupling_budget=True,
            disabled_links_black=True, allowable_rms_phase_error_deg=options.phase_error_deg,
            allowable_phase_error_metric="maximum")}
    dynamics_plot = renderer_without_output(simulate_and_plot_best_design)
    dynamics = dynamics_plot(design, results.config,
            validation_delay_count=results.config.simulation_time_seconds/results.config.delay_seconds,
            wavelength_m=options.wavelength_m,
            detuning_distribution_ghz=designs.context.detunings_ghz,
            legend_fontsize=plt.rcParams["legend.fontsize"], noise_iterations=r,
            dual_axis_power_panel=True, far_field_panel=options.show_far_field,
            emitter_spacing_m=options.emitter_spacing_um*1e-6,
            far_field_theta_range_deg=options.theta_range_deg,
            far_field_n_theta=options.n_theta,
            far_field_max_time_points=options.max_time_points,
            far_field_element_fwhm_deg=options.element_fwhm_deg,
            far_field_final_slice_panel=options.far_field_final_slice,
            far_field_slice_convolution_fwhm_deg=options.angular_blur_fwhm_deg,
            precomputed_simulation=selected_simulation(results, index),
            allowable_phase_error_deg=options.phase_error_deg)
    figures["dynamics"] = dynamics["figure"]
    return figures


def plot_inspection(results, *, select="best", output_directory=None, show=True, options):
    """Selected design/dynamics, mean coupling matrices, then ensemble dynamics.

    select=None skips selection and both selected-design figures entirely.
    """
    from rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse import (
        summarize_topology_trajectories, plot_topology_ensemble)
    from . import save_figures

    if select is None and not options.show_ensemble:
        raise ValueError("select=None requires show_ensemble=True")
    chosen = results.select(select) if select is not None else None
    designs = results.designs
    target = designs.context.target_phases_rad
    directory = Path(output_directory or designs.output_directory / "figures")
    with plt.rc_context({"text.usetex": options.use_tex, "figure.dpi": options.dpi}):
        figures = selected_inspection_figures(results, chosen, options) if chosen is not None else {}
        if options.show_ensemble:
            summary = summarize_topology_trajectories(results.simulation, target, results.config,
                n_topologies=len(designs), noise_iterations=results.noise_realizations, orientations=results.orientations,
                include_trajectories=options.trajectory_mode == "all")
            figures["ensemble_couplings"] = plot_ensemble_couplings(
                results, summary["valid_topologies"], phase_error_deg=options.phase_error_deg)
            figures["ensemble"] = plot_topology_ensemble(summary, target,
                trajectory_mode=options.trajectory_mode,
                profile_smoothing_bins=options.profile_smoothing_bins)
        fix_automatic_legend_positions(figures, options.legend_location)
        label = f"candidate_{chosen.index+1}_{select}" if chosen is not None else "ensemble"
        save_figures(figures, directory, label, options.dpi)
        if show:
            plt.show()
    return figures
