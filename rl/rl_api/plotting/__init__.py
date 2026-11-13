"""Plots consume designs/results; plotting never starts a new simulation.

Existing renderers retain topology arcs and circular ensemble statistics.
The inspection layout reuses the original renderer code in a private namespace
so the original modules' output paths and plotting globals remain untouched.
"""
from pathlib import Path
import numpy as np

from .options import PlotOptions


def design_dictionary(designs, index):
    """Build the existing renderer's edge-list representation from one matrix."""
    if not 0 <= index < len(designs):
        raise IndexError("Candidate index is outside this batch")
    kappa = designs.coupling_strengths_per_ns[index]
    phi = designs.coupling_phases_rad[index]
    receiver, source = np.nonzero(kappa > 0)
    return dict(target_phases_rad=designs.context.target_phases_rad,
        detuning_distribution_ghz=designs.context.detunings_ghz,
        receiver_indices=receiver, source_indices=source,
        kappa_per_ns=kappa[receiver, source], phi_p_rad=phi[receiver, source])


def save_figures(figures, directory, label, dpi):
    """One explicit export location; original RL output folders are not touched."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(directory / f"{label}_{name}.png", dpi=dpi)


def plot_proposal(designs, index, *, output_directory=None, show=True, options=None):
    """Plot coupling matrices and directed topology without simulating."""
    import matplotlib.pyplot as plt
    from rl.inference_plots import plot_inference_summary
    options = options or PlotOptions()
    directory = Path(output_directory or designs.output_directory / "figures")
    with plt.rc_context({"text.usetex": options.use_tex}):
        figure = plot_inference_summary(design_dictionary(designs, index),
            designs.coupling_strengths_per_ns[index], designs.coupling_phases_rad[index],
            bins=options.topology_bins or designs.context.n_lasers, max_display_edges=options.max_display_edges,
            edge_selection=options.edge_selection)
        figures = {"design": figure}
        save_figures(figures, directory, f"candidate_{index+1}", options.dpi)
        if show:
            plt.show()
    return figures


def selected_simulation(results, index):
    """Slice existing candidate/noise trajectories, retaining the shared time grid."""
    start = index*results.noise_realizations
    stop = start+results.noise_realizations
    return (results.simulation[0], *(part[start:stop] for part in results.simulation[1:]))


def phase_comparison(results, index):
    """Plot settled relative phases against targets on the unit circle."""
    import matplotlib.pyplot as plt
    r = results.noise_realizations
    phases = results.aligned_achieved_phases[index*r:(index+1)*r]
    mean = np.angle(np.mean(np.exp(1j*phases), axis=0))
    from .._compat import core
    target = core.make_relative_to_laser_1(results.designs.context.target_phases_rad)
    error = np.angle(np.exp(1j*(mean-target)))
    angle = np.linspace(0, 2*np.pi, 300)
    figure, axis = plt.subplots(figsize=(5,5), constrained_layout=True)
    axis.plot(np.cos(angle), np.sin(angle), color=".6")
    axis.scatter(np.ones_like(target), np.zeros_like(target), label="target", zorder=3)
    axis.scatter(np.cos(error), np.sin(error), marker="s", label="achieved", zorder=4)
    axis.set(xlabel="cos(phase error)", ylabel="sin(phase error)",
             title="Settled relative phase comparison", aspect="equal")
    axis.legend(); axis.grid(alpha=.2)
    return figure


def plot_evaluation(results, *, select="best", output_directory=None, show=True, options=None):
    """Plot selected matrices, trajectories, phase errors, ensemble and far field.

    A dictionary of figures is returned. Source data stay in EvaluationBatch;
    candidate selection has no hidden reward threshold or extra integration.
    """
    import matplotlib.pyplot as plt
    from rl.inference_plots import plot_inference_summary
    from rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse import (
        plot_topology_ensemble, summarize_topology_trajectories)
    from .far_field import calculate_far_field, plot_far_field
    options = options or PlotOptions()
    if options.layout == "inspection" or select is None:
        from .inspection import plot_inspection
        return plot_inspection(results, select=select, output_directory=output_directory,
                               show=show, options=options)
    chosen = results.select(select)
    index = chosen.index
    designs = results.designs
    directory = Path(output_directory or designs.output_directory / "figures")
    one_simulation = selected_simulation(results, index)
    r = results.noise_realizations
    orientation = results.orientations[index*r:(index+1)*r]
    with plt.rc_context({"text.usetex": options.use_tex}):
        figures = {"design": plot_inference_summary(design_dictionary(designs, index),
            chosen.coupling_strengths_per_ns, chosen.coupling_phases_rad,
            bins=designs.context.n_lasers, max_display_edges=options.max_display_edges,
            edge_selection=options.edge_selection), "phase_comparison": phase_comparison(results, index)}
        summary = summarize_topology_trajectories(one_simulation, designs.context.target_phases_rad,
            results.config, n_topologies=1, noise_iterations=r, orientations=orientation,
            include_trajectories=True)
        figures["dynamics"] = plot_topology_ensemble(summary, designs.context.target_phases_rad,
            trajectory_mode="all", profile_smoothing_bins=options.profile_smoothing_bins)
        if options.show_ensemble:
            summary = summarize_topology_trajectories(results.simulation, designs.context.target_phases_rad,
                results.config, n_topologies=len(designs), noise_iterations=r,
                orientations=results.orientations, include_trajectories=options.trajectory_mode == "all")
            figures["ensemble"] = plot_topology_ensemble(summary, designs.context.target_phases_rad,
                trajectory_mode=options.trajectory_mode, profile_smoothing_bins=options.profile_smoothing_bins)
        if options.show_far_field:
            figures["far_field"] = plot_far_field(calculate_far_field(one_simulation, options))
        label = f"candidate_{index+1}_{select}"
        save_figures(figures, directory, label, options.dpi)
        if show:
            plt.show()
    return figures


__all__ = ["PlotOptions", "plot_proposal", "plot_evaluation"]
