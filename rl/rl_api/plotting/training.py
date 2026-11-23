"""Training figure style shared by live progress and completed-run plots."""
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile


def save_training_progress(figure, directory, dpi, filename='current_training_progress.png'):
    """Save the run's PNG and a live-viewer copy one folder above it.

    Replace complete files atomically so a viewer never reads half a PNG.
    Render once, then copy those same image bytes to the parent directory.
    """
    directory = Path(directory)
    image = directory / filename
    temporary = image.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=dpi)
    temporary.replace(image)

    live_image = directory.parent / image.name
    live_temporary = directory.parent / f".{directory.name}_progress.png.tmp"
    copyfile(image, live_temporary)
    live_temporary.replace(live_image)


@contextmanager
def training_figure(history, config, settings):
    """Apply the run's resolution and simple_example.py font while rendering.

    Keep creation, saving and notebook display inside this context. Other
    figures keep their own Matplotlib settings when the context ends.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from .._compat import engine

    with plt.rc_context():
        rc("figure", dpi=settings.plot_dpi)
        rc("savefig", dpi=settings.plot_dpi)
        rc("text", usetex=settings.plot_use_tex)
        if settings.plot_use_tex:
            # Match the user's simple_example.py, whose final family is serif.
            rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
            rc("font", family="serif")
        figure = engine.training_figure(history, config)
        iteration = len(history["training_reward"])
        method = 'Actor–critic' if 'critic_loss' in history else 'REINFORCE'
        figure.axes[0].set_title(
            f"{method} training (iteration {iteration}/{config.training_iterations})")
        if 'seen_topology_held_out_reward' in history:
            import numpy as np
            grid = figure.add_gridspec(1, 3)
            figure.set_size_inches(16.5, 4)
            for index, axis in enumerate(figure.axes[:2]):
                axis.set_subplotspec(grid[index])
            reward_axis = figure.axes[1]
            x = np.arange(1, iteration+1)
            reward_axis.plot(x, history['seen_topology_held_out_reward'], '--',
                             color='tab:blue', label='trained-topology RMS')
            reward_axis.set_title('Held-out targets: topology comparison')
            for line in reward_axis.lines:
                if line.get_label() == 'overall RMS reward':
                    line.set_label('unseen-topology RMS')
            reward_axis.legend(loc='lower right', ncol=2)
            success_axis = figure.add_subplot(grid[2])
            success_axis.plot(x, history['seen_topology_held_out_phase_rms_success_fraction'],
                              label='trained topologies', color='tab:blue')
            success_axis.plot(x, history['held_out_phase_rms_success_fraction'],
                              label='unseen topologies', color='tab:orange')
            success_axis.set(xlabel='iteration', ylabel='fraction within target',
                             ylim=(-.05, 1.05), title=r'New targets: $3^\circ$ success')
            success_axis.grid(alpha=.25)
            success_axis.legend(loc='lower right')
        if 'held_out_tree_success_fraction' in history:
            figure.axes[1].set_title('Density-conditioned held-out targets')
            for line in figure.axes[0].lines:
                if line.get_label().startswith('mean supplied fraction'):
                    line.set_label(r'mean enabled fraction $\rho$')
            figure.axes[0].legend(loc='lower right')
        yield figure
