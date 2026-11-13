"""Training figure style shared by live progress and completed-run plots."""
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile


def save_training_progress(figure, directory, dpi):
    """Save the run's PNG and a live-viewer copy one folder above it.

    Replace complete files atomically so a viewer never reads half a PNG.
    Render once, then copy those same image bytes to the parent directory.
    """
    directory = Path(directory)
    image = directory / "current_training_progress.png"
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
        figure.axes[0].set_title(
            f"REINFORCE training (iteration {iteration}/{config.training_iterations})")
        yield figure
