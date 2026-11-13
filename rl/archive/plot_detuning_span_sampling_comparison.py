#%%
"""Compare realized detuning spans from the old and new samplers."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow this file to run from either the repository root or the ``rl`` folder.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    DesignerConfig,
    sample_detuning_distributions,
)
from rl.paths import RESULTS_DIR


MAXIMUM_SPAN_GHZ = 5.0
N_SAMPLES = 100_000
N_LASERS_VALUES = (2, 3, 5, 9)
N_BINS = 50
RANDOM_SEED = 17


def sample_realized_spans(
    n_lasers: int,
    sampling_mode: str,
    seed: int,
) -> np.ndarray:
    """Return peak-to-peak detuning spans from one sampling method."""
    config = DesignerConfig(
        n_lasers=n_lasers,
        training_n_lasers=(n_lasers,),
        detuning_span_ghz=MAXIMUM_SPAN_GHZ,
        detuning_curriculum_initial_half_span_ghz=0.1,
        detuning_curriculum_warmup_iterations=200,
        detuning_curriculum_iterations=1000,
        detuning_span_sampling_mode=sampling_mode,
    )
    detunings_ghz = sample_detuning_distributions(
        N_SAMPLES,
        np.random.default_rng(seed),
        config,
        iteration=config.detuning_curriculum_iterations,
    )
    return np.ptp(detunings_ghz, axis=1)


def main() -> Path:
    """Create and save the sampler-comparison figure."""
    figure, axes = plt.subplots(
        2,
        len(N_LASERS_VALUES),
        figsize=(16, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    bins = np.linspace(0.0, MAXIMUM_SPAN_GHZ, N_BINS + 1)

    for panel_index, n_lasers in enumerate(N_LASERS_VALUES):
        original_spans = sample_realized_spans(
            n_lasers,
            "independent",
            RANDOM_SEED + panel_index,
        )
        stratified_spans = sample_realized_spans(
            n_lasers,
            "stratified_span",
            RANDOM_SEED + 100 + panel_index,
        )

        original_axis = axes[0, panel_index]
        new_axis = axes[1, panel_index]
        original_axis.hist(
            original_spans,
            bins=bins,
            density=True,
            alpha=0.75,
            color="tab:blue",
            label="sampled density",
        )
        new_axis.hist(
            stratified_spans,
            bins=bins,
            density=True,
            alpha=0.75,
            color="tab:orange",
            label="sampled density",
        )
        original_axis.axvline(
            np.mean(original_spans),
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label=rf"mean={np.mean(original_spans):.2f} GHz",
        )
        new_axis.axvline(
            np.mean(stratified_spans),
            color="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label=rf"mean={np.mean(stratified_spans):.2f} GHz",
        )
        original_axis.set_title(rf"$M={n_lasers}$")
        original_axis.grid(alpha=0.25)
        new_axis.grid(alpha=0.25)
        original_axis.legend(loc="upper left", fontsize=8)
        new_axis.legend(loc="upper left", fontsize=8)

        if n_lasers == 2:
            span_grid = np.linspace(0.0, MAXIMUM_SPAN_GHZ, 400)
            triangular_density = (
                2.0
                * (MAXIMUM_SPAN_GHZ - span_grid)
                / MAXIMUM_SPAN_GHZ**2
            )
            original_axis.plot(
                span_grid,
                triangular_density,
                color="navy",
                linewidth=2.0,
                label="theoretical triangular density",
            )
            original_axis.legend(loc="upper right", fontsize=8)

    for axis in axes[1]:
        axis.set_xlabel(r"realized peak-to-peak span $D$ (GHz)")
    axes[0, 0].set_ylabel("original sampler\nprobability density")
    axes[1, 0].set_ylabel("new sampler\nprobability density")
    figure.suptitle(
        "Realized detuning-span distributions: original versus new sampler"
    )

    output_directory = RESULTS_DIR / "analyses" / "detuning"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "detuning_span_sampling_comparison.png"
    figure.savefig(output_path, dpi=300)
    print(f"Saved comparison figure: {output_path}")
    plt.show()
    return output_path


#%%
if __name__ == "__main__":
    comparison_figure_path = main()
