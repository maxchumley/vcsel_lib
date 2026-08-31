#%%
"""Inspect a variable-size pooled or legacy BiGRU policy."""

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
)
from rl.conditional_coupling_run_variable_m import (
    plot_selected_design,
    simulate_and_plot_best_design,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "current"  # "best" or "current"
# These identify which multi-size training run produced the checkpoint.
TRAINING_N_LASERS = tuple(range(2, 11))
# Match the compact checkpoint label used by the selected training sizes.
# It updates automatically when TRAINING_N_LASERS is changed.
CHECKPOINT_SIZE_LABEL = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
# This is the size to inspect. It may be a trained or unseen array size.
N_LASERS = 5
MODEL_SUFFIX = "_bigru_balanced"  # Optional checkpoint suffix.
# "auto" detects the architecture stored in the checkpoint. Use "bigru" or
# "pooled" to require a particular architecture and catch filename mistakes.
MODEL_ARCHITECTURE = "auto"

# Output location and optional suffix for both saved figures.
FIGURE_DIRECTORY = RESULTS_DIR / "gru_results"
FIGURE_SUFFIX = f"{N_LASERS}_in_phase"  # For example: "splay_noise"

# Enter target phases as multiples of pi. The first entry is the reference.
MANUAL_TARGET_PHASES_PI = 1.0*(
    2.0 * np.arange(N_LASERS) / N_LASERS
)





N_CLUSTERS = 1
N_CLUSTERS = np.copy(N_LASERS)

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
MANUAL_DETUNINGS_GHZ = 2.0*0.2*np.linspace(-5.0, 5.0, N_LASERS)

# 5.0*np.random.uniform(-1,1, N_LASERS)

# np.array(
#     [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0]
# )

NUMBER_OF_CANDIDATES = 1
VALIDATION_TIME_NS = 500.0

# Noise is applied only during the final validation, not candidate selection.
# Set NOISE_AMPLITUDE=0.0 and NOISE_ITERATIONS=1 for a deterministic run.
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1

NOISE_RANDOM_SEED = None

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
    training_sizes = CHECKPOINT_SIZE_LABEL
    checkpoint = MODEL_DIR / (
        f"conditional_coupling_variable_m_{training_sizes}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

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

    design = design_for_target(
        policy,
        target_phases_rad,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        top_k=1,
        detuning_distribution_ghz=detunings_ghz,
    )[0]

    print(f"Phase reward: {design['phase_reward']:.4f}")
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
    figure_label = f"manual_target{figure_suffix}"

    # Scale this script's kappa heatmap to the selected matrix instead of the
    # policy's full allowed 0..maximum_kappa_per_ns range.
    plotted_kappa_max = max(
        float(np.max(design["kappa_per_ns"])),
        1.0e-12,
    )
    plotting_config = replace(
        config,
        maximum_kappa_per_ns=plotted_kappa_max,
    )
    with plt.rc_context(PLOT_STYLE):
        plot_selected_design(
            design,
            plotting_config,
            desired_state=figure_label,
            vertical_layout=False,
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
        )
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


if __name__ == "__main__":
    results = main()

# %%
