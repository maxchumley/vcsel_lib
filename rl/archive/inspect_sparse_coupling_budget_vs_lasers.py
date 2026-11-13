#%% Run the sparse coupling-budget sweep
"""Compare dense and sparse coupling budgets across trained array sizes.

For every requested number of lasers, the script draws independent detuning
distributions from the checkpoint's completed training curriculum.  All
detuning cases for one array size are evaluated in one batched policy call and
one grouped VCSEL simulation.  The reported budget is the directed sum
``sum(kappa_ij, i != j)`` in inverse nanoseconds; gated-off edges contribute
zero.
"""

from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (  # noqa: E402
    _initialize_simulation_worker,
    evaluate_policy,
    load_checkpoint,
    sample_detuning_distributions,
)
from rl.paths import MODEL_DIR, RESULTS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

DENSE_CHECKPOINT_KIND = "best"
SPARSE_CHECKPOINT_KIND = "current"
DENSE_MODEL_SUFFIX = "_bigru_M2-9_span1_edge_std"
SPARSE_MODEL_SUFFIX = f"{DENSE_MODEL_SUFFIX}_sparse"
TRAINING_SIZE_LABEL = "2to9"
CHECKPOINT_DIRECTORY = MODEL_DIR

# None evaluates every M stored in the checkpoint's training range. Supply a
# tuple such as (2, 3, 4) to inspect only selected trained sizes.
LASER_COUNTS: tuple[int, ...] | None = None
NUMBER_OF_DETUNING_DISTRIBUTIONS = 20
RANDOM_SEED = 1234

# Keep all 20 conditions together in one vectorized simulation on one process.
# With this set to 1, no multiprocessing pool is created.
NUMBER_OF_WORKERS = 1

# Deterministic evaluation isolates variation caused by detuning distributions.
NOISE_AMPLITUDE = 0.0

FIGURE_DIRECTORY = RESULTS_DIR / "analyses" / "coupling_and_sparsity"
FIGURE_FILENAME = "dense_vs_sparse_coupling_budget_vs_lasers.png"


def checkpoint_path(kind: str, suffix: str) -> Path:
    """Return one selected variable-M checkpoint path."""
    if kind not in {"best", "current", "final"}:
        raise ValueError(
            "checkpoint kind must be 'best', 'current', or 'final'"
        )
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return CHECKPOINT_DIRECTORY / (
        f"conditional_coupling_variable_m_{TRAINING_SIZE_LABEL}_lasers_"
        f"{kind}{suffix}.pt"
    )


def total_directed_coupling_per_case(kappa_per_ns: np.ndarray) -> np.ndarray:
    """Return ``sum(kappa_ij, i != j)`` for every coupling matrix."""
    kappa = np.asarray(kappa_per_ns, dtype=float)
    if kappa.ndim != 3 or kappa.shape[1] != kappa.shape[2]:
        raise ValueError("kappa_per_ns must have shape (cases, M, M)")
    diagonal = np.trace(kappa, axis1=1, axis2=2)
    return np.sum(kappa, axis=(1, 2)) - diagonal


def finite_mean_and_std(values: np.ndarray) -> tuple[float, float]:
    """Return the finite sample mean and standard deviation."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    spread = np.std(finite, ddof=1) if finite.size > 1 else 0.0
    return float(np.mean(finite)), float(spread)


def evaluate_coupling_budgets() -> dict[str, Any]:
    """Compare both models on the same full-range detuning distributions."""
    if NUMBER_OF_DETUNING_DISTRIBUTIONS < 2:
        raise ValueError(
            "NUMBER_OF_DETUNING_DISTRIBUTIONS must be at least 2 "
            "to calculate a sample standard deviation"
        )
    if NUMBER_OF_WORKERS < 1:
        raise ValueError("NUMBER_OF_WORKERS must be positive")

    checkpoint_specs = {
        "dense": (DENSE_CHECKPOINT_KIND, DENSE_MODEL_SUFFIX),
        "sparse": (SPARSE_CHECKPOINT_KIND, SPARSE_MODEL_SUFFIX),
    }
    policies = {}
    configs = {}
    metadata_by_model = {}
    checkpoints = {}
    for model_name, (kind, suffix) in checkpoint_specs.items():
        checkpoint = checkpoint_path(kind, suffix)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{model_name.capitalize()} checkpoint not found: "
                f"{checkpoint}"
            )
        policy, _, config, metadata = load_checkpoint(
            checkpoint,
            device="cpu",
            architecture="bigru",
        )
        policies[model_name] = policy
        configs[model_name] = config
        metadata_by_model[model_name] = metadata
        checkpoints[model_name] = checkpoint

    if getattr(policies["dense"], "enable_sparse_gates", False):
        raise ValueError("Dense checkpoint unexpectedly has sparse gates")
    if not getattr(policies["sparse"], "enable_sparse_gates", False):
        raise ValueError("Sparse checkpoint does not have sparse gates")

    trained_sizes = tuple(configs["sparse"].training_n_lasers)
    if tuple(configs["dense"].training_n_lasers) != trained_sizes:
        raise ValueError("Dense and sparse checkpoints have different M ranges")
    laser_counts = trained_sizes if LASER_COUNTS is None else LASER_COUNTS
    untrained_sizes = sorted(set(laser_counts) - set(trained_sizes))
    if untrained_sizes:
        raise ValueError(
            "LASER_COUNTS contains sizes outside the trained range: "
            f"{untrained_sizes}; trained sizes are {trained_sizes}"
        )

    active_workers = min(
        NUMBER_OF_WORKERS,
        NUMBER_OF_DETUNING_DISTRIBUTIONS,
    )
    for model_name in ("dense", "sparse"):
        print(
            f"Loaded {model_name} checkpoint "
            f"{checkpoints[model_name].name} "
            f"(iteration {metadata_by_model[model_name]['iteration']})."
        )
    print(
        f"Evaluating M={laser_counts}: "
        f"{NUMBER_OF_DETUNING_DISTRIBUTIONS} detuning distributions per M "
        f"with {active_workers} workers."
    )

    simulation_pool = None
    if active_workers > 1:
        simulation_pool = mp.get_context("spawn").Pool(
            processes=active_workers,
            initializer=_initialize_simulation_worker,
        )

    records: list[dict[str, Any]] = []
    try:
        for n_lasers in laser_counts:
            run_configs = {
                model_name: replace(
                    configs[model_name],
                    n_lasers=n_lasers,
                    n_jobs=active_workers,
                    jupyter_mode=False,
                    noise_amplitude=NOISE_AMPLITUDE,
                )
                for model_name in ("dense", "sparse")
            }
            rng = np.random.default_rng(RANDOM_SEED + n_lasers)
            detunings_ghz = sample_detuning_distributions(
                NUMBER_OF_DETUNING_DISTRIBUTIONS,
                rng,
                run_configs["sparse"],
                iteration=(
                    run_configs["sparse"].detuning_curriculum_iterations
                ),
            )
            targets_rad = np.zeros(
                (NUMBER_OF_DETUNING_DISTRIBUTIONS, n_lasers),
                dtype=float,
            )

            model_results = {}
            for model_name in ("dense", "sparse"):
                # Each model evaluates all 20 conditions in one vectorized
                # policy call and one grouped simulation.
                evaluation = evaluate_policy(
                    policies[model_name],
                    targets_rad,
                    run_configs[model_name],
                    detuning_distributions_ghz=detunings_ghz,
                    simulation_pool=simulation_pool,
                )
                coupling_samples = total_directed_coupling_per_case(
                    evaluation["kappa_per_ns"]
                )
                coupling_mean, coupling_std = finite_mean_and_std(
                    coupling_samples
                )
                phase_mean, phase_std = finite_mean_and_std(
                    evaluation["phase_reward"]
                )
                active_mean, active_std = finite_mean_and_std(
                    evaluation["active_connection_fraction"]
                )
                model_results[model_name] = {
                    "total_coupling_samples_per_ns": coupling_samples.copy(),
                    "total_coupling_mean_per_ns": coupling_mean,
                    "total_coupling_std_per_ns": coupling_std,
                    "phase_reward_samples": np.asarray(
                        evaluation["phase_reward"], dtype=float
                    ).copy(),
                    "phase_reward_mean": phase_mean,
                    "phase_reward_std": phase_std,
                    "active_fraction_samples": np.asarray(
                        evaluation["active_connection_fraction"], dtype=float
                    ).copy(),
                    "active_fraction_mean": active_mean,
                    "active_fraction_std": active_std,
                }
                print(
                    f"M={n_lasers:2d} | {model_name:6s} | "
                    f"total coupling={coupling_mean:.3f} +/- "
                    f"{coupling_std:.3f} ns^-1 | "
                    f"phase reward={phase_mean:.4f} +/- {phase_std:.4f} | "
                    f"active={active_mean:.3f} +/- {active_std:.3f}"
                )
            records.append(
                {
                    "n_lasers": int(n_lasers),
                    "detuning_distributions_ghz": detunings_ghz.copy(),
                    "dense": model_results["dense"],
                    "sparse": model_results["sparse"],
                }
            )
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
            simulation_pool.join()

    return {
        "policies": policies,
        "configs": configs,
        "checkpoints": checkpoints,
        "metadata": metadata_by_model,
        "results": records,
    }


# Only the original process may launch the persistent simulation pool. This
# also remains safe when the cell is run through VS Code/Jupyter on macOS.
if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    coupling_budget_sweep = evaluate_coupling_budgets()


#%% Plot existing sweep results without rerunning simulations
def plot_coupling_budgets(
    sweep: dict[str, Any] | list[dict[str, Any]],
) -> plt.Figure:
    """Plot mean total directed coupling with detuning-distribution spread."""
    records = sweep["results"] if isinstance(sweep, dict) else sweep
    if not isinstance(records, list) or not records:
        raise TypeError("sweep must contain a nonempty results list")

    counts = np.asarray(
        [record["n_lasers"] for record in records], dtype=int
    )
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model_name, label in (
        ("dense", "dense model"),
        ("sparse", "sparse model"),
    ):
        means = np.asarray(
            [
                record[model_name]["total_coupling_mean_per_ns"]
                for record in records
            ],
            dtype=float,
        )
        spreads = np.asarray(
            [
                record[model_name]["total_coupling_std_per_ns"]
                for record in records
            ],
            dtype=float,
        )
        finite = (
            np.isfinite(counts)
            & np.isfinite(means)
            & np.isfinite(spreads)
        )
        if not np.any(finite):
            raise ValueError(
                f"No finite {model_name} coupling-budget results are available"
            )
        axis.errorbar(
            counts[finite],
            means[finite],
            yerr=spreads[finite],
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=label,
        )
    axis.set(
        xlabel="number of lasers $M$",
        ylabel=r"total directed coupling $\sum_{i\ne j}\kappa_{ij}$ "
        r"(ns$^{-1}$)",
        title="In-phase design: dense and sparse coupling budgets",
        xticks=counts,
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left")

    output_path = FIGURE_DIRECTORY / FIGURE_FILENAME
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    return figure


if "coupling_budget_sweep" in globals():
    coupling_budget_figure = plot_coupling_budgets(
        coupling_budget_sweep
    )
    plt.show()
