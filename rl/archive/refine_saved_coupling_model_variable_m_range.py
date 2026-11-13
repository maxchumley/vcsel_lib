#%%
"""Diagnose a saved variable-M policy with an M=2-only continuation.

This refinement optimizes only the original phase-locking objective. It does
not add sparse gates or coupling-budget penalties. The source checkpoint is
loaded read-only, a fresh optimizer is used, and diagnostic checkpoints are
saved under a separate size label and suffix. The source model's original
M=2..9 normalization remains unchanged while only M=2 supplies updates.

All executable work lives in :func:`main` so multiprocessing remains safe on
macOS and Windows.
"""

from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

import torch

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (  # noqa: E402
    BiGRUPolicyNetwork,
    DesignerConfig,
    load_checkpoint,
    train_reinforce,
)
from rl.paths import MODEL_DIR, RESULTS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------

SOURCE_CHECKPOINT_KIND = "current"  # "best", "current", or "final"
SOURCE_TRAINING_N_LASERS = tuple(range(2, 10))
SOURCE_MODEL_SUFFIX = (
    "_bigru_M2-9_span5_edge_std_degree_scaled_"
    "stratified_span_std_floor"
)


# ---------------------------------------------------------------------------
# Range-refinement controls
# ---------------------------------------------------------------------------

# Use only M=2 to test whether removing mixed-size gradient interference lets
# the same shared policy rapidly repair its two-laser mapping.
REFINEMENT_N_LASERS = (2,)

REFINEMENT_ITERATIONS = 300
REFINEMENT_LEARNING_RATE = 3.0e-4

# detuning_span_ghz is the total interval before each sampled row is centered:
# 5 GHz corresponds nominally to samples in -2.5..+2.5 GHz.
REFINEMENT_DETUNING_SPAN_GHZ = 5.0

# Keep the diagnostic at the complete 5 GHz span from its first update.
DETUNING_INITIAL_HALF_SPAN_GHZ = 2.5
DETUNING_WARMUP_ITERATIONS = 0
DETUNING_CURRICULUM_ITERATIONS = 0

# Retain useful exploration throughout this short diagnostic. The scheduled
# lower bound below guarantees sigma >= exp(-1.5) for all 300 updates.
REFINEMENT_MAXIMUM_LOG_STD = -0.5
EXPLORATION_ANNEAL_START_ITERATION = REFINEMENT_ITERATIONS + 1
EXPLORATION_ANNEAL_ITERATIONS = 0
FINAL_MAXIMUM_LOG_STD = -1.5
REFINEMENT_MINIMUM_LOG_STD = -1.5

# Preserve the existing target-level parallel/vectorized simulation layout.
TARGETS_PER_BATCH = 14
CANDIDATES_PER_TARGET = 64
CANDIDATES_PER_WORKER_TASK = 64
N_JOBS = os.cpu_count() or 1

# Frequent validation makes the M=2 recovery rate directly visible.
HELD_OUT_TARGET_COUNT = 64
VALIDATION_INTERVAL = 10

OUTPUT_MODEL_SUFFIX = (
    f"{SOURCE_MODEL_SUFFIX}_M2_only_diagnostic"
)

RUN_REFINEMENT = True
LIVE_PLOT = True


def checkpoint_path(
    kind: str,
    training_sizes: tuple[int, ...],
    suffix: str,
) -> Path:
    """Return a compact variable-M checkpoint path."""
    if kind not in {"best", "current", "final"}:
        raise ValueError("checkpoint kind must be 'best', 'current', or 'final'")
    size_label = f"{training_sizes[0]}to{training_sizes[-1]}"
    normalized_suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return MODEL_DIR / (
        f"conditional_coupling_variable_m_{size_label}_lasers_"
        f"{kind}{normalized_suffix}.pt"
    )


SOURCE_CHECKPOINT = checkpoint_path(
    SOURCE_CHECKPOINT_KIND,
    SOURCE_TRAINING_N_LASERS,
    SOURCE_MODEL_SUFFIX,
)
REFINED_BEST_CHECKPOINT = checkpoint_path(
    "best", REFINEMENT_N_LASERS, OUTPUT_MODEL_SUFFIX
)
REFINED_CURRENT_CHECKPOINT = checkpoint_path(
    "current", REFINEMENT_N_LASERS, OUTPUT_MODEL_SUFFIX
)
REFINED_FINAL_CHECKPOINT = checkpoint_path(
    "final", REFINEMENT_N_LASERS, OUTPUT_MODEL_SUFFIX
)


def make_refinement_config(
    source_config: DesignerConfig,
) -> DesignerConfig:
    """Build the full-span M=2-only diagnostic configuration."""
    return replace(
        source_config,
        n_lasers=REFINEMENT_N_LASERS[-1],
        training_n_lasers=REFINEMENT_N_LASERS,
        training_n_laser_weights=None,
        training_iterations=REFINEMENT_ITERATIONS,
        learning_rate=REFINEMENT_LEARNING_RATE,
        learning_rate_switch_iteration=REFINEMENT_ITERATIONS + 1,
        learning_rate_after_switch=REFINEMENT_LEARNING_RATE,
        targets_per_batch=TARGETS_PER_BATCH,
        candidates_per_target=CANDIDATES_PER_TARGET,
        candidates_per_worker_task=CANDIDATES_PER_WORKER_TASK,
        n_jobs=N_JOBS,
        detuning_span_ghz=REFINEMENT_DETUNING_SPAN_GHZ,
        detuning_curriculum_initial_half_span_ghz=(
            DETUNING_INITIAL_HALF_SPAN_GHZ
        ),
        detuning_curriculum_warmup_iterations=DETUNING_WARMUP_ITERATIONS,
        detuning_curriculum_iterations=DETUNING_CURRICULUM_ITERATIONS,
        # This field is validated when the copied dataclass is constructed;
        # the loaded policy parameters themselves are not reinitialized.
        initial_log_std=REFINEMENT_MAXIMUM_LOG_STD,
        maximum_log_std=REFINEMENT_MAXIMUM_LOG_STD,
        enable_minimum_log_std_annealing=True,
        initial_minimum_log_std=REFINEMENT_MINIMUM_LOG_STD,
        minimum_log_std_anneal_start_iteration=REFINEMENT_ITERATIONS + 1,
        minimum_log_std_anneal_iterations=0,
        enable_exploration_annealing=False,
        exploration_anneal_start_iteration=(
            EXPLORATION_ANNEAL_START_ITERATION
        ),
        exploration_anneal_iterations=EXPLORATION_ANNEAL_ITERATIONS,
        final_maximum_log_std=FINAL_MAXIMUM_LOG_STD,
        held_out_target_count=HELD_OUT_TARGET_COUNT,
        validation_interval=VALIDATION_INTERVAL,
        random_seed=source_config.random_seed + 50_000,
        best_checkpoint_file=str(REFINED_BEST_CHECKPOINT),
        current_checkpoint_file=str(REFINED_CURRENT_CHECKPOINT),
        final_checkpoint_file=str(REFINED_FINAL_CHECKPOINT),
    )


def main() -> dict[str, Any]:
    """Load the source checkpoint and optionally run phase-only refinement."""
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {SOURCE_CHECKPOINT}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    policy, _, source_config, metadata = load_checkpoint(
        SOURCE_CHECKPOINT,
        architecture="bigru",
    )
    if not isinstance(policy, BiGRUPolicyNetwork):
        raise TypeError("range refinement requires a BiGRU policy")
    if policy.enable_sparse_gates:
        raise ValueError(
            "range refinement expects the ungated source model; select the "
            "dense checkpoint suffix"
        )
    if tuple(source_config.training_n_lasers) != SOURCE_TRAINING_N_LASERS:
        raise ValueError(
            "Checkpoint training sizes do not match "
            "SOURCE_TRAINING_N_LASERS"
        )

    refinement_config = make_refinement_config(source_config)

    # Preserve the exact M features seen by the source model. Setting both
    # limits to 2 would move normalized M from -1 to 0 and confound the test.
    policy.minimum_training_n_lasers = min(SOURCE_TRAINING_N_LASERS)
    policy.maximum_training_n_lasers = max(SOURCE_TRAINING_N_LASERS)
    policy.minimum_log_std = REFINEMENT_MINIMUM_LOG_STD
    policy.maximum_log_std = REFINEMENT_MAXIMUM_LOG_STD

    print(f"Loaded source model: {SOURCE_CHECKPOINT}")
    print(
        f"Source checkpoint iteration={metadata['iteration']}, "
        f"held-out reward={metadata['held_out_reward']:.4f}"
    )
    print(
        "Phase-only diagnostic refinement: "
        f"M={REFINEMENT_N_LASERS[0]} only, "
        f"detuning span={REFINEMENT_DETUNING_SPAN_GHZ:g} GHz, "
        f"lr={REFINEMENT_LEARNING_RATE:.1e}"
    )
    print("No sparsity or coupling-budget penalty is active.")
    print(f"Refined outputs will use suffix: {OUTPUT_MODEL_SUFFIX}")

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=REFINEMENT_LEARNING_RATE,
    )
    history = None
    if RUN_REFINEMENT:
        policy, optimizer, history = train_reinforce(
            refinement_config,
            policy=policy,
            optimizer=optimizer,
            live_plot=LIVE_PLOT,
        )

    return {
        "policy": policy,
        "optimizer": optimizer,
        "config": refinement_config,
        "source_metadata": metadata,
        "history": history,
    }


#%%
# The process-name check prevents spawned workers from recursively training.
if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    range_refinement_result = main()
