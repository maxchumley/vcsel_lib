#%%
"""Continue the scale-aware sparse GNN with exponential LR decay.

This keeps the original scale-aware run untouched.  It resumes its current
checkpoint at the saved global iteration and performs 5,000 more refinement
updates, decaying Adam's learning rate exponentially from 1e-4 to 1e-6.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "rl").is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_sparse_tolerance import (  # noqa: E402
    load_sparse_checkpoint,
    train_sparse_tolerance_policy,
)
from rl.paths import MODEL_DIR  # noqa: E402


TRAINING_N_LASERS = tuple(range(3, 11))
SIZE_LABEL = f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
SOURCE_SUFFIX = "_gnn_sparse_tol_scale_aware_span10ghz"
REFINEMENT_SUFFIX = (
    "_gnn_sparse_tol_scale_aware_span10ghz_refine5000_exp_lr1e-4to1e-6"
)

ADDITIONAL_ITERATIONS = 5_000
INITIAL_LEARNING_RATE = 1.0e-4
FINAL_LEARNING_RATE = 1.0e-6
LIVE_PLOT = True


def checkpoint_path(kind: str, suffix: str) -> Path:
    """Return a checkpoint path for the configured variable-M family."""
    if kind not in {"best", "current", "final"}:
        raise ValueError("checkpoint kind must be 'best', 'current', or 'final'")
    return MODEL_DIR / (
        f"conditional_coupling_variable_m_{SIZE_LABEL}_lasers_{kind}{suffix}.pt"
    )


SOURCE_CHECKPOINT = checkpoint_path("current", SOURCE_SUFFIX)
BEST_CHECKPOINT = checkpoint_path("best", REFINEMENT_SUFFIX)
CURRENT_CHECKPOINT = checkpoint_path("current", REFINEMENT_SUFFIX)
FINAL_CHECKPOINT = checkpoint_path("final", REFINEMENT_SUFFIX)


def main() -> dict:
    """Resume the completed scale-aware run for the refinement horizon."""
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {SOURCE_CHECKPOINT}")

    policy, optimizer, source_config, metadata = load_sparse_checkpoint(
        SOURCE_CHECKPOINT, device="cpu"
    )
    start_iteration = int(metadata["iteration"])
    if tuple(source_config.training_n_lasers) != TRAINING_N_LASERS:
        raise ValueError("Source checkpoint has unexpected training array sizes")
    if not source_config.scale_aware_detuning_features:
        raise ValueError("Source checkpoint is not scale-aware")

    total_iterations = start_iteration + ADDITIONAL_ITERATIONS
    config = replace(
        source_config,
        training_iterations=total_iterations,
        learning_rate=INITIAL_LEARNING_RATE,
        learning_rate_switch_iteration=total_iterations + 1,
        learning_rate_after_switch=INITIAL_LEARNING_RATE,
        learning_rate_final=FINAL_LEARNING_RATE,
        learning_rate_decay_start_iteration=start_iteration,
        learning_rate_decay_iterations=ADDITIONAL_ITERATIONS,
        # Do not reopen q exploration merely because this extension has a
        # later final iteration.  Continue from the source run's settled
        # refinement ceiling throughout the LR-decay pass.
        initial_q_log_std=source_config.final_maximum_q_log_std,
        maximum_q_log_std=source_config.final_maximum_q_log_std,
        final_maximum_q_log_std=source_config.final_maximum_q_log_std,
        best_checkpoint_file=str(BEST_CHECKPOINT),
        current_checkpoint_file=str(CURRENT_CHECKPOINT),
        final_checkpoint_file=str(FINAL_CHECKPOINT),
    )
    for group in optimizer.param_groups:
        group["lr"] = INITIAL_LEARNING_RATE

    print(f"Resuming: {SOURCE_CHECKPOINT}")
    print(
        f"Source iteration={start_iteration}; continuing through "
        f"{total_iterations} ({ADDITIONAL_ITERATIONS} additional updates)."
    )
    print(
        "Exponential LR decay: "
        f"{INITIAL_LEARNING_RATE:.1e} -> {FINAL_LEARNING_RATE:.1e}."
    )
    policy, optimizer, history = train_sparse_tolerance_policy(
        config,
        policy,
        optimizer,
        live_plot=LIVE_PLOT,
        start_iteration=start_iteration,
    )
    return {
        "policy": policy,
        "optimizer": optimizer,
        "config": config,
        "history": history,
        "source_metadata": metadata,
    }


if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    scale_aware_sparse_tolerance_refinement_result = main()
