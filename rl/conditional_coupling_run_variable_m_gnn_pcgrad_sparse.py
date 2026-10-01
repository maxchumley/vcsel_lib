#%%
"""Train the selector-free tolerance-conditioned sparse GNN policy.

This is a new checkpoint and result family.  It does not modify the existing
fixed-budget/selector experiment in
``conditional_coupling_run_variable_m_gnn_pcgrad.py``.
"""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import sys

import torch

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_run_variable_m_gnn_pcgrad import (
    config as previous_gnn_config,
)
from rl.conditional_coupling_sparse_tolerance import (
    SparseToleranceConfig,
    SparseToleranceGNNPolicy,
    load_sparse_checkpoint,
    train_sparse_tolerance_policy,
)
from rl.paths import MODEL_DIR


TRAIN_NEW_MODEL = True
TRAINING_N_LASERS = tuple(range(3, 11))
MODEL_SUFFIX = "_gnn_sparse_tol_pcgrad"
SIZE_LABEL = f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"

_checkpoint_prefix = (
    f"conditional_coupling_variable_m_{SIZE_LABEL}_lasers"
)
_settings = asdict(previous_gnn_config)
_settings.update(
    training_n_lasers=TRAINING_N_LASERS,
    n_lasers=TRAINING_N_LASERS[0],
    training_iterations=2500,
    encoder_architecture="gnn",
    condition_on_edge_budget=False,
    enable_sparse_gates=False,
    enable_connected_edge_budgets=False,
    enable_learned_connected_sparsity=False,
    enable_learned_backbone_density=False,
    enable_conditional_backbone_budget=False,
    enable_budget_error_selector=False,
    monotonic_budget_selector=False,
    selector_use_deterministic_max_error=False,
    use_relative_phase_retention=False,
    use_lexicographic_rank_advantages=False,
    include_dense_reference_candidate=False,
    # 64 joint sparse-policy samples plus eight dense phase replay samples.
    candidates_per_target=72,
    candidates_per_worker_task=72,
    dense_replay_candidates=8,
    # Stage 1 ends exactly when the 4 GHz detuning curriculum completes.
    detuning_span_ghz=4.0,
    detuning_curriculum_warmup_iterations=200,
    detuning_curriculum_iterations=1000,
    sparse_stage_start_iteration=1000,
    full_sparse_stage_start_iteration=1300,
    refinement_stage_start_iteration=2200,
    # Stage 2 starts with strict tolerances and mild pruning.  Stage 3 widens
    # both ranges; strict examples remain in every all-M update.
    dense_stage_tolerance_deg=3.0,
    tolerance_minimum_deg=1.0,
    mild_tolerance_maximum_deg=3.0,
    strict_tolerance_maximum_deg=5.0,
    tolerance_maximum_deg=60.0,
    strict_tolerance_fraction=0.5,
    mild_minimum_q=0.8,
    initial_q_mean=0.95,
    initial_q_log_std=0.0,
    minimum_q_log_std=-2.0,
    maximum_q_log_std=0.75,
    final_maximum_q_log_std=-0.5,
    infeasible_reward_tau_deg=5.0,
    held_out_strict_tolerance_deg=3.0,
    plot_update_interval=1,
    train_all_sizes_each_iteration=True,
    targets_per_batch=14,
    gradient_combination_mode="pcgrad",
    per_size_gradient_clip=1.0,
    learning_rate=1.0e-4,
    learning_rate_after_switch=1.0e-4,
    best_checkpoint_file=str(
        MODEL_DIR / f"{_checkpoint_prefix}_best{MODEL_SUFFIX}.pt"
    ),
    current_checkpoint_file=str(
        MODEL_DIR / f"{_checkpoint_prefix}_current{MODEL_SUFFIX}.pt"
    ),
    final_checkpoint_file=str(
        MODEL_DIR / f"{_checkpoint_prefix}_final{MODEL_SUFFIX}.pt"
    ),
)
config = SparseToleranceConfig(**_settings)


def main() -> dict:
    """Train a fresh sparse policy or load the existing isolated checkpoint."""
    if TRAIN_NEW_MODEL:
        policy = SparseToleranceGNNPolicy(config).to(torch.device(config.device))
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        policy, optimizer, history = train_sparse_tolerance_policy(
            config,
            policy,
            optimizer,
            live_plot=True,
        )
        loaded_config = config
        metadata = {
            "iteration": config.training_iterations,
            "held_out_reward": history["held_out_dense_reward"][-1],
        }
    else:
        checkpoint = Path(config.current_checkpoint_file)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} does not exist. Set TRAIN_NEW_MODEL=True for "
                "the first run."
            )
        policy, optimizer, loaded_config, metadata = load_sparse_checkpoint(
            checkpoint, device=config.device
        )
        history = None
    return {
        "policy": policy,
        "optimizer": optimizer,
        "config": loaded_config,
        "history": history,
        "metadata": metadata,
    }


if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    sparse_training_result = main()

#%%
