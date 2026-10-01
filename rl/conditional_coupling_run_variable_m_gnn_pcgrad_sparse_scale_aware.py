#%%
"""Train a scale-aware, tolerance-conditioned sparse variable-M GNN.

Detuning inputs are represented as a normalized per-array pattern plus the
physical peak-to-peak span.  This is a separate checkpoint family from the
fixed-scale sparse tolerance experiment.
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

from rl.conditional_coupling_run_variable_m_gnn_pcgrad import (  # noqa: E402
    config as previous_gnn_config,
)
from rl.conditional_coupling_sparse_tolerance import (  # noqa: E402
    SparseToleranceConfig,
    SparseToleranceGNNPolicy,
    train_sparse_tolerance_policy,
)
from rl.paths import MODEL_DIR  # noqa: E402


TRAINING_N_LASERS = tuple(range(3, 11))
SIZE_LABEL = f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
MODEL_SUFFIX = "_gnn_sparse_tol_scale_aware_span10ghz"
TRAINING_ITERATIONS = 3_000
LIVE_PLOT = True

_checkpoint_prefix = f"conditional_coupling_variable_m_{SIZE_LABEL}_lasers"
_settings = asdict(previous_gnn_config)
_settings.update(
    training_n_lasers=TRAINING_N_LASERS,
    n_lasers=TRAINING_N_LASERS[0],
    training_iterations=TRAINING_ITERATIONS,
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
    scale_aware_detuning_features=True,
    # 100 iterations at a 0.1-GHz peak-to-peak span, then a 500-iteration
    # expansion to 10 GHz.  Dense phase-control refinement continues at the
    # full range for iterations 600--1099; sparsity begins at 1100.
    detuning_span_ghz=10.0,
    detuning_curriculum_initial_half_span_ghz=0.05,
    detuning_curriculum_warmup_iterations=100,
    detuning_curriculum_iterations=600,
    detuning_span_sampling_mode="stratified_span",
    sparse_stage_start_iteration=1100,
    full_sparse_stage_start_iteration=1400,
    refinement_stage_start_iteration=2200,
    # Preserve the established tolerance-conditioned, constraint-first sparse
    # objective after the dense refinement interval.
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
    candidates_per_target=72,
    candidates_per_worker_task=72,
    dense_replay_candidates=8,
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
    """Train the isolated scale-aware sparse tolerance policy from scratch."""
    policy = SparseToleranceGNNPolicy(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    policy, optimizer, history = train_sparse_tolerance_policy(
        config, policy, optimizer, live_plot=LIVE_PLOT
    )
    return {
        "policy": policy,
        "optimizer": optimizer,
        "config": config,
        "history": history,
    }


if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    scale_aware_sparse_training_result = main()
