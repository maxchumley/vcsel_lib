#%%
"""Train a variable-M GNN conditioned on a connected backbone budget.

This is a separate entry point and checkpoint family. It leaves the original
GNN, BiGRU, pooled-policy runners, and their checkpoints unchanged.
"""

from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import sys

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_run_variable_m as base_run
from rl.paths import MODEL_DIR, RESULTS_DIR


TRAIN_NEW_MODEL = True
TRAINING_N_LASERS = tuple(range(3, 11))
FILE_SUFFIX = (
    "_gnn_M3-10_span4_maxerr_selector_pcgrad"
)

training_size_label = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
config = replace(
    base_run.config,
    training_n_lasers=TRAINING_N_LASERS,
    training_n_laser_weights=None,
    # Architectural comparison controls: retain the exact GNN and decoder.
    encoder_architecture="gnn",
    gnn_message_passing_steps=3,
    gnn_message_hidden_size=64,
    condition_on_n_lasers=False,
    condition_log_std_on_n_lasers=False,
    modulate_edge_decoder_by_n_lasers=False,
    edge_conditioned_log_std=True,
    # Do not use independent edge gates. Sparsity is controlled by the exact
    # rho input and the deterministic connected-backbone selector.
    enable_sparse_gates=False,
    enable_connected_edge_budgets=False,
    condition_on_edge_budget=True,
    enable_learned_connected_sparsity=False,
    # Split 72 candidates across up to eight exact connected budgets. At each
    # budget, one deterministic policy-mean design supervises the selector
    # with maximum per-laser phase error. The other eight candidates retain
    # the unchanged phase-only REINFORCE update. Rho remains an input
    # condition rather than a policy action.
    enable_learned_backbone_density=False,
    enable_conditional_backbone_budget=True,
    enable_budget_error_selector=True,
    selector_budget_levels_per_target=8,
    selector_labels_per_target=8,
    monotonic_budget_selector=True,
    selector_use_deterministic_max_error=True,
    # This affects only the fixed progress diagnostic. Selector labels during
    # training are sampled from each measured error curve's feasible range.
    selector_validation_error_deg=5.0,
    use_relative_phase_retention=False,
    include_dense_reference_candidate=False,
    use_lexicographic_rank_advantages=False,
    successful_sparsity_reward_weight=0.0,
    coupling_cost_tiebreak_fraction=0.0,
    learning_rate=1.0e-4,
    learning_rate_after_switch=1.0e-4,
    # Use a 4 GHz total peak-to-peak detuning span for this new run.  The
    # existing 0.1 GHz warm-up and curriculum schedule are retained.
    detuning_span_ghz=4.0,
    detuning_curriculum_warmup_iterations=200,
    detuning_curriculum_iterations=1000,
    # Spread the fixed 14-target budget across M=3,...,10 each iteration.
    train_all_sizes_each_iteration=True,
    targets_per_batch=14,
    candidates_per_target=72,
    candidates_per_worker_task=72,
    # Clip every M gradient independently, project conflicting components,
    # average the eight projected gradients, then apply the existing global
    # gradient clip before the single optimizer step.
    gradient_combination_mode="pcgrad",
    per_size_gradient_clip=1.0,
    best_checkpoint_file=str(
        MODEL_DIR
        / (
            "conditional_coupling_variable_m_"
            f"{training_size_label}_lasers_best.pt"
        )
    ),
    current_checkpoint_file=str(
        MODEL_DIR
        / (
            "conditional_coupling_variable_m_"
            f"{training_size_label}_lasers_current.pt"
        )
    ),
    final_checkpoint_file=str(
        MODEL_DIR
        / (
            "conditional_coupling_variable_m_"
            f"{training_size_label}_lasers_final.pt"
        )
    ),
)
config = base_run._apply_checkpoint_suffix(config, FILE_SUFFIX)


def main() -> dict:
    """Run the isolated GNN+PCGrad experiment."""
    previous_suffix = base_run.FILE_SUFFIX
    previous_figure_directory = base_run.figure_directory
    try:
        base_run.FILE_SUFFIX = FILE_SUFFIX
        base_run.figure_directory = (
            RESULTS_DIR
            / "inference"
            / "deterministic_max_error_selector_current"
        )
        base_run.figure_directory.mkdir(parents=True, exist_ok=True)
        return base_run.main(config, train_new_model=TRAIN_NEW_MODEL)
    finally:
        base_run.FILE_SUFFIX = previous_suffix
        base_run.figure_directory = previous_figure_directory


if (
    __name__ == "__main__"
    and mp.current_process().name == "MainProcess"
):
    gnn_pcgrad_training_result = main()
