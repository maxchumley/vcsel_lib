#%%
"""Run GNN+PCGrad with bounded degree conditioning and diverse targets.

This is an isolated experiment with its own checkpoint family.  It retains
the shared GNN and edge decoder, adds [1/(M-1), 1/(M-1)^2] to their edge
features, and gives every trained array size three independent targets in
each optimizer update.
"""

from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import sys


for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl import conditional_coupling_run_variable_m as base_run
from rl import conditional_coupling_run_variable_m_gnn as gnn_run
from rl.paths import MODEL_DIR


TRAIN_NEW_MODEL = True
FILE_SUFFIX = (
    "_gnn_M2-9_span5_edge_std_degree_scaled_stratified_span_"
    "bounded_degree_pcgrad_diverse_targets"
)

training_size_label = (
    f"{gnn_run.config.training_n_lasers[0]}to"
    f"{gnn_run.config.training_n_lasers[-1]}"
)
config = replace(
    gnn_run.config,
    encoder_architecture="gnn",
    gnn_message_passing_steps=3,
    gnn_message_hidden_size=64,
    # Use only smooth, bounded degree information; retain one shared decoder
    # and no M-conditioned modulation head.
    condition_on_n_lasers=False,
    condition_log_std_on_n_lasers=False,
    condition_on_bounded_degree=True,
    modulate_edge_decoder_by_n_lasers=False,
    edge_conditioned_log_std=True,
    # Three independent targets for every M per update.  This is 1,152
    # simulated target-candidate cases versus 896 in the first PCGrad run.
    train_all_sizes_each_iteration=True,
    targets_per_batch=24,
    candidates_per_target=48,
    candidates_per_worker_task=48,
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
    """Run the isolated bounded-degree GNN+PCGrad experiment."""
    previous_suffix = base_run.FILE_SUFFIX
    try:
        base_run.FILE_SUFFIX = FILE_SUFFIX
        return base_run.main(config, train_new_model=TRAIN_NEW_MODEL)
    finally:
        base_run.FILE_SUFFIX = previous_suffix


if (
    __name__ == "__main__"
    and mp.current_process().name == "MainProcess"
):
    bounded_degree_training_result = main()
