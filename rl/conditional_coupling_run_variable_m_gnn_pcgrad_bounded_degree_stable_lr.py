#%%
"""Restart the bounded-degree GNN+PCGrad experiment at a stable LR.

Relative to ``conditional_coupling_run_variable_m_gnn_pcgrad_bounded_degree``,
this isolated run lowers the initial learning rate to 3e-4, extends the easy
detuning warmup to iteration 300, and reaches the full 5 GHz span at iteration
1200.  The later 1e-4 learning-rate switch and exploration schedule remain at
iteration 1500.  Existing checkpoints are never reused or overwritten.
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
from rl import (
    conditional_coupling_run_variable_m_gnn_pcgrad_bounded_degree
    as bounded_degree_run,
)
from rl.paths import MODEL_DIR


TRAIN_NEW_MODEL = True
FILE_SUFFIX = (
    "_gnn_M2-9_span5_edge_std_degree_scaled_stratified_span_"
    "bounded_degree_pcgrad_diverse_targets_lr3e4_slow_curriculum"
)

training_size_label = (
    f"{bounded_degree_run.config.training_n_lasers[0]}to"
    f"{bounded_degree_run.config.training_n_lasers[-1]}"
)
config = replace(
    bounded_degree_run.config,
    learning_rate=3.0e-4,
    learning_rate_switch_iteration=1500,
    learning_rate_after_switch=1.0e-4,
    detuning_curriculum_warmup_iterations=300,
    detuning_curriculum_iterations=1200,
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
    """Run the isolated lower-learning-rate restart from new weights."""
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
    stable_lr_training_result = main()
