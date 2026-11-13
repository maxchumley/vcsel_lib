#%%
"""Run the message-passing variable-M coupling policy.

This is a separate entry point so the existing pooled and BiGRU experiment
configuration, source files, and checkpoints remain untouched. Importing this
module only constructs the configuration; training starts solely from
``main()`` or by executing the guarded cell at the bottom.
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
from rl.paths import MODEL_DIR


TRAIN_NEW_MODEL = True
TRAINING_N_LASERS = tuple(range(3, 11))
FILE_SUFFIX = (
    "_gnn_M3-10_span5_edge_std_degree_scaled_stratified_span_"
    "no_M_features_lr1e-4"
)

# Retain the current experiment's optimizer, simulator, sampling, detuning,
# validation, and exploration schedules. Change only the representation used
# to construct each edge action, and remove explicit M-dependent adaptation.
training_size_label = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
config = replace(
    base_run.config,
    training_n_lasers=TRAINING_N_LASERS,
    training_n_laser_weights=None,
    encoder_architecture="gnn",
    gnn_message_passing_steps=3,
    gnn_message_hidden_size=64,
    condition_on_n_lasers=False,
    condition_log_std_on_n_lasers=False,
    modulate_edge_decoder_by_n_lasers=False,
    edge_conditioned_log_std=True,
    enable_sparse_gates=False,
    # Use the lower rate for the entire run; the scheduled switch therefore
    # leaves the optimizer at the same value.
    learning_rate=1.0e-4,
    learning_rate_after_switch=1.0e-4,
    detuning_curriculum_warmup_iterations=200,
    detuning_curriculum_iterations=1000,
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
    """Run the isolated GNN experiment without changing the BiGRU runner."""
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
    gnn_training_result = main()
