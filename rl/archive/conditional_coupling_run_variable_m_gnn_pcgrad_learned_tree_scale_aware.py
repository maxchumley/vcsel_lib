#%%
"""Train the learned-tree policy. CLI: --train; notebook: run this cell."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_scale_aware import config as source_config
from rl.archive.conditional_coupling_learned_tree_tolerance import (
    LearnedTreeConfig, LearnedTreePolicy, infer_learned_tree, topology_summary,
    train,
)
from rl.paths import MODEL_DIR

TRAINING_N_LASERS = tuple(range(3, 11))
TRAINING_ITERATIONS = 3000
WARMUP_ITERATIONS = 100
DETUNING_RAMP_ITERATIONS = 500
FULL_SPAN_PHASE_REFINEMENT_ITERATIONS = 500
DETUNING_SPAN_GHZ = 4.0
MILD_SPARSE_START = WARMUP_ITERATIONS + DETUNING_RAMP_ITERATIONS + FULL_SPAN_PHASE_REFINEMENT_ITERATIONS
FULL_SPARSE_START = 1400
REFINEMENT_START = 2200
LEARNING_RATE = 1.0e-4
TRAINING_DEVICE = "cpu"
SIMULATION_WORKERS = source_config.n_jobs
# Running this file as a Jupyter/VS Code cell uses this action. Change to
# "smoke-test" to check inference without starting the full training run.
NOTEBOOK_ACTION = "train"
SIZE_LABEL = f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
MODEL_SUFFIX = "_learned_tree_scale_aware_span4ghz"
PREFIX = f"conditional_coupling_variable_m_{SIZE_LABEL}_lasers"


def make_config():
    settings = asdict(source_config)
    settings.update(
        training_n_lasers=TRAINING_N_LASERS, n_lasers=TRAINING_N_LASERS[0],
        training_iterations=TRAINING_ITERATIONS,
        detuning_span_ghz=DETUNING_SPAN_GHZ,
        detuning_curriculum_warmup_iterations=WARMUP_ITERATIONS,
        detuning_curriculum_iterations=WARMUP_ITERATIONS + DETUNING_RAMP_ITERATIONS,
        sparse_stage_start_iteration=MILD_SPARSE_START,
        full_sparse_stage_start_iteration=FULL_SPARSE_START,
        refinement_stage_start_iteration=REFINEMENT_START,
        learning_rate=LEARNING_RATE, learning_rate_after_switch=LEARNING_RATE,
        device=TRAINING_DEVICE, n_jobs=SIMULATION_WORKERS,
        # Match the existing scale-aware runner: the VS Code Interactive
        # window displays the updated figure at every plot interval.
        jupyter_mode=source_config.jupyter_mode,
        edge_conditioned_log_std=False, condition_log_std_on_n_lasers=False,
    )
    for kind in ("best", "current", "final"):
        settings[f"{kind}_checkpoint_file"] = str(MODEL_DIR / f"{PREFIX}_{kind}{MODEL_SUFFIX}.pt")
    return LearnedTreeConfig(**settings)


def inference_smoke_test(policy):
    """Check untrained inference only; these are not simulated designs."""
    results = []
    for m in (100, 1000):
        target = 2 * np.pi * np.arange(m) / m
        detunings = np.linspace(-2.0, 2.0, m)
        infer_learned_tree(policy, target, detunings, 20.0, q_override=0.0)
        for q in (0.0, 0.5, 0.8):
            start = time.perf_counter()
            design = infer_learned_tree(policy, target, detunings, 20.0, q_override=q)
            elapsed = time.perf_counter() - start
            repeated = infer_learned_tree(policy, target, detunings, 20.0, q_override=q)
            for key in ("source_indices", "receiver_indices", "kappa_per_ns", "phi_p_rad"):
                np.testing.assert_array_equal(design[key], repeated[key])
            report = topology_summary(design)
            report.update(root_index=int(design["root_index"]), inference_seconds=elapsed,
                          device=str(next(policy.parameters()).device),
                          simulation="not run; M>100 dynamics unsupported")
            results.append(report)
    print(json.dumps(results, indent=2))
    return results


def _running_in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


def main(action=None):
    if action is None:
        if _running_in_notebook():
            action = NOTEBOOK_ACTION
        else:
            parser = argparse.ArgumentParser(description=__doc__)
            choices = parser.add_mutually_exclusive_group(required=True)
            choices.add_argument("--train", action="store_true")
            choices.add_argument("--smoke-test", action="store_true")
            args = parser.parse_args()
            action = "smoke-test" if args.smoke_test else "train"
    if action not in {"train", "smoke-test"}:
        raise ValueError("action must be 'train' or 'smoke-test'")
    config = make_config()
    torch.manual_seed(config.random_seed)
    policy = LearnedTreePolicy(config).to(config.device)
    if action == "smoke-test":
        return inference_smoke_test(policy)
    print("Learned-tree training: q=0 -> M-1 links; q=1 -> all directed links.")
    print(f"M={TRAINING_N_LASERS}; detuning curriculum 0.1 to {DETUNING_SPAN_GHZ} GHz")
    print(f"Current checkpoint: {config.current_checkpoint_file}")
    return train(policy)


if __name__ == "__main__":
    main()
