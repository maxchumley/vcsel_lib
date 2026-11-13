#%%
"""Separate sparse-graph prototype. Pass --train to start the configured run."""
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
from rl.archive.conditional_coupling_sparse_graph_tolerance import (
    SparseGraphConfig, SparseGraphPolicy, infer_sparse_graph, topology_summary, train,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


TRAINING_N_LASERS = tuple(range(3, 11))
MAX_EXTRA_EDGES_PER_NODE = 4
OPTIONAL_CANDIDATES_PER_NODE = 8
# The backbone is fixed; it has no candidate count or trainable topology head.
TRAINING_ITERATIONS = 3000
WARMUP_ITERATIONS = 100
DETUNING_RAMP_ITERATIONS = 500
FULL_SPAN_PHASE_REFINEMENT_ITERATIONS = 500
DETUNING_SPAN_GHZ = 10.0
MILD_SPARSE_START = WARMUP_ITERATIONS + DETUNING_RAMP_ITERATIONS + FULL_SPAN_PHASE_REFINEMENT_ITERATIONS
FULL_SPARSE_START = 1400
REFINEMENT_START = 2200
LEARNING_RATE = 1.0e-4
TRAINING_DEVICE = "cpu"
# Physical simulation is unchanged and remains on CPU, restricted to M <= 100.
SIMULATION_WORKERS = source_config.n_jobs
SIZE_LABEL = f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
MODEL_SUFFIX = f"_sparse_graph_prototype_scale_aware_span10ghz_k{MAX_EXTRA_EDGES_PER_NODE}_c{OPTIONAL_CANDIDATES_PER_NODE}"
PREFIX = f"conditional_coupling_variable_m_{SIZE_LABEL}_lasers"
PROGRESS_FILE = RESULTS_DIR / "sparse_graph_prototype" / f"{PREFIX}{MODEL_SUFFIX}_progress.json"


def make_config():
    settings = asdict(source_config)
    settings.update(
        training_n_lasers=TRAINING_N_LASERS, n_lasers=TRAINING_N_LASERS[0],
        max_extra_edges_per_node=MAX_EXTRA_EDGES_PER_NODE,
        optional_candidates_per_node=OPTIONAL_CANDIDATES_PER_NODE,
        training_iterations=TRAINING_ITERATIONS,
        detuning_span_ghz=DETUNING_SPAN_GHZ,
        detuning_curriculum_warmup_iterations=WARMUP_ITERATIONS,
        detuning_curriculum_iterations=WARMUP_ITERATIONS+DETUNING_RAMP_ITERATIONS,
        sparse_stage_start_iteration=MILD_SPARSE_START,
        full_sparse_stage_start_iteration=FULL_SPARSE_START,
        refinement_stage_start_iteration=REFINEMENT_START,
        learning_rate=LEARNING_RATE, learning_rate_after_switch=LEARNING_RATE,
        device=TRAINING_DEVICE, n_jobs=SIMULATION_WORKERS, jupyter_mode=False,
        edge_conditioned_log_std=False,
        # The prototype learns one shared kappa std and one shared phase std.
        condition_log_std_on_n_lasers=False,
    )
    for kind in ("best", "current", "final"):
        settings[f"{kind}_checkpoint_file"] = str(MODEL_DIR / f"{PREFIX}_{kind}{MODEL_SUFFIX}.pt")
    return SparseGraphConfig(**settings)


def inference_smoke_test(policy):
    """Benchmark untrained inference only; no claim of physical solution quality."""
    device = next(policy.parameters()).device
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    results = []
    for m in (100, 1000):
        target = 2*np.pi*np.arange(m)/m
        detuning = np.linspace(-5, 5, m)
        infer_sparse_graph(policy, target, detuning, 20.0)  # warm up backend
        for q in (0.0, 1.0):
            synchronize()
            start = time.perf_counter()
            design = infer_sparse_graph(policy, target, detuning, 20.0, q_override=q)
            synchronize()
            elapsed = time.perf_counter()-start
            repeated = infer_sparse_graph(policy, target, detuning, 20.0, q_override=q)
            for key in ("source_indices", "receiver_indices", "kappa_per_ns", "phi_p_rad"):
                np.testing.assert_array_equal(design[key], repeated[key])
            report = topology_summary(design)
            assert report["active_edges"] <= m-1+m*policy.config.max_extra_edges_per_node
            assert q != 0.0 or report["active_edges"] == m-1
            report.update(inference_seconds=elapsed, device=str(device),
                          simulation="not run; M>100 dynamics unsupported",
                          memory_scope="returned edge arrays only; excludes model and temporary tensors")
            results.append(report)
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--smoke-test", action="store_true", help="untrained M=100/1000 inference only")
    args = parser.parse_args()
    config = make_config()
    torch.manual_seed(config.random_seed)
    policy = SparseGraphPolicy(config).to(config.device)
    if args.smoke_test:
        return inference_smoke_test(policy)
    print("New sparse-graph run: q=1 is maximum sparse budget, q=0 is the fixed chain.")
    print(f"K={MAX_EXTRA_EDGES_PER_NODE}, C={OPTIONAL_CANDIDATES_PER_NODE}; M={TRAINING_N_LASERS}")
    print(f"Current checkpoint: {config.current_checkpoint_file}")
    return train(policy, progress_file=PROGRESS_FILE)


if __name__ == "__main__":
    main()
