#%%
"""Train the 4-GHz scale-aware sparse policy for high-accuracy sparse designs.

The GNN, q head, edge decoder, staged curriculum, simulator, and PCGrad
training loop are the same as the current sparse scale-aware run. Only the
opt-in post-pruning allocation, fixed tolerance, and accuracy-first reward differ.
"""

import multiprocessing as mp
from pathlib import Path
import sys

import torch

for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_scale_aware import (
    _settings as _scale_aware_settings,
    LIVE_PLOT,
)
from rl.archive.conditional_coupling_sparse_tolerance import (
    SparseToleranceConfig,
    SparseToleranceGNNPolicy,
    train_sparse_tolerance_policy,
)
from rl.paths import MODEL_DIR


MODEL_SUFFIX = "_gnn_sparse_tol_scale_aware_span4ghz_dense_reference_budget"
_settings = dict(_scale_aware_settings)
_settings.update(
    # Keep the original T feature channel for architecture compatibility, but
    # feed it the same 3-degree quality threshold for every training target.
    fixed_tolerance_deg=3.0,
    redistribute_pruned_coupling=True,
    # A sparse candidate must meet T=3 deg and stay within 0.25 deg of the
    # same target's deterministic dense reference. Once it does, the full
    # feasible tier rewards lower q and lower actual total kappa equally.
    dense_reference_error_slack_deg=0.25,
    coupling_budget_reward_weight=0.5,
    feasible_phase_accuracy_weight=0.0,
)
for _kind in ("best", "current", "final"):
    _settings[f"{_kind}_checkpoint_file"] = str(
        MODEL_DIR
        / f"conditional_coupling_variable_m_3to10_lasers_{_kind}{MODEL_SUFFIX}.pt"
    )
config = SparseToleranceConfig(**_settings)


def main() -> dict:
    policy = SparseToleranceGNNPolicy(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    policy, optimizer, history = train_sparse_tolerance_policy(
        config, policy, optimizer, live_plot=LIVE_PLOT
    )
    return {"policy": policy, "optimizer": optimizer,
            "config": config, "history": history}


if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    budget_aware_sparse_training_result = main()
