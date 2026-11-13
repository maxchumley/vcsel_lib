"""Budget-aware sparse reward and post-pruning allocation invariants."""

import numpy as np
from pathlib import Path
import sys
from dataclasses import replace

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rl.archive.conditional_coupling_sparse_tolerance import (
    redistribute_retained_coupling,
    tiered_sparse_budget_reward,
)
from rl.archive import conditional_coupling_sparse_tolerance as sparse
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_budget_aware import (
    config as run_config,
)
from rl.archive.inspect_saved_coupling_model_variable_m_sparse_budget_aware import (
    _stack_validation_matrices,
)


def test_reallocation_keeps_mask_and_never_restores_full_dense_budget():
    dense = np.array([[[0.0, 0.6, 0.6],
                       [0.4, 0.0, 0.4],
                       [0.3, 0.3, 0.0]]])
    pruned = np.array([[[0.0, 0.6, 0.0],
                        [0.4, 0.0, 0.0],
                        [0.0, 0.3, 0.0]]])
    result = redistribute_retained_coupling(dense, pruned, 1.0)
    np.testing.assert_array_equal(result > 0, pruned > 0)
    assert np.all(result >= pruned)
    assert np.max(result) <= 1.0
    assert result.sum() > pruned.sum()
    assert result.sum() < dense.sum()


def test_budget_reward_prefers_cheaper_feasible_designs_only():
    kappa = np.zeros((1, 3, 3, 3))
    kappa[0, 0, 0, 1] = 1.0
    kappa[0, 1, 0, 1] = 0.3
    kappa[0, 2, 0, 1] = 0.1
    reward, feasible = tiered_sparse_budget_reward(
        np.array([[0.99, 0.99, 0.99]]),
        np.array([[2.0, 2.0, 10.0]]),
        np.array([3.0]),
        np.array([[0.5, 0.5, 0.1]]),
        kappa, 1.0, 5.0, 0.5,
    )
    assert feasible.tolist() == [[True, True, False]]
    assert reward[0, 1] > reward[0, 0] > reward[0, 2]
    assert reward[0, 2] < 0.5


def test_accuracy_first_reward_does_not_trade_one_degree_for_cheaper_links():
    kappa = np.zeros((1, 2, 3, 3))
    kappa[0, 0, 0, 1] = 1.0
    kappa[0, 1, 0, 1] = 0.1
    reward, feasible = tiered_sparse_budget_reward(
        np.array([[0.999, 0.999]]),
        np.array([[0.1, 1.1]]),
        np.array([3.0]),
        np.array([[0.9, 0.1]]),
        kappa, 1.0, 5.0, 0.5, phase_accuracy_weight=0.99,
    )
    assert feasible.all()
    assert reward[0, 0] > reward[0, 1]


def test_training_tolerance_is_fixed_in_every_stage():
    for iteration in (0, 1000, 1300, 2200, 4999):
        np.testing.assert_allclose(
            sparse.sample_allowable_tolerances(
                12, iteration, run_config, np.random.default_rng(4)
            ),
            3.0,
        )
        assert sparse.maximum_tolerance_at(iteration, run_config) == 3.0


def test_dense_reference_gate_rejects_cheaper_but_less_accurate_design():
    kappa = np.zeros((1, 3, 3, 3))
    kappa[0, 0, 0, 1] = 0.8
    kappa[0, 1, 0, 1] = 0.2
    kappa[0, 2, 0, 1] = 0.1
    reward, feasible = tiered_sparse_budget_reward(
        np.ones((1, 3)), np.array([[0.2, 0.3, 0.5]]),
        np.array([3.0]), np.array([[0.7, 0.3, 0.1]]),
        kappa, 1.0, 5.0, 0.5, phase_accuracy_weight=0.0,
        reference_error_deg=np.array([0.1]), reference_slack_deg=0.25,
    )
    assert feasible.tolist() == [[True, True, False]]
    assert reward[0, 1] > reward[0, 0] > reward[0, 2]


def test_evaluation_compares_sparse_with_dense_in_one_batch(monkeypatch):
    config = replace(run_config, n_lasers=3, training_n_lasers=(3, 4),
                     n_jobs=1, device="cpu")
    policy = sparse.SparseToleranceGNNPolicy(config)

    def fake_batch(targets, kappa, phi, config, **kwargs):
        assert kappa.shape == (1, 2, 3, 3)
        aligned = np.zeros((1, 2, 3))
        aligned[0, 0, 1:] = np.deg2rad(0.2)
        aligned[0, 1, 1:] = np.deg2rad(0.1)
        return {
            "phase_reward": np.ones((1, 2)),
            "aligned_achieved_phases_rad": aligned,
            "achieved_phases_rad": aligned.copy(),
            "orientation": np.ones((1, 2)),
        }

    monkeypatch.setattr(sparse.base, "simulate_candidates_parallel", fake_batch)
    result = sparse.evaluate_sparse_policy(
        policy, np.zeros(3), np.array([-0.1, 0.0, 0.1]), 3.0, config
    )
    np.testing.assert_allclose(result["dense_reference_error_deg"], [0.1])
    np.testing.assert_allclose(result["maximum_phase_error_deg"], [0.2])
    assert result["constraint_satisfied"].tolist() == [True]


def test_inspector_stacks_sparse_and_dense_cases_as_3d_matrices():
    sparse_kappa = np.eye(3)
    sparse_phi = np.zeros((3, 3))
    dense_kappa = np.full((1, 1, 3, 3), 2.0)
    dense_phi = np.full((1, 1, 3, 3), 0.5)
    kappa, phi = _stack_validation_matrices(
        sparse_kappa, sparse_phi, dense_kappa, dense_phi, 3
    )
    assert kappa.shape == phi.shape == (6, 3, 3)
    np.testing.assert_allclose(kappa[:3], np.repeat(sparse_kappa[None], 3, axis=0))
    np.testing.assert_allclose(kappa[3:], np.repeat(dense_kappa[0, 0][None], 3, axis=0))
    np.testing.assert_allclose(phi[3:], np.repeat(dense_phi[0, 0][None], 3, axis=0))


def test_sparse_training_batch_uses_new_reward_and_allocator(monkeypatch):
    config = replace(
        run_config, training_n_lasers=(3, 4), n_lasers=3,
        training_iterations=2201, targets_per_batch=2,
        candidates_per_target=4, candidates_per_worker_task=4,
        dense_replay_candidates=2, n_jobs=1, device="cpu",
    )
    policy = sparse.SparseToleranceGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)

    def fake_simulation(indexed_item):
        size_index, candidate_start, candidate_stop, work = indexed_item
        _, start, stop, targets, kappa, _, _, _ = work
        return {
            "size_batch_index": size_index,
            "candidate_start_index": candidate_start,
            "candidate_stop_index": candidate_stop,
            "start_index": start,
            "stop_index": stop,
            "phase_reward": np.ones(kappa.shape[:2]),
            "aligned_achieved_phases_rad": np.broadcast_to(
                targets[:, None, :],
                (len(targets), kappa.shape[1], kappa.shape[-1]),
            ).copy(),
        }

    monkeypatch.setattr(sparse.base, "simulate_indexed_target_chunk", fake_simulation)
    metrics = sparse.run_sparse_multisize_training_batch(
        policy, optimizer, config, np.random.default_rng(7), 2200
    )
    assert metrics["stage"] == "refinement"
    assert 0.0 <= metrics["normalized_coupling_budget"] <= 1.0
    assert 0.0 <= metrics["training_reward"] <= 1.0
    assert metrics["constraint_satisfaction_rate"] == 1.0
