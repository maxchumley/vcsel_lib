"""Focused tests for the selector-free sparse tolerance experiment."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from rl import conditional_coupling_sparse_tolerance as sparse


def small_config(**changes) -> sparse.SparseToleranceConfig:
    values = dict(
        n_lasers=4,
        training_n_lasers=(3, 4),
        training_iterations=20,
        targets_per_batch=2,
        candidates_per_target=10,
        dense_replay_candidates=2,
        candidates_per_worker_task=10,
        encoder_architecture="gnn",
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=False,
        enable_connected_edge_budgets=False,
        enable_learned_connected_sparsity=False,
        enable_learned_backbone_density=False,
        enable_conditional_backbone_budget=False,
        enable_budget_error_selector=False,
        monotonic_budget_selector=False,
        selector_use_deterministic_max_error=False,
        condition_on_edge_budget=False,
        use_relative_phase_retention=False,
        use_lexicographic_rank_advantages=False,
        include_dense_reference_candidate=False,
        detuning_span_ghz=4.0,
        detuning_curriculum_warmup_iterations=1,
        detuning_curriculum_iterations=4,
        sparse_stage_start_iteration=4,
        full_sparse_stage_start_iteration=8,
        refinement_stage_start_iteration=16,
        n_jobs=1,
        jupyter_mode=False,
    )
    values.update(changes)
    return sparse.SparseToleranceConfig(**values)


def test_tolerance_is_an_explicit_normalized_node_feature():
    config = small_config()
    policy = sparse.SparseToleranceGNNPolicy(config)
    target = np.zeros((2, config.n_lasers))
    detuning = np.zeros_like(target)
    encoded = sparse.encode_sparse_tolerance_context(
        target, detuning, np.array([18.0, 90.0]), policy, config
    ).detach().cpu().numpy()
    assert encoded.shape == (2, config.n_lasers, 4)
    np.testing.assert_allclose(encoded[0, :, 3], 0.1)
    np.testing.assert_allclose(encoded[1, :, 3], 0.5)
    np.testing.assert_allclose(encoded[0, :, :3], encoded[1, :, :3])


def test_scale_aware_detunings_separate_pattern_from_physical_span():
    config = small_config(scale_aware_detuning_features=True)
    policy = sparse.SparseToleranceGNNPolicy(config)
    target = np.zeros((2, config.n_lasers))
    detuning = np.array(
        [
            [-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0],
            [-3.0, -1.0, 1.0, 3.0],
        ]
    )
    encoded = sparse.encode_sparse_tolerance_context(
        target, detuning, 18.0, policy, config
    ).detach().cpu().numpy()
    assert encoded.shape == (2, config.n_lasers, 5)
    np.testing.assert_allclose(
        encoded[:, :, 2],
        np.tile([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], (2, 1)),
    )
    np.testing.assert_allclose(encoded[:, :, 3], 0.1)
    np.testing.assert_allclose(encoded[0, :, 4], 4.0)
    np.testing.assert_allclose(encoded[1, :, 4], 6.0)


def test_maximum_circular_error_wraps_across_pi_boundary():
    target = np.deg2rad(np.array([[0.0, 179.0, -179.0]]))
    achieved = np.deg2rad(np.array([[[0.0, -179.0, 179.0]]]))
    error = sparse.maximum_circular_phase_error_deg(target, achieved)
    np.testing.assert_allclose(error, [[2.0]], atol=1.0e-10)


def test_tiered_reward_prefers_accuracy_until_feasible_then_sparsity():
    phase = np.array([[0.8, 0.9, 0.9, 0.9]])
    error = np.array([[20.0, 10.0, 4.0, 4.0]])
    q = np.array([[0.0, 1.0, 1.0, 0.2]])
    reward, feasible = sparse.tiered_sparse_reward(
        phase, error, np.array([5.0]), q, tau_deg=5.0
    )
    assert reward[0, 1] > reward[0, 0]
    assert not feasible[0, 0] and not feasible[0, 1]
    assert feasible[0, 2] and feasible[0, 3]
    assert reward[0, 3] > reward[0, 2]


def test_infeasible_designs_receive_no_sparsity_benefit():
    phase = np.array([[0.95, 0.95]])
    error = np.array([[30.0, 30.0]])
    q = np.array([[0.0, 1.0]])
    reward, feasible = sparse.tiered_sparse_reward(
        phase, error, np.array([5.0]), q, tau_deg=5.0
    )
    assert not np.any(feasible)
    np.testing.assert_allclose(reward[0, 0], reward[0, 1])
    assert np.max(reward) < 0.5


def test_connected_pruning_obeys_exact_link_count_bounds():
    config = small_config(n_lasers=6, training_n_lasers=(5, 6))
    rng = np.random.default_rng(4)
    maximum = config.n_lasers * (config.n_lasers - 1)
    kappa = rng.uniform(0.1, 1.0, size=(3, config.n_lasers, config.n_lasers))
    for matrix in kappa:
        np.fill_diagonal(matrix, 0.0)
    phi = np.zeros_like(kappa)
    requested = sparse.q_to_active_link_counts(
        np.array([0.0, 0.4, 1.0]), config.n_lasers
    )
    pruned, _, rho = sparse.base.prune_coupling_batch_to_link_budget(
        kappa, phi, requested
    )
    assert requested[0] == config.n_lasers - 1
    assert requested[-1] == maximum
    np.testing.assert_array_equal(
        np.count_nonzero(pruned, axis=(1, 2)), requested
    )
    np.testing.assert_allclose(rho, requested / maximum)

    for matrix in pruned:
        adjacency = (matrix + matrix.T) > 0.0
        visited = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                if int(neighbor) not in visited:
                    visited.add(int(neighbor))
                    frontier.append(int(neighbor))
        assert len(visited) == config.n_lasers


def test_curriculum_tolerance_and_density_bounds():
    config = small_config()
    rng = np.random.default_rng(9)
    dense = sparse.sample_allowable_tolerances(20, 0, config, rng)
    mild = sparse.sample_allowable_tolerances(20, 5, config, rng)
    full = sparse.sample_allowable_tolerances(20, 12, config, rng)
    refined = sparse.sample_allowable_tolerances(20, 18, config, rng)
    np.testing.assert_allclose(dense, config.dense_stage_tolerance_deg)
    assert np.min(mild) >= config.tolerance_minimum_deg
    assert np.max(mild) <= config.mild_tolerance_maximum_deg
    assert np.min(full) >= config.tolerance_minimum_deg
    assert np.max(full) <= sparse.maximum_tolerance_at(12, config)
    assert np.any(full <= config.strict_tolerance_maximum_deg)
    assert np.max(refined) <= config.tolerance_maximum_deg
    assert sparse.minimum_q_at(0, config) == 1.0
    assert sparse.minimum_q_at(5, config) == config.mild_minimum_q
    assert 0.0 < sparse.minimum_q_at(12, config) < config.mild_minimum_q
    assert sparse.minimum_q_at(18, config) == 0.0
