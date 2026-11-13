"""Regression checks for density tasks, weighting, and shared backward graphs."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl import conditional_coupling_bernoulli_sparse as sparse
from rl import conditional_coupling_designer_variable_m as base
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import config as run_config


def test_density_bands_use_excess_links_and_exact_third_boundaries():
    # M=4 has a 3-link tree and 12-link all-to-all graph.
    np.testing.assert_array_equal(
        sparse.topology_density_band_ids(np.array([3, 5, 6, 8, 9, 12]), 4),
        [0, 0, 1, 1, 2, 2],
    )
    for size in range(3, 10):
        np.testing.assert_array_equal(
            sparse.topology_density_band_ids([size - 1, size * (size - 1)], size),
            [0, 2],
        )


def test_weighted_pcgrad_preserves_unconflicted_objective():
    combined, diagnostics = base.combine_task_gradients_pcgrad(
        torch.eye(2), 10.0, np.random.default_rng(7), task_weights=[1.0, 3.0])
    torch.testing.assert_close(combined, torch.tensor([0.25, 0.75]))
    assert diagnostics["negative_gradient_pair_fraction"] == 0.0
    opposed, _ = base.combine_task_gradients_pcgrad(
        torch.tensor([[2.0, 0.0], [-3.0, 0.0]]), 1.0,
        np.random.default_rng(7), task_weights=[1.0, 3.0])
    torch.testing.assert_close(opposed, torch.zeros(2))


@pytest.mark.parametrize("weights", [[1.0], [0.0, 1.0], [float("nan"), 1.0]])
def test_invalid_pcgrad_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="task_weights"):
        base.combine_task_gradients_pcgrad(
            torch.eye(2), 1.0, np.random.default_rng(0), task_weights=weights)


def small_config(**changes):
    settings = dict(
        training_n_lasers=(3, 4), n_lasers=3, targets_per_batch=4, n_jobs=1,
        phase_reward_mode="cosine", topology_sampling_mode="per_target",
        pcgrad_task_grouping="array_size_topology_density", pcgrad_topology_groups=3,
        candidates_per_target=72, actions_per_topology=8,
        topology_curriculum_start_iteration=0, topology_curriculum_end_iteration=500,
        node_hidden_sizes=(8,), node_embedding_dim=8, gnn_message_hidden_size=8,
        gnn_message_passing_steps=1, edge_hidden_sizes=(16,),
        coupling_budget_reference_by_m=((3, 1.0), (4, 1.0)),
    )
    settings.update(changes)
    return replace(run_config, **settings)


def fake_chunk(work):
    group, start, stop, item = work
    _, target_start, target_stop, *_ = item
    # Nonconstant rewards within each eight-action mask group.
    rewards = 0.5 + 0.3 * np.cos(np.arange(start, stop))
    return dict(size_batch_index=group, candidate_start_index=start,
                candidate_stop_index=stop, start_index=target_start,
                stop_index=target_stop,
                phase_reward=np.tile(rewards, (target_stop - target_start, 1)))


@pytest.mark.parametrize("iteration,expected_bands", [(0, ("dense",)),
                                                     (500, sparse.DENSITY_BANDS)])
def test_density_pcgrad_handles_curriculum_and_shared_graphs(
        monkeypatch, iteration, expected_bands):
    config = small_config()
    policy = sparse.BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    monkeypatch.setattr(base, "simulate_indexed_target_chunk", fake_chunk)
    monkeypatch.setattr(sparse, "phase_de_teacher_search",
                        lambda _work: pytest.fail("DE teacher is disabled"))
    combine = base.combine_task_gradients_pcgrad
    captured = {}

    def record(gradients, clip, rng, **kwargs):
        captured["weights"] = np.asarray(kwargs["task_weights"])
        return combine(gradients, clip, rng, **kwargs)

    monkeypatch.setattr(base, "combine_task_gradients_pcgrad", record)
    metrics = sparse.run_training_batch(policy, optimizer, config,
                                        np.random.default_rng(9), iteration)
    labels = [f"M{size}:{band}" for size in (3, 4) for band in expected_bands]
    assert metrics["pcgrad_task_labels"] == labels
    assert metrics["pcgrad_task_count"] == len(labels)
    assert metrics["de_teacher_searches"] == metrics["de_teacher_simulations"] == 0
    assert np.isfinite(metrics["gradient_norm"])
    assert 0 <= metrics["negative_gradient_pair_fraction"] <= 1
    weights = captured["weights"].reshape(2, -1)
    np.testing.assert_allclose(weights.sum(axis=1), [0.5, 0.5])
    assert np.all(weights > 0)


def test_split_tasks_preserve_original_gradients_before_projection(monkeypatch):
    config = small_config(gradient_clip=1e6, per_size_gradient_clip=1e6)
    legacy_config = replace(config, pcgrad_task_grouping="array_size")
    torch.manual_seed(12)
    policy = sparse.BernoulliSparseGNNPolicy(config)
    legacy_policy = deepcopy(policy)
    legacy_policy.config = legacy_config
    monkeypatch.setattr(base, "simulate_indexed_target_chunk", fake_chunk)

    def weighted_mean(gradients, _clip, _rng, *, task_weights=None):
        weights = (gradients.new_full((len(gradients),), 1 / len(gradients))
                   if task_weights is None else gradients.new_tensor(task_weights))
        return (gradients * weights[:, None]).sum(dim=0), {}

    monkeypatch.setattr(base, "combine_task_gradients_pcgrad", weighted_mean)
    for model, local_config in [(policy, config), (legacy_policy, legacy_config)]:
        torch.manual_seed(41)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        sparse.run_training_batch(model, optimizer, local_config,
                                  np.random.default_rng(9), 500)
    for actual, expected in zip(policy.parameters(), legacy_policy.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_old_checkpoint_defaults_to_array_size_pcgrad(tmp_path):
    config = small_config()
    policy = sparse.BernoulliSparseGNNPolicy(config)
    path = tmp_path / "old.pt"
    sparse.save_checkpoint(path, policy, torch.optim.Adam(policy.parameters()), config, 1, 0.5)
    saved = torch.load(path, weights_only=False)
    saved["config"].pop("pcgrad_task_grouping")
    torch.save(saved, path)
    _, restored, _ = sparse.load_checkpoint(path)
    assert restored.pcgrad_task_grouping == "array_size"
