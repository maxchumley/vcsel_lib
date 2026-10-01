from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from rl import conditional_coupling_designer_variable_m as variable_m
from rl import conditional_coupling_run_variable_m as variable_m_run


def small_config(**changes):
    config = variable_m.DesignerConfig(
        n_lasers=5,
        training_n_lasers=(3, 4, 5, 6, 7),
        training_iterations=1,
        targets_per_batch=2,
        candidates_per_target=4,
        node_hidden_sizes=(8,),
        node_embedding_dim=8,
        gru_hidden_size=6,
        edge_hidden_sizes=(12,),
        held_out_target_count=2,
        jupyter_mode=False,
    )
    return replace(config, **changes)


def selector_config(**changes):
    return small_config(
        encoder_architecture="gnn",
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=False,
        enable_connected_edge_budgets=False,
        condition_on_edge_budget=True,
        enable_learned_backbone_density=False,
        enable_conditional_backbone_budget=True,
        enable_budget_error_selector=True,
        monotonic_budget_selector=True,
        **changes,
    )


def test_conditional_budget_sweep_covers_every_exact_connected_count():
    for n_lasers in (3, 4, 10):
        minimum_links = n_lasers - 1
        maximum_links = n_lasers * (n_lasers - 1)
        feasible_count = maximum_links - minimum_links + 1
        observed = {
            int(
                variable_m.sample_conditional_link_counts(
                    1, n_lasers, iteration
                )[0]
            )
            for iteration in range(feasible_count)
        }
        assert observed == set(range(minimum_links, maximum_links + 1))


def test_multibudget_assignment_preserves_candidate_count_and_endpoints():
    rng = np.random.default_rng(9)
    for n_lasers, expected_levels in ((3, 5), (10, 8)):
        assignments = variable_m.sample_multibudget_candidate_counts(
            target_count=3,
            candidates_per_target=64,
            n_lasers=n_lasers,
            budget_levels_per_target=8,
            rng=rng,
            iteration=4,
        )
        assert assignments.shape == (3, 64)
        for row in assignments:
            levels, counts = np.unique(row, return_counts=True)
            assert len(levels) == expected_levels
            assert levels[0] == n_lasers - 1
            assert levels[-1] == n_lasers * (n_lasers - 1)
            assert counts.max() - counts.min() <= 1


def test_72_candidates_leave_64_stochastic_candidates_for_eight_budgets():
    assignments = variable_m.sample_multibudget_candidate_counts(
        target_count=2,
        candidates_per_target=72,
        n_lasers=10,
        budget_levels_per_target=8,
        rng=np.random.default_rng(12),
        iteration=0,
    )
    deterministic = variable_m.deterministic_budget_candidate_mask(
        assignments
    )

    assert np.all(np.sum(deterministic, axis=1) == 8)
    assert np.all(np.sum(~deterministic, axis=1) == 64)
    for target_index in range(len(assignments)):
        for link_count in np.unique(assignments[target_index]):
            budget_mask = assignments[target_index] == link_count
            assert np.count_nonzero(
                deterministic[target_index] & budget_mask
            ) == 1
            assert np.count_nonzero(
                ~deterministic[target_index] & budget_mask
            ) == 8


def test_nonincreasing_isotonic_fit_removes_upward_steps():
    measured = np.array([0.50, 0.35, 0.40, 0.15])
    fitted = variable_m.nonincreasing_isotonic_fit(measured)

    assert np.all(np.diff(fitted) <= 1.0e-12)
    np.testing.assert_allclose(fitted, [0.50, 0.375, 0.375, 0.15])


def test_monotonic_selector_never_increases_budget_with_tolerance():
    config = selector_config(n_lasers=5)
    policy = variable_m.GNNPolicyNetwork(config)
    target = np.linspace(0.0, np.pi, config.n_lasers)[None, :]
    detuning = np.linspace(-1.0, 1.0, config.n_lasers)[None, :]
    features = variable_m.encode_for_policy(
        target,
        policy,
        config,
        detuning,
        active_link_counts=np.array([20]),
    ).repeat(20, 1, 1)
    tolerances = torch.linspace(0.0, 1.0, 20)

    predicted_q = policy.predict_normalized_budget(features, tolerances)

    assert torch.all(torch.diff(predicted_q) <= 1.0e-7)


def test_budget_selector_predicts_one_connected_exact_budget():
    config = selector_config(n_lasers=5)
    policy = variable_m.GNNPolicyNetwork(config)
    target = np.linspace(0.0, np.pi, config.n_lasers)
    detuning = np.linspace(-1.0, 1.0, config.n_lasers)

    selection = variable_m.predict_conditional_link_budget(
        policy,
        target,
        detuning,
        config,
        allowable_rms_phase_error_deg=5.0,
    )

    assert 0.0 <= selection["normalized_budget_q"] <= 1.0
    assert config.n_lasers - 1 <= selection["active_link_count"]
    assert selection["active_link_count"] <= config.n_lasers * (
        config.n_lasers - 1
    )
    assert 0.0 <= selection["normalized_sparsity_score"] <= 1.0


def test_budget_error_selector_loss_trains_both_auxiliary_heads():
    config = selector_config(n_lasers=5)
    policy = variable_m.GNNPolicyNetwork(config)
    targets = np.stack(
        (
            np.linspace(0.0, np.pi, config.n_lasers),
            np.linspace(0.0, -np.pi, config.n_lasers),
        )
    )
    detunings = np.stack(
        (
            np.linspace(-1.0, 1.0, config.n_lasers),
            np.linspace(-2.0, 2.0, config.n_lasers),
        )
    )
    active_counts = np.array(
        [
            [config.n_lasers - 1] * 2
            + [config.n_lasers * (config.n_lasers - 1)] * 2,
            [config.n_lasers - 1] * 2
            + [config.n_lasers * (config.n_lasers - 1)] * 2,
        ]
    )
    features = variable_m.encode_for_policy(
        targets,
        policy,
        config,
        detunings,
        active_link_counts=np.full(
            len(targets), config.n_lasers * (config.n_lasers - 1)
        ),
    )
    achieved = np.repeat(targets[:, None, :], 4, axis=1)
    achieved[:, :2, 1:] += np.deg2rad(30.0)
    achieved[:, 2:, 1:] += np.deg2rad(5.0)

    loss, error_mae_deg, budget_mae = variable_m.budget_error_selector_loss(
        policy,
        features,
        targets,
        active_counts,
        achieved,
        config.n_lasers,
        config,
        np.random.default_rng(3),
    )
    loss.backward()

    assert np.isfinite(error_mae_deg)
    assert 0.0 <= budget_mae <= 1.0
    assert policy.budget_error_head[-1].bias.grad is not None
    assert policy.budget_selector_head[-1].bias.grad is not None


def test_deterministic_max_error_selector_uses_policy_mean_candidates():
    config = selector_config(
        n_lasers=5,
        candidates_per_target=6,
        selector_budget_levels_per_target=2,
        selector_use_deterministic_max_error=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    targets = np.linspace(0.0, np.pi, config.n_lasers)[None, :]
    detunings = np.linspace(-1.0, 1.0, config.n_lasers)[None, :]
    active_counts = np.array([[4, 4, 4, 20, 20, 20]])
    deterministic = np.array(
        [[True, False, False, True, False, False]]
    )
    features = variable_m.encode_for_policy(
        targets,
        policy,
        config,
        detunings,
        active_link_counts=np.array([20]),
    )
    achieved = np.repeat(targets[:, None, :], 6, axis=1)
    achieved[:, 0, 1:] += np.deg2rad(35.0)
    achieved[:, 3, 1:] += np.deg2rad(5.0)
    # Stochastic candidates deliberately contain unrelated larger errors.
    achieved[:, [1, 2, 4, 5], 1:] += np.deg2rad(90.0)

    loss, error_mae_deg, budget_mae = (
        variable_m.budget_error_selector_loss(
            policy,
            features,
            targets,
            active_counts,
            achieved,
            config.n_lasers,
            config,
            np.random.default_rng(3),
            deterministic_candidate_mask=deterministic,
        )
    )
    loss.backward()

    assert np.isfinite(error_mae_deg)
    assert 0.0 <= budget_mae <= 1.0
    assert policy.budget_error_head[-1].bias.grad is not None
    assert policy.budget_selector_head[-1].bias.grad is not None


def test_budget_advantages_exclude_deterministic_candidates():
    rewards = np.array([[100.0, 1.0, 3.0, -100.0, 4.0, 8.0]])
    budgets = np.array([[4, 4, 4, 20, 20, 20]])
    stochastic = np.array([[False, True, True, False, True, True]])

    advantages = variable_m.budget_group_standardized_advantages(
        rewards, budgets, candidate_mask=stochastic
    )

    np.testing.assert_allclose(advantages, [[0.0, -1.0, 1.0, 0.0, -1.0, 1.0]])


def test_multisize_selector_reuses_existing_simulation_results(monkeypatch):
    config = selector_config(
        training_n_lasers=(3, 4),
        targets_per_batch=2,
        candidates_per_target=6,
        candidates_per_worker_task=6,
        selector_budget_levels_per_target=2,
        selector_use_deterministic_max_error=True,
        gradient_combination_mode="pcgrad",
        n_jobs=1,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)

    def fake_simulation(indexed_work_item):
        (
            size_batch_index,
            candidate_start,
            candidate_stop,
            work_item,
        ) = indexed_work_item
        (
            chunk_index,
            start_index,
            stop_index,
            targets,
            kappa,
            _phi_p,
            _detunings,
            _size_config,
        ) = work_item
        target_count, candidate_count = kappa.shape[:2]
        aligned = np.repeat(
            variable_m.make_relative_to_laser_1(targets)[:, None, :],
            candidate_count,
            axis=1,
        )
        active_counts = np.count_nonzero(kappa, axis=(2, 3))
        maximum_links = _size_config.n_lasers * (
            _size_config.n_lasers - 1
        )
        minimum_links = _size_config.n_lasers - 1
        relative_budget = (active_counts - minimum_links) / (
            maximum_links - minimum_links
        )
        error_rad = 0.55 - 0.45 * relative_budget
        aligned[:, :, 1:] += error_rad[:, :, None]
        return {
            "size_batch_index": size_batch_index,
            "chunk_index": chunk_index,
            "start_index": start_index,
            "stop_index": stop_index,
            "candidate_start_index": candidate_start,
            "candidate_stop_index": candidate_stop,
            "phase_reward": np.repeat(
                np.linspace(0.4, 0.9, candidate_count)[None, :],
                target_count,
                axis=0,
            ),
            "aligned_achieved_phases_rad": aligned,
        }

    monkeypatch.setattr(
        variable_m, "simulate_indexed_target_chunk", fake_simulation
    )
    metrics = variable_m.run_multisize_training_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(4),
        iteration=0,
    )

    assert np.isfinite(metrics["loss"])
    for size_metrics in metrics["size_metrics"].values():
        assert np.isfinite(size_metrics["error_prediction_mae_deg"])
        assert 0.0 <= size_metrics["selector_budget_mae"] <= 1.0


def test_one_policy_instance_accepts_multiple_array_sizes():
    config = small_config()
    policy = variable_m.PolicyNetwork(config)
    parameter_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in policy.named_parameters()
    }

    for n_lasers in (3, 5, 7, 10):
        target = np.linspace(-np.pi, np.pi, n_lasers)[None, :]
        detuning = np.linspace(-5.0, 5.0, n_lasers)[None, :]
        size_config = replace(config, n_lasers=n_lasers)
        node_features = variable_m.encode_for_policy(
            target, policy, size_config, detuning
        )
        means = policy(node_features)

        assert node_features.shape == (1, n_lasers, 3)
        assert means.shape == (1, 2 * n_lasers * (n_lasers - 1))
        assert {
            name: tuple(parameter.shape)
            for name, parameter in policy.named_parameters()
        } == parameter_shapes


def test_gradients_from_different_sizes_reach_same_modules():
    config = small_config()
    policy = variable_m.PolicyNetwork(config)
    loss = 0.0
    for n_lasers in (3, 5, 7, 10):
        node_features = torch.randn(2, n_lasers, 3)
        loss = loss + policy(node_features).square().mean()
    loss.backward()

    assert policy.node_encoder[0].weight.grad is not None
    assert policy.global_encoder[0].weight.grad is not None
    assert policy.global_encoder[2].weight.grad is not None
    assert policy.edge_decoder[-1].weight.grad is not None


def test_scalar_log_stds_broadcast_to_dynamic_action_width():
    policy = variable_m.PolicyNetwork(small_config())
    assert policy.log_std_kappa.ndim == 0
    assert policy.log_std_phi.ndim == 0

    for n_lasers in (3, 10):
        distribution = policy.distribution(
            torch.zeros(2, n_lasers, 3)
        )
        expected = 2 * n_lasers * (n_lasers - 1)
        assert distribution.mean.shape == (2, expected)
        assert distribution.stddev.shape == (2, expected)


def test_bigru_conditions_edge_means_and_exploration_on_array_size():
    config = small_config(
        encoder_architecture="bigru",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)

    # 4*GRU width + four physical edge features + two M features.
    assert policy.edge_decoder[0].in_features == 4 * 6 + 4 + 2
    with torch.no_grad():
        policy.size_log_std_head.weight.copy_(torch.eye(2))
    std_m2 = policy.distribution(torch.zeros(1, 2, 3)).stddev[0, 0]
    std_m5 = policy.distribution(torch.zeros(1, 5, 3)).stddev[0, 0]
    assert not torch.isclose(std_m2, std_m5)


def test_bigru_size_modulation_starts_as_identity_and_receives_gradients():
    base_config = small_config(
        encoder_architecture="bigru",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
    )
    modulated_config = replace(
        base_config, modulate_edge_decoder_by_n_lasers=True
    )
    base_policy = variable_m.BiGRUPolicyNetwork(base_config)
    modulated_policy = variable_m.BiGRUPolicyNetwork(modulated_config)
    incompatible = modulated_policy.load_state_dict(
        base_policy.state_dict(), strict=False
    )

    assert set(incompatible.missing_keys) == {
        "edge_size_modulation.weight",
        "edge_size_modulation.bias",
    }
    assert incompatible.unexpected_keys == []
    assert modulated_policy.edge_size_modulation.out_features == 24

    for n_lasers in (2, 5):
        node_features = torch.randn(2, n_lasers, 3)
        base_distribution = base_policy.distribution(node_features)
        modulated_distribution = modulated_policy.distribution(node_features)
        torch.testing.assert_close(
            modulated_distribution.mean, base_distribution.mean
        )
        torch.testing.assert_close(
            modulated_distribution.stddev, base_distribution.stddev
        )

    node_features = torch.randn(2, 3, 3)
    modulated_policy(node_features).square().mean().backward()
    assert modulated_policy.edge_size_modulation.weight.grad is not None
    assert torch.count_nonzero(
        modulated_policy.edge_size_modulation.weight.grad
    ) > 0


def test_bigru_predicts_one_exploration_width_per_edge_action():
    config = small_config(
        encoder_architecture="bigru",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)

    assert policy.edge_decoder[-1].out_features == 4
    for n_lasers in (2, 5):
        distribution = policy.distribution(
            torch.zeros(2, n_lasers, 3)
        )
        expected = 2 * n_lasers * (n_lasers - 1)
        assert distribution.mean.shape == (2, expected)
        assert distribution.stddev.shape == (2, expected)
        torch.testing.assert_close(
            distribution.stddev,
            torch.full_like(
                distribution.stddev,
                np.exp(config.initial_log_std),
            ),
        )

    distribution = policy.distribution(torch.zeros(2, 2, 3))
    sampled_actions = (distribution.mean + 2.0 * distribution.stddev).detach()
    (-distribution.log_prob(sampled_actions).mean()).backward()
    assert torch.count_nonzero(policy.edge_decoder[-1].weight.grad[2:]) > 0


def test_gnn_accepts_multiple_sizes_without_explicit_m_conditioning():
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        modulate_edge_decoder_by_n_lasers=False,
        edge_conditioned_log_std=True,
        gnn_message_passing_steps=2,
        gnn_message_hidden_size=10,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    parameter_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in policy.named_parameters()
    }

    loss = 0.0
    for n_lasers in (2, 3, 5, 9):
        distribution = policy.distribution(
            torch.randn(2, n_lasers, 3)
        )
        expected_width = 2 * n_lasers * (n_lasers - 1)
        assert distribution.mean.shape == (2, expected_width)
        assert distribution.stddev.shape == (2, expected_width)
        loss = loss + distribution.mean.square().mean()

    loss.backward()
    assert policy.node_encoder[0].weight.grad is not None
    assert policy.message_layers[0].message_mlp[0].weight.grad is not None
    assert policy.message_layers[0].update_mlp[0].weight.grad is not None
    assert policy.edge_decoder[-1].weight.grad is not None
    assert not hasattr(policy, "edge_size_modulation")
    assert not hasattr(policy, "bigru")
    assert {
        name: tuple(parameter.shape)
        for name, parameter in policy.named_parameters()
    } == parameter_shapes


def test_gnn_bounded_degree_conditioning_is_shared_and_bounded(tmp_path):
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
        gnn_message_passing_steps=2,
        gnn_message_hidden_size=10,
    )
    policy = variable_m.GNNPolicyNetwork(config)

    expected_features = {
        2: [1.0, 1.0],
        3: [0.5, 0.25],
        9: [0.125, 0.015625],
    }
    for n_lasers, expected in expected_features.items():
        features = policy.bounded_degree_features(
            n_lasers,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(features, torch.tensor(expected))
        assert torch.all((features >= 0.0) & (features <= 1.0))

        distribution = policy.distribution(
            torch.randn(2, n_lasers, 3)
        )
        expected_width = 2 * n_lasers * (n_lasers - 1)
        assert distribution.mean.shape == (2, expected_width)

    message_input = policy.message_layers[0].message_mlp[0]
    edge_input = policy.edge_decoder[0]
    assert message_input.in_features == 2 * config.node_embedding_dim + 6
    assert edge_input.in_features == 2 * config.node_embedding_dim + 6
    assert not hasattr(policy, "edge_size_modulation")

    checkpoint = tmp_path / "bounded_degree_gnn.pt"
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    node_features = torch.randn(2, 5, 3)
    expected = policy.distribution(node_features)
    variable_m.save_checkpoint(
        checkpoint,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.8,
    )
    loaded_policy, _, loaded_config, _ = variable_m.load_checkpoint(checkpoint)
    actual = loaded_policy.distribution(node_features)
    assert loaded_config.condition_on_bounded_degree is True
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.stddev, expected.stddev)


def test_bounded_degree_conditioning_rejects_duplicate_size_features():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        small_config(
            encoder_architecture="gnn",
            condition_on_n_lasers=True,
            condition_on_bounded_degree=True,
        )


def test_bounded_degree_stable_lr_runner_is_an_isolated_restart():
    from rl import (
        conditional_coupling_run_variable_m_gnn_pcgrad_bounded_degree
        as original_run,
    )
    from rl import (
        conditional_coupling_run_variable_m_gnn_pcgrad_bounded_degree_stable_lr
        as stable_run,
    )

    config = stable_run.config
    assert stable_run.TRAIN_NEW_MODEL is True
    assert config.learning_rate == pytest.approx(3.0e-4)
    assert config.learning_rate_switch_iteration == 1500
    assert config.learning_rate_after_switch == pytest.approx(1.0e-4)
    assert config.detuning_curriculum_warmup_iterations == 300
    assert config.detuning_curriculum_iterations == 1200
    assert config.training_iterations == 2500
    assert config.targets_per_batch == 24
    assert config.candidates_per_target == 48
    assert config.condition_on_bounded_degree is True
    assert config.gradient_combination_mode == "pcgrad"
    assert stable_run.FILE_SUFFIX in config.current_checkpoint_file
    assert (
        config.current_checkpoint_file
        != original_run.config.current_checkpoint_file
    )


def test_gnn_is_permutation_equivariant():
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=4,
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        modulate_edge_decoder_by_n_lasers=False,
        edge_conditioned_log_std=True,
        gnn_message_passing_steps=2,
        gnn_message_hidden_size=10,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    node_features = torch.randn(2, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    original = policy(node_features)
    permuted = policy(node_features[:, permutation])
    receivers, sources = variable_m.directed_link_indices(4)
    link_count = len(receivers)

    for action_offset in (0, link_count):
        original_matrix = torch.zeros(2, 4, 4)
        original_matrix[:, receivers, sources] = original[
            :, action_offset : action_offset + link_count
        ]
        permuted_matrix = torch.zeros(2, 4, 4)
        permuted_matrix[:, receivers, sources] = permuted[
            :, action_offset : action_offset + link_count
        ]
        expected = original_matrix[:, permutation][:, :, permutation]
        torch.testing.assert_close(permuted_matrix, expected)


def test_gnn_checkpoint_round_trip(tmp_path):
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        modulate_edge_decoder_by_n_lasers=False,
        edge_conditioned_log_std=True,
        gnn_message_passing_steps=2,
        gnn_message_hidden_size=10,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "gnn.pt"
    node_features = torch.randn(2, 4, 3)
    expected = policy.distribution(node_features)

    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy, _, loaded_config, metadata = variable_m.load_checkpoint(path)
    actual = loaded_policy.distribution(node_features)

    assert saved["format"] == variable_m.GNN_CHECKPOINT_FORMAT
    assert saved["encoder_architecture"] == "gnn"
    assert isinstance(loaded_policy, variable_m.GNNPolicyNetwork)
    assert loaded_config.encoder_architecture == "gnn"
    assert loaded_config.modulate_edge_decoder_by_n_lasers is False
    assert metadata == {"iteration": 4, "held_out_reward": 0.7}
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.stddev, expected.stddev)


def test_bigru_smooth_edge_log_std_bounds_preserve_initial_scale_and_gradient():
    config = small_config(
        encoder_architecture="bigru",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        smooth_edge_log_std_bounds=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)

    distribution = policy.distribution(torch.zeros(2, 2, 3))
    torch.testing.assert_close(
        distribution.stddev,
        torch.full_like(distribution.stddev, np.exp(config.initial_log_std)),
    )

    raw_below_floor = torch.tensor(-6.0, requires_grad=True)
    bounded = variable_m.smoothly_bound_log_std(
        raw_below_floor,
        config.minimum_log_std,
        config.maximum_log_std,
    )
    bounded.backward()
    assert bounded > config.minimum_log_std
    assert bounded < config.maximum_log_std
    assert raw_below_floor.grad > 0.0


def test_minimum_log_std_schedule_holds_then_anneals_to_final_floor():
    config = small_config(
        enable_minimum_log_std_annealing=True,
        initial_minimum_log_std=-1.5,
        minimum_log_std=-3.0,
        minimum_log_std_anneal_start_iteration=1000,
        minimum_log_std_anneal_iterations=500,
    )

    assert variable_m.minimum_log_std_at(0, config) == pytest.approx(-1.5)
    assert variable_m.minimum_log_std_at(1000, config) == pytest.approx(-1.5)
    assert variable_m.minimum_log_std_at(1250, config) == pytest.approx(-2.25)
    assert variable_m.minimum_log_std_at(1500, config) == pytest.approx(-3.0)
    assert variable_m.minimum_log_std_at(2000, config) == pytest.approx(-3.0)


def test_array_size_curriculum_adds_one_size_per_interval():
    config = small_config(
        training_n_lasers=tuple(range(2, 10)),
        enable_array_size_curriculum=True,
        array_size_curriculum_initial_count=1,
        array_size_curriculum_start_iteration=500,
        array_size_curriculum_add_interval=100,
    )

    expected = {
        0: (2,),
        499: (2,),
        500: (2, 3),
        599: (2, 3),
        600: (2, 3, 4),
        1000: tuple(range(2, 9)),
        1100: tuple(range(2, 10)),
        1199: tuple(range(2, 10)),
    }
    for iteration, active_sizes in expected.items():
        assert variable_m.active_training_n_lasers_at(
            iteration, config
        ) == active_sizes


def test_bigru_edge_exploration_checkpoint_round_trip(tmp_path):
    config = small_config(
        encoder_architecture="bigru",
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "bigru_edge_std.pt"

    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy, _, loaded_config, _ = variable_m.load_checkpoint(path)

    assert saved["format"] == variable_m.BIGRU_EDGE_STD_CHECKPOINT_FORMAT
    assert loaded_config.edge_conditioned_log_std is True
    assert loaded_policy.edge_decoder[-1].out_features == 4


def test_bigru_size_modulation_checkpoint_round_trip(tmp_path):
    config = small_config(
        encoder_architecture="bigru",
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        modulate_edge_decoder_by_n_lasers=True,
        edge_conditioned_log_std=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "bigru_size_modulated.pt"
    node_features = torch.randn(2, 3, 3)
    expected = policy.distribution(node_features)

    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy, _, loaded_config, _ = variable_m.load_checkpoint(path)
    actual = loaded_policy.distribution(node_features)

    assert saved["format"] == variable_m.BIGRU_MODULATED_CHECKPOINT_FORMAT
    assert loaded_config.modulate_edge_decoder_by_n_lasers is True
    assert loaded_policy.edge_size_modulation.out_features == 24
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.stddev, expected.stddev)


def test_dense_bigru_upgrades_to_initially_equivalent_sparse_policy():
    config = small_config(
        encoder_architecture="bigru",
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
    )
    dense_policy = variable_m.BiGRUPolicyNetwork(config)
    node_features = torch.randn(2, 3, 3)
    dense_mean = dense_policy(node_features).detach()
    dense_std = dense_policy.distribution(node_features).stddev.detach()

    sparse_policy, sparse_config = variable_m.add_sparse_gates_to_policy(
        dense_policy, config
    )

    assert sparse_config.enable_sparse_gates is True
    torch.testing.assert_close(sparse_policy(node_features), dense_mean)
    torch.testing.assert_close(
        sparse_policy.distribution(node_features).stddev, dense_std
    )
    assert torch.all(sparse_policy.deterministic_gates(node_features) == 1)
    torch.testing.assert_close(
        sparse_policy.gate_distribution(node_features).probs,
        torch.full((2, 6), torch.sigmoid(torch.tensor(5.0))),
    )


def test_dense_gnn_upgrades_to_initially_equivalent_sparse_policy():
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
    )
    dense_policy = variable_m.GNNPolicyNetwork(config)
    node_features = torch.randn(2, 3, 3)
    dense_mean = dense_policy(node_features).detach()
    dense_std = dense_policy.distribution(node_features).stddev.detach()

    sparse_policy, sparse_config = variable_m.add_sparse_gates_to_policy(
        dense_policy, config
    )

    assert isinstance(sparse_policy, variable_m.GNNPolicyNetwork)
    assert sparse_config.enable_sparse_gates is True
    torch.testing.assert_close(sparse_policy(node_features), dense_mean)
    torch.testing.assert_close(
        sparse_policy.distribution(node_features).stddev, dense_std
    )
    assert torch.all(sparse_policy.deterministic_gates(node_features) == 1)
    torch.testing.assert_close(
        sparse_policy.gate_distribution(node_features).probs,
        torch.full((2, 6), torch.sigmoid(torch.tensor(5.0))),
    )


def test_sparse_sampler_guarantees_dense_and_single_edge_probes():
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    node_features = torch.randn(2, 3, 3)
    actions, gates, log_probabilities = (
        variable_m.sample_sparse_coupling_designs(
            policy,
            node_features,
            candidates_per_target=8,
            include_dense_reference=True,
            single_edge_probe_fraction=0.25,
            exploratory_mask_fraction=0.25,
        )
    )

    assert gates.shape == (2, 8, 6)
    torch.testing.assert_close(
        actions[:, 0], policy.distribution(node_features).mean
    )
    assert torch.all(gates[:, 0] == 1)
    assert torch.all(log_probabilities[:, 0] == 0)
    assert torch.all(gates[:, 1:3].sum(dim=2) == 5)
    (-log_probabilities.mean()).backward()
    assert policy.edge_gate_head.bias.grad is not None


def test_connected_budget_sampler_has_exact_counts_and_no_disconnected_nodes():
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        enable_connected_edge_budgets=True,
        condition_on_edge_budget=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    targets = np.zeros((2, 3))
    active_link_counts = np.array([2, 4])
    node_features = variable_m.encode_for_policy(
        targets,
        policy,
        config,
        active_link_counts=active_link_counts,
    )
    actions, gates, log_probabilities = (
        variable_m.sample_connected_coupling_designs(
            policy,
            node_features,
            candidates_per_target=8,
            active_link_counts=active_link_counts,
        )
    )

    assert node_features.shape == (2, 3, 4)
    np.testing.assert_allclose(
        node_features[:, :, 3].numpy(),
        np.array([[2 / 6] * 3, [4 / 6] * 3]),
    )
    assert actions.shape == (2, 8, 12)
    assert gates.shape == (2, 8, 6)
    assert log_probabilities.shape == (2, 8)
    np.testing.assert_array_equal(
        gates.sum(dim=2).numpy(),
        np.array([[2] * 8, [4] * 8]),
    )

    receivers, sources = variable_m.directed_link_indices(3)
    for gate in gates.reshape(-1, 6).numpy():
        adjacency = np.zeros((3, 3), dtype=bool)
        active = np.flatnonzero(gate)
        adjacency[receivers[active], sources[active]] = True
        adjacency |= adjacency.T
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        assert reached == {0, 1, 2}

    (-log_probabilities.mean()).backward()
    assert policy.edge_gate_head.weight.grad is not None


def test_learned_connected_sampler_makes_link_count_a_policy_action():
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        initial_gate_logit=0.0,
        enable_learned_connected_sparsity=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    node_features = variable_m.encode_for_policy(
        np.zeros((2, 3)), policy, config
    )
    _, gates, log_probabilities = (
        variable_m.sample_learned_connected_coupling_designs(
            policy, node_features, candidates_per_target=32
        )
    )

    active_counts = gates.sum(dim=2)
    assert torch.all(active_counts >= 2)
    assert torch.all(active_counts <= 6)
    assert torch.any(active_counts < 6)
    assert torch.unique(active_counts).numel() > 1

    receivers, sources = variable_m.directed_link_indices(3)
    for gate in gates.reshape(-1, 6).numpy():
        adjacency = np.zeros((3, 3), dtype=bool)
        active = np.flatnonzero(gate)
        adjacency[receivers[active], sources[active]] = True
        adjacency |= adjacency.T
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency.astype(int)
        assert np.linalg.matrix_rank(laplacian) == 2

    sparse_advantage = -(active_counts - active_counts.mean(dim=1, keepdim=True))
    loss = -(log_probabilities * sparse_advantage.detach()).mean()
    loss.backward()
    assert policy.edge_gate_head.weight.grad is not None
    assert torch.linalg.vector_norm(policy.edge_gate_head.weight.grad) > 0


def test_learned_connected_sampler_can_reserve_dense_reference():
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3,),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        initial_gate_logit=0.0,
        enable_learned_connected_sparsity=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    node_features = variable_m.encode_for_policy(
        np.zeros((2, 3)), policy, config
    )

    _, gates, log_probabilities = (
        variable_m.sample_learned_connected_coupling_designs(
            policy,
            node_features,
            candidates_per_target=16,
            include_dense_reference=True,
        )
    )

    torch.testing.assert_close(gates[:, 0], torch.ones(2, 6))
    assert torch.any(gates[:, 1:] < 1.0)
    (-log_probabilities.mean()).backward()
    assert policy.edge_gate_head.weight.grad is not None


def test_normalized_squared_coupling_cost_uses_per_link_cap():
    kappa = np.zeros((3, 3, 3))
    off_diagonal = ~np.eye(3, dtype=bool)
    kappa[0, off_diagonal] = 5.0
    kappa[1, 0, 1] = 5.0
    kappa[2, off_diagonal] = 2.5

    cost = variable_m.normalized_squared_coupling_cost(kappa, 5.0)

    np.testing.assert_allclose(cost, np.array([1.0, 1.0 / 6.0, 0.25]))


def test_backbone_design_uses_only_displayed_links_and_stays_connected():
    n_lasers = 5
    config = small_config(
        n_lasers=n_lasers,
        normalize_incoming_coupling_by_degree=True,
    )
    kappa = np.arange(1, n_lasers * n_lasers + 1, dtype=float).reshape(
        n_lasers, n_lasers
    )
    np.fill_diagonal(kappa, 0.0)
    phi = np.linspace(-np.pi, np.pi, n_lasers * n_lasers).reshape(
        n_lasers, n_lasers
    )
    design = {
        "kappa_per_ns": kappa,
        "phi_p_rad": phi,
        "magnitude_gates": np.ones(n_lasers * (n_lasers - 1)),
    }

    backbone = variable_m_run.apply_relative_coupling_backbone(
        design, config, edge_limit_factor=1.5
    )

    active = backbone["kappa_per_ns"] > 0.0
    assert backbone["active_link_count"] == 8
    assert np.count_nonzero(active) == 8
    assert np.all(backbone["phi_p_rad"][~active] == 0.0)
    receivers, sources = variable_m.directed_link_indices(n_lasers)
    np.testing.assert_array_equal(
        backbone["magnitude_gates"], active[receivers, sources]
    )
    np.testing.assert_array_equal(design["kappa_per_ns"], kappa)

    # Weak connectivity: traverse the undirected projection from laser zero.
    reached = {0}
    while True:
        expanded = reached | {
            receiver
            for receiver in range(n_lasers)
            for source in reached
            if active[receiver, source] or active[source, receiver]
        }
        if expanded == reached:
            break
        reached = expanded
    assert reached == set(range(n_lasers))


def test_coupling_cost_is_strictly_third_priority_after_link_count():
    phase = np.full((1, 3), 0.99)
    # Candidate 0 has two strong links; candidate 1 has three zero-cost links;
    # candidate 2 has two weaker links. Fewer links must beat lower cost, then
    # lower cost must break the tie at equal link count.
    active_fraction = np.array([[2.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0]])
    coupling_cost = np.array([[1.0, 0.0, 0.25]])

    reward, sparsity_bonus, coupling_bonus, successful = (
        variable_m.successful_sparse_reward_components(
            phase,
            active_fraction,
            coupling_cost,
            n_lasers=3,
            phase_threshold=0.98,
            sparsity_weight=0.05,
            coupling_cost_tiebreak_fraction=0.25,
        )
    )

    assert np.all(successful)
    assert reward[0, 0] > reward[0, 1]
    assert reward[0, 2] > reward[0, 0]
    assert sparsity_bonus[0, 0] == pytest.approx(sparsity_bonus[0, 2])
    assert coupling_bonus[0, 2] > coupling_bonus[0, 0]


def test_relative_phase_retention_uses_per_target_best_candidate():
    phase = np.array(
        [
            [0.999, 0.996, 0.990],
            [0.950, 0.947, 0.940],
        ]
    )
    active_fraction = np.full_like(phase, 0.5)
    coupling_cost = np.full_like(phase, 0.25)

    reward, _, _, eligible = (
        variable_m.successful_sparse_reward_components(
            phase,
            active_fraction,
            coupling_cost,
            n_lasers=3,
            phase_threshold=0.98,
            sparsity_weight=0.05,
            coupling_cost_tiebreak_fraction=0.25,
            phase_reference=np.max(phase, axis=1),
            phase_retention_tolerance=0.005,
        )
    )

    np.testing.assert_array_equal(
        eligible,
        np.array(
            [
                [True, True, False],
                [True, True, False],
            ]
        ),
    )
    assert reward[1, 0] > 1.0
    assert reward[1, 2] == pytest.approx(0.940)


def test_lexicographic_rank_advantages_need_no_resource_weights():
    phase = np.array([[0.999, 0.997, 0.996, 0.990, 0.980]])
    active_fraction = np.array([[1.0, 0.5, 0.5, 0.1, 0.9]])
    coupling_cost = np.array([[0.1, 0.8, 0.2, 0.0, 0.0]])

    advantages, eligible, ranks = (
        variable_m.lexicographic_resource_rank_advantages(
            phase,
            active_fraction,
            coupling_cost,
            phase_reference=np.array([0.999]),
            phase_retention_tolerance=0.005,
        )
    )

    np.testing.assert_array_equal(
        eligible, np.array([[True, True, True, False, False]])
    )
    # Ineligible candidates come first and are ordered only by phase. Among
    # eligible candidates, fewer links wins before lower coupling cost.
    np.testing.assert_array_equal(ranks, np.array([[2.0, 3.0, 4.0, 1.0, 0.0]]))
    np.testing.assert_allclose(
        advantages, np.array([[0.0, 0.5, 1.0, -0.5, -1.0]])
    )


def test_lexicographic_rank_accepts_one_inference_candidate():
    advantages, eligible, ranks = (
        variable_m.lexicographic_resource_rank_advantages(
            np.array([[0.95]]),
            np.array([[0.5]]),
            np.array([[0.25]]),
            phase_reference=np.array([0.95]),
            phase_retention_tolerance=0.005,
        )
    )

    np.testing.assert_array_equal(eligible, np.array([[True]]))
    np.testing.assert_array_equal(ranks, np.array([[0.0]]))
    np.testing.assert_array_equal(advantages, np.array([[0.0]]))


def test_coupling_cost_tiebreak_requires_learned_sparsity():
    with pytest.raises(ValueError, match="learned_connected_sparsity"):
        small_config(coupling_cost_tiebreak_fraction=0.25)


def test_learned_connected_training_simulates_sparse_masks_and_updates_gate(
    monkeypatch,
):
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3,),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        initial_gate_logit=0.0,
        enable_learned_connected_sparsity=True,
        targets_per_batch=2,
        candidates_per_target=16,
        sparsity_phase_reward_threshold=0.98,
        successful_sparsity_reward_weight=0.05,
        n_jobs=1,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    observed_active_counts = []

    def fake_simulation(_targets, kappa, _phi, _config, **_kwargs):
        active_counts = np.count_nonzero(kappa, axis=(2, 3))
        observed_active_counts.extend(active_counts.reshape(-1).tolist())
        # All candidates satisfy the phase target, so differences in the
        # REINFORCE reward come from their learned sparse masks.
        return {"phase_reward": np.full(active_counts.shape, 0.99)}

    monkeypatch.setattr(
        variable_m, "simulate_candidates_parallel", fake_simulation
    )
    metrics = variable_m.run_training_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(11),
        iteration=0,
    )

    assert min(observed_active_counts) >= 2
    assert max(observed_active_counts) <= 6
    assert any(count < 6 for count in observed_active_counts)
    assert metrics["active_connection_fraction"] < 1.0
    assert metrics["sparsity_bonus"] > 0.0
    assert policy.edge_gate_head.weight.grad is not None
    assert torch.linalg.vector_norm(policy.edge_gate_head.weight.grad) > 0


def test_deterministic_connected_gates_use_exact_requested_budget():
    logits = np.array(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    gates = variable_m.deterministic_connected_gates(
        logits, np.array([2, 5]), n_lasers=3
    )

    np.testing.assert_array_equal(gates.sum(axis=1), np.array([2, 5]))
    receivers, sources = variable_m.directed_link_indices(3)
    for gate in gates:
        adjacency = np.zeros((3, 3), dtype=bool)
        active = np.flatnonzero(gate)
        adjacency[receivers[active], sources[active]] = True
        adjacency |= adjacency.T
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency.astype(int)
        assert np.linalg.matrix_rank(laplacian) == 2


def test_edge_budget_curriculum_expands_from_dense_to_spanning_tree():
    config = small_config(
        encoder_architecture="gnn",
        enable_sparse_gates=True,
        enable_connected_edge_budgets=True,
        condition_on_edge_budget=True,
        edge_budget_warmup_iterations=10,
        edge_budget_curriculum_iterations=20,
    )

    assert variable_m.minimum_active_links_at(0, 5, config) == 20
    assert variable_m.minimum_active_links_at(9, 5, config) == 20
    assert variable_m.minimum_active_links_at(29, 5, config) == 4


def test_sparsest_design_search_stops_at_first_passing_budget(monkeypatch):
    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        enable_sparse_gates=True,
        enable_connected_edge_budgets=True,
        condition_on_edge_budget=True,
    )
    tested_budgets = []

    def fake_design(_policy, _target, _config, *, active_link_count, **_kwargs):
        tested_budgets.append(active_link_count)
        return [{"phase_reward": 0.99 if active_link_count >= 4 else 0.95}]

    monkeypatch.setattr(variable_m, "design_for_target", fake_design)
    result = variable_m.design_sparsest_for_target(
        object(),
        np.zeros(3),
        config,
        minimum_phase_reward=0.98,
    )

    assert tested_budgets == [2, 3, 4]
    assert result["active_link_count"] == 4
    assert result["phase_target_satisfied"] is True


def test_sparse_sampling_and_gate_decoding():
    config = small_config(
        encoder_architecture="bigru",
        n_lasers=3,
        training_n_lasers=(2, 3, 4, 5),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)
    node_features = torch.randn(2, 3, 3)
    actions, gates, log_probabilities = (
        variable_m.sample_sparse_coupling_designs(
            policy, node_features, candidates_per_target=4
        )
    )

    assert actions.shape == (2, 4, 12)
    assert gates.shape == (2, 4, 6)
    assert log_probabilities.shape == (2, 4)
    (-log_probabilities.mean()).backward()
    assert policy.edge_gate_head.bias.grad is not None

    flat_actions = actions.detach().numpy().reshape(-1, 12)
    flat_gates = np.ones((len(flat_actions), 6))
    flat_gates[:, 0] = 0.0
    kappa, _ = variable_m.decode_action(
        flat_actions,
        config.maximum_kappa_per_ns,
        3,
        magnitude_gates=flat_gates,
    )
    receivers, sources = variable_m.directed_link_indices(3)
    assert np.all(kappa[:, receivers[0], sources[0]] == 0.0)


def test_sparse_bigru_checkpoint_round_trip(tmp_path):
    config = small_config(
        encoder_architecture="bigru",
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "bigru_sparse.pt"
    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy, _, loaded_config, _ = variable_m.load_checkpoint(path)

    assert saved["format"] == variable_m.BIGRU_SPARSE_CHECKPOINT_FORMAT
    assert loaded_config.enable_sparse_gates is True
    assert loaded_policy.edge_gate_head.out_features == 1


def test_sparse_gnn_checkpoint_round_trip(tmp_path):
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(3, 4, 5),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "gnn_sparse.pt"
    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    loaded_policy, _, loaded_config, _ = variable_m.load_checkpoint(path)

    assert isinstance(loaded_policy, variable_m.GNNPolicyNetwork)
    assert loaded_config.enable_sparse_gates is True
    assert loaded_policy.edge_gate_head.out_features == 1


def test_sparse_refinement_batch_combines_phase_and_resource_rewards(
    monkeypatch,
):
    from rl import refine_saved_coupling_model_variable_m_minimal_coupling as sparse

    config = small_config(
        encoder_architecture="bigru",
        n_lasers=3,
        training_n_lasers=(2, 3),
        condition_on_n_lasers=True,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        n_jobs=1,
    )
    policy = variable_m.BiGRUPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)

    def fake_simulation(
        _targets,
        kappa,
        _phi_p,
        _config,
        **_kwargs,
    ):
        target_count, candidate_count = kappa.shape[:2]
        rewards = np.linspace(0.7, 0.95, candidate_count)[None, :]
        return {
            "phase_reward": np.repeat(rewards, target_count, axis=0)
        }

    monkeypatch.setattr(
        sparse, "simulate_candidates_parallel", fake_simulation
    )
    metrics = sparse.sparse_refinement_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(4),
        iteration=0,
        dense_total_coupling_reference=(
            config.n_lasers
            * (config.n_lasers - 1)
            * config.maximum_kappa_per_ns
        ),
        coupling_weight=0.05,
        sparsity_weight=0.05,
        sample_gates=True,
        simulation_pool=None,
    )

    assert metrics["training_reward"] < metrics["phase_reward"]
    assert 0.0 <= metrics["coupling_budget"] <= 1.0
    assert 0.0 <= metrics["active_fraction"] <= 1.0
    assert np.isfinite(metrics["loss"])

    topology_metrics = sparse.sparse_refinement_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(5),
        iteration=20,
        dense_total_coupling_reference=(
            config.n_lasers
            * (config.n_lasers - 1)
            * config.maximum_kappa_per_ns
        ),
        coupling_weight=0.0,
        sparsity_weight=0.2,
        sample_gates=True,
        simulation_pool=None,
        use_phase_constrained_topology_reward=True,
        phase_reward_floor=0.8,
    )

    assert 0.0 <= topology_metrics["phase_eligible_fraction"] <= 1.0


def test_phase_relative_topology_reward_keeps_phase_primary():
    from rl import refine_saved_coupling_model_variable_m_minimal_coupling as sparse

    phase_reward = np.array([[1.0, 0.998, 0.990]])
    active_fraction = np.array([[1.0, 0.5, 0.0]])
    reward, eligible, required = sparse.phase_relative_topology_rewards(
        phase_reward,
        active_fraction,
        sparsity_weight=0.002,
        phase_reward_floor=0.98,
        phase_retention_tolerance=0.005,
    )

    np.testing.assert_allclose(required, [[0.995]])
    np.testing.assert_array_equal(eligible, [[True, True, False]])
    np.testing.assert_allclose(reward, [[1.0, 0.999, 0.990]])


def test_sparse_gnn_multisize_pcgrad_evaluates_disabled_links(monkeypatch):
    from rl import refine_saved_coupling_model_variable_m_minimal_coupling as sparse

    config = small_config(
        encoder_architecture="gnn",
        n_lasers=3,
        training_n_lasers=(3, 4),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        condition_on_bounded_degree=True,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        train_all_sizes_each_iteration=True,
        gradient_combination_mode="pcgrad",
        targets_per_batch=2,
        candidates_per_target=8,
        candidates_per_worker_task=4,
        n_jobs=1,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    observed_disabled_links = []

    def fake_simulation(indexed_work_item):
        (
            size_batch_index,
            candidate_start_index,
            candidate_stop_index,
            work_item,
        ) = indexed_work_item
        (
            chunk_index,
            start_index,
            stop_index,
            _targets,
            kappa,
            _phi_p,
            _detunings,
            _size_config,
        ) = work_item
        n_lasers = kappa.shape[-1]
        off_diagonal = ~np.eye(n_lasers, dtype=bool)
        disabled = np.sum(kappa[..., off_diagonal] == 0.0, axis=-1)
        observed_disabled_links.extend(disabled.reshape(-1).tolist())
        phase_reward = 0.99 - 0.001 * disabled
        return {
            "size_batch_index": size_batch_index,
            "chunk_index": chunk_index,
            "start_index": start_index,
            "stop_index": stop_index,
            "candidate_start_index": candidate_start_index,
            "candidate_stop_index": candidate_stop_index,
            "phase_reward": phase_reward,
        }

    monkeypatch.setattr(
        sparse, "simulate_indexed_target_chunk", fake_simulation
    )
    metrics = sparse.multisize_pcgrad_refinement_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(7),
        iteration=100,
        dense_coupling_reference_by_m={3: 1.0, 4: 1.0},
        coupling_weight=0.0,
        sparsity_weight=0.1,
        sample_gates=True,
        simulation_pool=None,
        use_phase_constrained_topology_reward=True,
    )

    assert 0 in observed_disabled_links
    assert max(observed_disabled_links) >= 1
    assert metrics["active_fraction"] < 1.0
    assert set(metrics["size_metrics"]) == {3, 4}
    assert np.isfinite(metrics["loss"])


@pytest.mark.parametrize(
    ("architecture", "gradient_mode"),
    [("bigru", "mean"), ("gnn", "mean"), ("gnn", "pcgrad")],
)
def test_multisize_update_uses_every_configured_size(
    monkeypatch, architecture, gradient_mode
):
    config = small_config(
        encoder_architecture=architecture,
        training_n_lasers=(2, 3, 4),
        condition_on_n_lasers=architecture == "bigru",
        condition_log_std_on_n_lasers=False,
        gradient_combination_mode=gradient_mode,
        targets_per_batch=3,
        candidates_per_target=4,
        candidates_per_worker_task=2,
        n_jobs=1,
    )
    policy = variable_m.make_policy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    observed_work_items = []

    def fake_simulation(indexed_work_item):
        observed_work_items.append(indexed_work_item)
        (
            size_batch_index,
            candidate_start_index,
            candidate_stop_index,
            work_item,
        ) = indexed_work_item
        (
            chunk_index,
            start_index,
            stop_index,
            targets,
            kappa,
            _phi_p,
            _detunings,
            _size_config,
        ) = work_item
        target_count, candidate_count = kappa.shape[:2]
        rewards = np.linspace(0.2, 0.9, candidate_count)[None, :]
        return {
            "size_batch_index": size_batch_index,
            "chunk_index": chunk_index,
            "start_index": start_index,
            "stop_index": stop_index,
            "candidate_start_index": candidate_start_index,
            "candidate_stop_index": candidate_stop_index,
            "phase_reward": np.repeat(rewards, target_count, axis=0),
        }

    monkeypatch.setattr(
        variable_m, "simulate_indexed_target_chunk", fake_simulation
    )
    metrics = variable_m.run_multisize_training_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(3),
        iteration=0,
    )

    assert set(metrics["size_metrics"]) == {2, 3, 4}
    assert len(observed_work_items) == 6
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["gradient_norm"])
    if gradient_mode == "pcgrad":
        assert 0.0 <= metrics["negative_gradient_pair_fraction"] <= 1.0
        assert -1.0 <= metrics["mean_gradient_cosine"] <= 1.0


def test_connected_budget_multisize_pcgrad_uses_spanning_tree_masks(
    monkeypatch,
):
    config = small_config(
        encoder_architecture="gnn",
        training_n_lasers=(3, 4),
        condition_on_n_lasers=False,
        condition_log_std_on_n_lasers=False,
        edge_conditioned_log_std=True,
        enable_sparse_gates=True,
        enable_connected_edge_budgets=True,
        condition_on_edge_budget=True,
        gradient_combination_mode="pcgrad",
        targets_per_batch=2,
        candidates_per_target=4,
        candidates_per_worker_task=2,
        n_jobs=1,
    )
    policy = variable_m.GNNPolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    observed_active_counts = []

    monkeypatch.setattr(
        variable_m,
        "sample_active_link_counts",
        lambda count, n_lasers, *_args: np.full(
            count, n_lasers - 1, dtype=np.int64
        ),
    )

    def fake_simulation(indexed_work_item):
        (
            size_batch_index,
            candidate_start_index,
            candidate_stop_index,
            work_item,
        ) = indexed_work_item
        (
            chunk_index,
            start_index,
            stop_index,
            _targets,
            kappa,
            _phi_p,
            _detunings,
            _size_config,
        ) = work_item
        active_counts = np.count_nonzero(kappa, axis=(2, 3))
        observed_active_counts.extend(active_counts.reshape(-1).tolist())
        target_count, candidate_count = kappa.shape[:2]
        rewards = np.linspace(0.4, 0.9, candidate_count)[None, :]
        return {
            "size_batch_index": size_batch_index,
            "chunk_index": chunk_index,
            "start_index": start_index,
            "stop_index": stop_index,
            "candidate_start_index": candidate_start_index,
            "candidate_stop_index": candidate_stop_index,
            "phase_reward": np.repeat(rewards, target_count, axis=0),
        }

    monkeypatch.setattr(
        variable_m, "simulate_indexed_target_chunk", fake_simulation
    )
    metrics = variable_m.run_multisize_training_batch(
        policy,
        optimizer,
        config,
        np.random.default_rng(9),
        iteration=50,
    )

    assert set(observed_active_counts) == {2, 3}
    assert metrics["active_connection_fraction"] < 1.0
    assert np.isfinite(metrics["loss"])
    assert policy.edge_gate_head.weight.grad is not None


def test_pcgrad_clips_each_task_and_removes_direct_opposition():
    task_gradients = torch.tensor(
        [[2.0, 0.0], [-3.0, 0.0]], dtype=torch.float32
    )
    combined, diagnostics = variable_m.combine_task_gradients_pcgrad(
        task_gradients,
        maximum_task_norm=1.0,
        rng=np.random.default_rng(4),
    )

    torch.testing.assert_close(
        diagnostics["raw_task_norms"], torch.tensor([2.0, 3.0])
    )
    torch.testing.assert_close(
        diagnostics["clipped_task_norms"], torch.ones(2)
    )
    assert diagnostics["negative_gradient_pair_fraction"] == 1.0
    assert diagnostics["mean_gradient_cosine"] == pytest.approx(-1.0)
    torch.testing.assert_close(combined, torch.zeros(2), atol=1.0e-7, rtol=0.0)


def test_dynamic_symmetry_action_width_and_decode():
    config = small_config(
        force_symmetric_kappa=True,
        force_symmetric_phi_p=False,
    )
    policy = variable_m.PolicyNetwork(config)
    n_lasers = 7
    unique_links = n_lasers * (n_lasers - 1) // 2
    directed_links = n_lasers * (n_lasers - 1)
    actions = policy(torch.zeros(1, n_lasers, 3)).detach().numpy()

    assert actions.shape == (1, unique_links + directed_links)
    kappa, phi_p = variable_m.decode_action(
        actions,
        config.maximum_kappa_per_ns,
        n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
    )
    np.testing.assert_allclose(kappa, kappa.transpose(0, 2, 1))
    assert np.allclose(np.diagonal(kappa[0]), 0.0)
    assert np.allclose(np.diagonal(phi_p[0]), 0.0)


def test_degree_scaled_decode_bounds_incoming_coupling_independent_of_m():
    maximum_kappa_per_ns = 100.0
    for n_lasers in (2, 3, 7):
        action_size = 2 * n_lasers * (n_lasers - 1)
        actions = np.zeros((1, action_size))
        kappa, _ = variable_m.decode_action(
            actions,
            maximum_kappa_per_ns,
            n_lasers,
            normalize_incoming_coupling_by_degree=True,
        )

        # A zero magnitude logit gives half of each link's allowed maximum.
        # Summing all M-1 incoming links therefore gives the same value for
        # every receiver and every array size.
        np.testing.assert_allclose(
            kappa.sum(axis=2),
            np.full((1, n_lasers), 0.5 * maximum_kappa_per_ns),
        )
        assert variable_m.effective_maximum_kappa_per_link(
            maximum_kappa_per_ns, n_lasers, True
        ) == pytest.approx(maximum_kappa_per_ns / (n_lasers - 1))

    # Since M=2 has one possible incoming edge, scaling leaves it unchanged.
    actions = np.zeros((1, 4))
    scaled, _ = variable_m.decode_action(
        actions, maximum_kappa_per_ns, 2,
        normalize_incoming_coupling_by_degree=True,
    )
    unscaled, _ = variable_m.decode_action(
        actions, maximum_kappa_per_ns, 2,
        normalize_incoming_coupling_by_degree=False,
    )
    np.testing.assert_allclose(scaled, unscaled)


@pytest.mark.parametrize("n_lasers", [2, 5, 9])
def test_stratified_detuning_sampler_covers_same_span_range(n_lasers):
    count = 28
    maximum_span_ghz = 5.0
    config = small_config(
        n_lasers=n_lasers,
        detuning_span_ghz=maximum_span_ghz,
        detuning_curriculum_initial_half_span_ghz=0.1,
        detuning_curriculum_warmup_iterations=200,
        detuning_curriculum_iterations=1000,
        detuning_span_sampling_mode="stratified_span",
    )
    detunings = variable_m.sample_detuning_distributions(
        count,
        np.random.default_rng(23),
        config,
        iteration=config.detuning_curriculum_iterations,
    )

    assert detunings.shape == (count, n_lasers)
    np.testing.assert_allclose(detunings.mean(axis=1), 0.0, atol=1.0e-14)
    assert np.all(np.diff(detunings, axis=1) >= 0.0)

    # Sorting the randomized batch widths must put exactly one width inside
    # each equal-width stratum from zero to the current curriculum maximum.
    sampled_spans = np.sort(np.ptp(detunings, axis=1))
    stratum_lower = maximum_span_ghz * np.arange(count) / count
    stratum_upper = maximum_span_ghz * np.arange(1, count + 1) / count
    assert np.all(sampled_spans >= stratum_lower)
    assert np.all(sampled_spans < stratum_upper)

    if n_lasers == 2:
        row_spans = np.ptp(detunings, axis=1)
        np.testing.assert_allclose(detunings[:, 0], -0.5 * row_spans)
        np.testing.assert_allclose(detunings[:, 1], 0.5 * row_spans)


def test_stratified_detuning_sampler_respects_curriculum_maximum():
    config = small_config(
        n_lasers=4,
        detuning_span_ghz=5.0,
        detuning_curriculum_initial_half_span_ghz=0.1,
        detuning_curriculum_warmup_iterations=200,
        detuning_curriculum_iterations=1000,
        detuning_span_sampling_mode="stratified_span",
    )
    warmup = variable_m.sample_detuning_distributions(
        28, np.random.default_rng(8), config, iteration=100
    )
    complete = variable_m.sample_detuning_distributions(
        28, np.random.default_rng(8), config, iteration=1000
    )

    assert np.max(np.ptp(warmup, axis=1)) < 0.2
    assert np.max(np.ptp(complete, axis=1)) < 5.0
    assert np.max(np.ptp(complete, axis=1)) > 4.8


def test_detuning_curriculum_half_span_matches_current_validation_limit():
    config = small_config(
        detuning_span_ghz=5.0,
        detuning_curriculum_initial_half_span_ghz=0.05,
        detuning_curriculum_warmup_iterations=200,
        detuning_curriculum_iterations=7300,
    )

    assert variable_m.detuning_curriculum_half_span_at(
        0, config
    ) == pytest.approx(0.05)
    assert variable_m.detuning_curriculum_half_span_at(
        200, config
    ) == pytest.approx(0.05)
    assert 2.0 * variable_m.detuning_curriculum_half_span_at(
        1500, config
    ) == pytest.approx(0.9971830986)
    assert variable_m.detuning_curriculum_half_span_at(
        7300, config
    ) == pytest.approx(2.5)
    assert variable_m.detuning_curriculum_half_span_at(
        None, config
    ) == pytest.approx(2.5)


def test_architecture_sanity_check_covers_requested_sizes():
    shapes = variable_m.run_architecture_sanity_checks(small_config())
    assert shapes == {
        2: (2, 4),
        3: (2, 12),
        5: (2, 40),
        7: (2, 84),
        10: (2, 180),
    }


def test_variable_m_checkpoint_round_trip_and_fixed_checkpoint_rejection(
    tmp_path,
):
    config = small_config(
        best_checkpoint_file=str(tmp_path / "best.pt"),
        current_checkpoint_file=str(tmp_path / "current.pt"),
        final_checkpoint_file=str(tmp_path / "final.pt"),
    )
    policy = variable_m.PolicyNetwork(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "variable.pt"
    variable_m.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=4,
        held_out_reward=0.7,
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy, _, loaded_config, metadata = variable_m.load_checkpoint(path)

    assert saved["variable_m_policy"] is True
    assert saved["training_n_lasers"] == (3, 4, 5, 6, 7)
    assert saved["node_embedding_dim"] == 8
    assert saved["global_embedding_dim"] == 16
    assert saved["gru_hidden_size"] == 6
    assert loaded_config.training_n_lasers == (3, 4, 5, 6, 7)
    assert loaded_policy(torch.zeros(1, 10, 3)).shape == (1, 180)
    assert metadata == {"iteration": 4, "held_out_reward": 0.7}

    fixed_path = tmp_path / "fixed.pt"
    torch.save(
        {"format": "conditional_coupling_edge_v3_local_detuning"},
        fixed_path,
    )
    with pytest.raises(ValueError, match="not a variable-M pooled"):
        variable_m.load_checkpoint(fixed_path)


def test_conditional_backbone_budgets_always_keep_a_spanning_tree():
    n_lasers = 6
    maximum_links = n_lasers * (n_lasers - 1)
    kappa = np.arange(
        1, 1 + 4 * n_lasers * n_lasers, dtype=float
    ).reshape(4, n_lasers, n_lasers)
    for matrix in kappa:
        np.fill_diagonal(matrix, 0.0)
    phi = np.zeros_like(kappa)
    requested_counts = np.array(
        [n_lasers - 1, n_lasers, maximum_links // 2, maximum_links]
    )
    pruned, _, realized = variable_m.prune_coupling_batch_to_link_budget(
        kappa, phi, requested_counts
    )

    for matrix, requested_count, density in zip(
        pruned, requested_counts, realized
    ):
        assert np.count_nonzero(matrix) == requested_count
        assert density == pytest.approx(requested_count / maximum_links)
        adjacency = (matrix + matrix.T) > 0.0
        reached = {0}
        while True:
            expanded = reached | {
                neighbor
                for node in reached
                for neighbor in np.flatnonzero(adjacency[node])
            }
            if expanded == reached:
                break
            reached = expanded
        assert reached == set(range(n_lasers))
