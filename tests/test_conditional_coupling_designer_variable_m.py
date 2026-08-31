from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from rl import conditional_coupling_designer_variable_m as variable_m


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
