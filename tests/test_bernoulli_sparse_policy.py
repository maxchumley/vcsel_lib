"""Focused tests for the separate direct Bernoulli-edge experiment."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt

from rl import conditional_coupling_bernoulli_sparse as bernoulli
from rl import conditional_coupling_designer_variable_m as designer
from rl.conditional_coupling_run_variable_m import phase_reward_trace
from rl.conditional_coupling_bernoulli_sparse import (
    BernoulliSparseGNNPolicy, calibrate_budget_references,
    budget_exploration_at,
    calibrated_logit_bias, coupling_cost_weight_at, decode_continuous_actions,
    decode_samples, dense_reference_reward, encode_context,
    evaluate_policy, infer_designs,
    lexicographic_candidate_scores, lexicographic_phase_error_limit_at,
    lexicographic_phase_reward_limit_at,
    maximum_phase_error_deg, sample_budget_actions,
    redistribute_coupling, reference_budget_for_m, relative_coupling_budget, reward_function,
    root_reach_fractions, run_training_batch, sample_candidates,
    set_gate_head_trainable_at,
    save_checkpoint, load_checkpoint, simulate_rooted_indexed_chunk, training_figure,
)
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import (
    config as active_run_config, MODEL_SUFFIX as ACTIVE_MODEL_SUFFIX,
)


# Preserve regression coverage for the completed cosine/curriculum family.
# The active run configuration has separate coverage in test_stratified_rms_run.py.
previous_topology_config = replace(
    active_run_config, training_iterations=5000, targets_per_batch=14,
    targets_per_worker_task=1, candidates_per_target=72,
    candidates_per_worker_task=72, actions_per_topology=8,
    topology_sampling_mode="per_target", phase_reward_mode="cosine",
    pcgrad_task_grouping="array_size_topology_density", pcgrad_topology_groups=3,
    topology_curriculum_start_iteration=0, topology_curriculum_end_iteration=500,
    detuning_curriculum_iterations=500,
    phase_exploration_fraction=.25, phase_exploration_start_std=np.pi,
    phase_exploration_start_iteration=500, phase_exploration_end_iteration=4000,
    minimum_log_std_anneal_start_iteration=1500, minimum_log_std_anneal_iterations=1000,
    exploration_anneal_start_iteration=1500, exploration_anneal_iterations=1000,
    learning_rate_decay_start_iteration=3500,
    current_checkpoint_file=active_run_config.current_checkpoint_file.replace(
        ACTIVE_MODEL_SUFFIX, "_star_wide_phase_topology_pcgrad_no_de"),
)

run_config = replace(previous_topology_config, topology_input_mode=False,
                     pcgrad_task_grouping="array_size",
                     actions_per_topology=1, objective_mode="lexicographic",
                     topology_star_backbone=False,
                     topology_curriculum_start_iteration=None,
                     topology_curriculum_end_iteration=None,
                     construct_rooted_backbone=True,
                     uniform_random_rooted_tree=True,
                     enable_sparse_gates=True,
                     budget_action_transform="scaled_softplus",
                     strength_exploration_fraction=0.0,
                     phase_exploration_fraction=0.0,
                     de_teacher_enabled=False,
                     gate_head_hold_iterations=previous_topology_config.training_iterations)

legacy_config = replace(run_config, construct_rooted_backbone=False,
                        uniform_random_rooted_tree=False,
                        require_rooted_connectivity=True,
                        objective_mode="weighted",
                        sparsity_reward_mode="total_sparsity_fraction",
                        link_cost_weight=0.25, coupling_cost_weight=0.15,
                        dense_reference_feasibility=False,
                        strength_action_transform="bounded_sigmoid",
                        uncapped_redistribution=False)


@pytest.fixture(scope="module")
def policy_and_config():
    config = replace(legacy_config, n_lasers=3)
    return BernoulliSparseGNNPolicy(config), config


def test_no_q_or_tolerance_head(policy_and_config):
    policy, _ = policy_and_config
    assert not hasattr(policy, "density_head")
    assert not hasattr(policy, "q_log_std")
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    assert features.shape == (1, 3, 4)
    mean, log_std, logits = policy.parameters_for_edges(features)
    assert mean.shape == log_std.shape == (1, 12)
    assert logits.shape == (1, 6)


def test_gate_head_hold_keeps_zero_logits_then_releases():
    config = replace(run_config, n_lasers=3, gate_head_hold_iterations=1000)
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)

    set_gate_head_trainable_at(policy, config.gate_head_hold_iterations - 1, config)
    assert not any(parameter.requires_grad for parameter in policy.edge_gate_head.parameters())
    _, _, log_prob, logits = sample_candidates(
        policy, features, 2, offsets=torch.tensor([[-4.0, 4.0]])
    )
    assert torch.count_nonzero(logits) == 0
    assert torch.sigmoid(logits - 4).max() < 0.02
    assert torch.sigmoid(logits + 4).min() > 0.98
    (-log_prob.mean()).backward()
    optimizer.step()
    assert torch.count_nonzero(policy.edge_gate_head.weight) == 0
    assert torch.count_nonzero(policy.edge_gate_head.bias) == 0
    assert all(parameter.grad is None for parameter in policy.edge_gate_head.parameters())

    optimizer.zero_grad(set_to_none=True)
    set_gate_head_trainable_at(policy, config.gate_head_hold_iterations, config)
    assert all(parameter.requires_grad for parameter in policy.edge_gate_head.parameters())
    released_features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    released_logits = policy.parameters_for_edges(released_features)[2]
    assert torch.count_nonzero(released_logits) == 0
    released_logits.sum().backward()
    assert policy.edge_gate_head.weight.grad is not None


def test_sampled_gates_are_binary_and_disable_exactly(policy_and_config):
    policy, config = policy_and_config
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    torch.manual_seed(11)
    actions, gates, log_prob, _ = sample_candidates(policy, features, 20)
    assert set(torch.unique(gates).tolist()) <= {0.0, 1.0}
    assert log_prob.shape == (1, 20)
    kappa, _, counts, rho, shortfall = decode_samples(actions, gates, config)
    assert np.array_equal(counts, gates.sum(-1).numpy().astype(int))
    assert np.allclose(rho, counts / 6)
    assert np.array_equal((kappa > 0).sum(axis=(-2, -1)), counts)
    assert np.all(kappa[kappa == 0] == 0)
    assert np.all(shortfall >= 0)


def test_redistribution_preserves_raw_total_and_respects_caps():
    raw = np.array([[1.0, 2.0, 3.0, 4.0]])
    used, shortfall = redistribute_coupling(
        raw, np.array([[True, True, False, False]]), cap=6.0,
    )
    np.testing.assert_allclose(used, [[4.0, 6.0, 0.0, 0.0]], atol=1e-10)
    np.testing.assert_allclose(used.sum(axis=1), raw.sum(axis=1), atol=1e-10)
    np.testing.assert_allclose(shortfall, 0, atol=1e-10)


def test_redistribution_reports_capacity_shortfall_and_all_off():
    raw = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    used, shortfall = redistribute_coupling(
        raw, np.array([[True, False, False], [False, False, False]]), cap=5.0,
    )
    np.testing.assert_array_equal(used, [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(shortfall, [4.0, 9.0])


def test_uncapped_redistribution_preserves_total_with_one_link():
    raw = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    used, shortfall = redistribute_coupling(
        raw, np.array([[True, False, False], [False, False, False]]), cap=None,
    )
    np.testing.assert_allclose(used, [[9.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(shortfall, [0.0, 9.0])


def test_uncapped_redistribution_handles_zero_raw_weight_on_active_link():
    used, shortfall = redistribute_coupling(
        np.array([[0.0, 2.0, 3.0]]),
        np.array([[True, False, False]]), cap=None,
    )
    np.testing.assert_allclose(used, [[5.0, 0.0, 0.0]])
    np.testing.assert_allclose(shortfall, 0.0)


def test_decode_samples_redistributes_before_simulation(policy_and_config):
    _, config = policy_and_config
    actions = torch.zeros((1, 1, 12))
    gates = torch.tensor([[[1., 1., 1., 1., 0., 0.]]])
    kappa, _, counts, rho, shortfall = decode_samples(actions, gates, config)
    cap = config.maximum_kappa_per_ns / (2 if config.normalize_incoming_coupling_by_degree else 1)
    assert counts[0, 0] == 4
    assert rho[0, 0] == pytest.approx(4 / 6)
    assert kappa[0, 0].sum() == pytest.approx(3 * cap)
    assert shortfall[0, 0] == pytest.approx(0, abs=1e-10)
    assert np.count_nonzero(kappa[0, 0]) == 4


def test_decode_samples_uncapped_preserves_total_beyond_raw_link_scale():
    config = replace(run_config, n_lasers=3)
    actions = torch.zeros((1, 1, 12))
    actions[:, :, :6] = 100.0
    gates = torch.tensor([[[1., 0., 0., 0., 0., 0.]]])
    kappa, _, counts, rho, shortfall = decode_samples(actions, gates, config)
    raw_link_strength = config.softplus_strength_scale_per_ns * np.logaddexp(0.0, 100.0) / 2
    assert counts[0, 0] == 1
    assert rho[0, 0] == pytest.approx(1 / 6)
    assert kappa[0, 0].sum() == pytest.approx(6 * raw_link_strength)
    assert kappa[0, 0].max() > config.maximum_kappa_per_ns / 2
    assert shortfall[0, 0] == pytest.approx(0.0)


def test_softplus_strengths_and_policy_mean_have_no_old_saturation():
    config = replace(run_config, n_lasers=3)
    actions = np.zeros((1, 12))
    actions[0, 0] = 1000.0
    kappa, _ = decode_continuous_actions(actions, config)
    assert kappa.max() == pytest.approx(1000 * config.softplus_strength_scale_per_ns / 2)
    kappa_with_different_legacy_max, _ = decode_continuous_actions(
        actions, replace(config, maximum_kappa_per_ns=1.0),
    )
    np.testing.assert_array_equal(kappa_with_different_legacy_max, kappa)
    policy = BernoulliSparseGNNPolicy(config)
    with torch.no_grad():
        policy.edge_output.weight.zero_()
        policy.edge_output.bias[0] = 20.0
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    mean, _, _ = policy.parameters_for_edges(features)
    assert torch.all(mean[0, :6] == 20.0)


@pytest.mark.parametrize("fraction", [0.05, 0.2, 0.8, 0.95])
def test_requested_fraction_is_expected_probability_not_exact_count(
    policy_and_config, fraction,
):
    policy, config = policy_and_config
    design = infer_designs(
        policy, np.zeros(3), np.array([-0.05, 0.0, 0.05]), config,
        count=1000, active_fraction=fraction, seed=3,
    )
    assert design["expected_active_fraction"] == pytest.approx(fraction, abs=1e-6)
    assert design["retained_link_fractions"].mean() == pytest.approx(fraction, abs=0.04)
    assert np.unique(design["active_link_counts"]).size > 1


def test_calibration_rejects_endpoints():
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError):
            calibrated_logit_bias(torch.zeros(1, 6), fraction)


def test_smooth_reward_has_no_error_threshold(policy_and_config):
    _, config = policy_and_config
    phase = np.array([0.2, 0.99])
    rho = np.array([0.5, 0.5])
    budget = np.array([0.1, 0.1])
    score = reward_function(phase, rho, budget, config,
                            root_reach_fraction=np.ones_like(phase))
    assert score[1] > score[0]
    assert score[0] == pytest.approx(phase[0] /
                                    (1 + config.link_cost_weight * rho[0] +
                                     config.coupling_cost_weight * budget[0]))


def test_phase_reward_stays_monotonic_at_dense_cost(policy_and_config):
    _, config = policy_and_config
    phase = np.linspace(0, 1, 101)
    score = reward_function(phase, np.ones_like(phase), np.ones_like(phase), config,
                            root_reach_fraction=np.ones_like(phase))
    assert np.all(np.diff(score) > 0)
    assert score[-1] == pytest.approx(1 / 1.4)


def test_dense_reference_feasibility_is_strict_and_phase_first():
    phase = np.array([[0.899, 0.9, 0.91], [0.4, 0.7, 0.8]])
    rho = np.array([[0.1, 0.8, 0.9], [0.1, 0.2, 0.9]])
    score, feasible = dense_reference_reward(phase, np.array([[0.9], [0.95]]), rho)
    np.testing.assert_array_equal(feasible, [[False, True, True], [False, False, False]])
    assert score[0, 2] > score[0, 1] > score[0, 0]
    assert score[1, 2] > score[1, 1] > score[1, 0]
    assert score[1, 2] == pytest.approx(phase[1, 2])
    low_score, low_feasible = dense_reference_reward(
        np.array([[0.15]]), np.array([[0.1]]), np.array([[0.2]]),
    )
    assert low_feasible[0, 0]
    assert low_score[0, 0] < 0.152


@pytest.mark.parametrize("n_lasers", [3, 9])
def test_links_per_laser_charges_only_extra_links(n_lasers):
    config = replace(run_config, n_lasers=n_lasers, link_cost_weight=0.05)
    possible = n_lasers * (n_lasers - 1)
    active = np.array([n_lasers - 1, 2 * n_lasers - 1])
    score = reward_function(np.ones(2), active / possible, np.zeros(2), config)
    np.testing.assert_allclose(score, [1.0, 1.0 / (1.0 + config.link_cost_weight)])


def test_invalid_sparsity_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="sparsity_reward_mode"):
        replace(run_config, sparsity_reward_mode="unknown")


def test_rooted_reach_respects_link_direction():
    receivers, sources = designer.directed_link_indices(3)
    def edges(*links):
        matrix = np.zeros((3, 3), dtype=int)
        for source, receiver in links:
            matrix[receiver, source] = 1
        return matrix[receivers, sources]

    masks = np.stack((
        edges(), edges((0, 1)), edges((0, 1), (1, 2)),
        edges((0, 1), (2, 1)), edges((0, 1), (0, 2)),
    ))
    np.testing.assert_allclose(root_reach_fractions(masks, 3),
                               [0.0, 0.5, 1.0, 0.5, 1.0])


@pytest.mark.parametrize("n_lasers", [3, 5, 9])
def test_constructed_tree_is_rooted_with_all_optional_gates_off(n_lasers):
    config = replace(run_config, n_lasers=n_lasers,
                     construct_rooted_backbone=True,
                     require_rooted_connectivity=False,
                     uniform_random_rooted_tree=False)
    policy = BernoulliSparseGNNPolicy(config)
    with torch.no_grad():
        policy.edge_gate_head.weight.zero_()
        policy.edge_gate_head.bias.fill_(-100.0)
    targets = np.zeros((2, n_lasers))
    detunings = np.tile(np.linspace(-0.05, 0.05, n_lasers), (2, 1))
    features = encode_context(targets, detunings, policy)
    torch.manual_seed(37)
    _, gates, log_prob, _ = sample_candidates(policy, features, 20)
    assert np.all(gates.sum(-1).numpy() == n_lasers - 1)
    assert np.all(root_reach_fractions(gates.numpy(), n_lasers) == 1.0)
    root_grad, edge_grad = torch.autograd.grad(
        log_prob.sum(), (policy.root_head.weight, policy.tree_edge_head.weight))
    assert torch.isfinite(root_grad).all() and torch.any(root_grad != 0)
    assert torch.isfinite(edge_grad).all() and torch.any(edge_grad != 0)


def test_constructed_inference_is_deterministic_and_rooted():
    config = replace(run_config, n_lasers=3, construct_rooted_backbone=True,
                     require_rooted_connectivity=False,
                     uniform_random_rooted_tree=False)
    policy = BernoulliSparseGNNPolicy(config)
    with torch.no_grad():
        policy.edge_gate_head.weight.zero_()
        policy.edge_gate_head.bias.fill_(-100.0)
    target, detunings = np.zeros(3), np.array([-0.05, 0.0, 0.05])
    first = infer_designs(policy, target, detunings, config, seed=1)
    second = infer_designs(policy, target, detunings, config, seed=2)
    np.testing.assert_array_equal(first["gates"], second["gates"])
    np.testing.assert_array_equal(first["kappa_per_ns"], second["kappa_per_ns"])
    np.testing.assert_array_equal(first["root_indices"], second["root_indices"])
    np.testing.assert_allclose(
        first["natural_gate_probabilities"],
        torch.sigmoid(torch.from_numpy(first["gate_logits"])).numpy(),
    )
    assert np.all(first["gate_probabilities"][first["tree_gates"][0]] == 1.0)
    assert first["active_link_counts"][0] == 2
    assert root_reach_fractions(first["gates"], 3)[0] == 1.0


def test_constructed_inference_can_sample_optional_gates_reproducibly():
    config = replace(run_config, n_lasers=3, construct_rooted_backbone=True,
                     require_rooted_connectivity=False,
                     uniform_random_rooted_tree=False)
    policy = BernoulliSparseGNNPolicy(config)
    target, detunings = np.zeros(3), np.array([-0.05, 0.0, 0.05])
    first = infer_designs(policy, target, detunings, config, count=128,
                          seed=23, sample_optional_gates=True)
    second = infer_designs(policy, target, detunings, config, count=128,
                           seed=23, sample_optional_gates=True)
    np.testing.assert_array_equal(first["gates"], second["gates"])
    np.testing.assert_array_equal(first["root_indices"], second["root_indices"])
    np.testing.assert_array_equal(first["tree_gates"], second["tree_gates"])
    np.testing.assert_allclose(first["optional_gate_probabilities"], 0.5)
    tree = first["tree_gates"].astype(bool)
    assert np.all(first["gates"][tree])
    optional = first["gates"][:, ~tree[0]]
    assert np.all(optional.any(axis=0))
    assert np.all((~optional.astype(bool)).any(axis=0))
    np.testing.assert_array_equal(root_reach_fractions(first["gates"], 3), 1.0)


def test_random_topology_ignores_root_and_edge_scores():
    first_root_scores = torch.tensor([[100.0, -100.0, -100.0]])
    second_root_scores = -first_root_scores
    first_edge_scores = torch.arange(6.0).reshape(1, 6) * 100.0
    second_edge_scores = -first_edge_scores
    torch.manual_seed(19)
    first_tree, first_log_prob, first_roots = bernoulli.select_rooted_tree(
        first_root_scores, first_edge_scores, 256, uniform_random=True)
    torch.manual_seed(19)
    second_tree, second_log_prob, second_roots = bernoulli.select_rooted_tree(
        second_root_scores, second_edge_scores, 256, uniform_random=True)
    torch.testing.assert_close(first_tree, second_tree)
    torch.testing.assert_close(first_roots, second_roots)
    assert torch.count_nonzero(first_log_prob) == 0
    assert torch.count_nonzero(second_log_prob) == 0
    assert torch.unique(first_roots).numel() == 3
    assert torch.unique(first_tree.reshape(256, 6), dim=0).shape[0] > 3
    np.testing.assert_array_equal(root_reach_fractions(first_tree.numpy(), 3), 1.0)


def test_random_topology_run_holds_gates_and_topology_heads():
    config = replace(run_config, n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    set_gate_head_trainable_at(policy, config.training_iterations - 1, config)
    assert not any(p.requires_grad for p in policy.edge_gate_head.parameters())
    assert not any(p.requires_grad for p in policy.root_head.parameters())
    assert not any(p.requires_grad for p in policy.tree_edge_head.parameters())
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    _, _, _, logits = sample_candidates(policy, features, 16)
    assert torch.count_nonzero(logits) == 0
    first = infer_designs(policy, np.zeros(3), np.array([-0.05, 0.0, 0.05]),
                          config, count=128, seed=23, sample_optional_gates=True)
    second = infer_designs(policy, np.zeros(3), np.array([-0.05, 0.0, 0.05]),
                           config, count=128, seed=23, sample_optional_gates=True)
    np.testing.assert_array_equal(first["tree_gates"], second["tree_gates"])
    assert np.unique(first["tree_gates"], axis=0).shape[0] > 3
    assert first["retained_link_fractions"].mean() < 0.9


def test_random_topology_heldout_uses_sparse_seeded_masks(monkeypatch):
    config = replace(run_config, n_lasers=3,
                     coupling_budget_reference_by_m=((3, 10.0),))
    policy = BernoulliSparseGNNPolicy(config)
    observed = []

    def fake_simulation(targets, kappa, _phi, _config, **_kwargs):
        observed.append((kappa > 0).sum(axis=(-2, -1)).copy())
        return dict(phase_reward=np.full((len(targets), 1), 0.9),
                    aligned_achieved_phases_rad=np.zeros((len(targets), 1, 3)))

    monkeypatch.setattr(bernoulli.base, "simulate_candidates_parallel", fake_simulation)
    targets = np.zeros((64, 3))
    detunings = np.tile([-0.05, 0.0, 0.05], (64, 1))
    first = evaluate_policy(policy, targets, detunings, config, seed=71)
    second = evaluate_policy(policy, targets, detunings, config, seed=71)
    np.testing.assert_array_equal(observed[0], observed[1])
    assert np.any(observed[0] < 6)
    assert np.all(observed[0] >= 2)
    assert first["retained_link_fraction"] == second["retained_link_fraction"]


@pytest.mark.parametrize("n_lasers", [3, 9])
def test_topology_input_sampler_stratifies_rooted_link_counts(n_lasers):
    masks = bernoulli.sample_stratified_rooted_topologies(
        3, 9, n_lasers, np.random.default_rng(31), torch.device("cpu"))
    n_edges = n_lasers * (n_lasers - 1)
    for row in masks.numpy():
        counts = row.sum(axis=-1).astype(int)
        assert counts.min() == n_lasers - 1 and counts.max() == n_edges
        if n_lasers == 3:
            levels, frequencies = np.unique(counts, return_counts=True)
            np.testing.assert_array_equal(levels, np.arange(2, 7))
            assert frequencies.min() >= 1 and frequencies.max() <= 2
        else:
            expected = np.rint(np.linspace(n_lasers - 1, n_edges, 9)).astype(int)
            np.testing.assert_array_equal(np.sort(counts), np.sort(expected))
    np.testing.assert_array_equal(root_reach_fractions(masks.numpy(), n_lasers), 1.0)


@pytest.mark.parametrize("n_lasers", [3, 9])
def test_training_topology_bias_favors_dense_masks_but_keeps_tree(n_lasers):
    device = torch.device("cpu")
    unbiased = bernoulli.sample_stratified_rooted_topologies(
        4, 9, n_lasers, np.random.default_rng(31), device)
    biased = bernoulli.sample_stratified_rooted_topologies(
        4, 9, n_lasers, np.random.default_rng(31), device, density_bias=3.0)
    counts = biased.numpy().sum(axis=-1)
    assert np.all(counts.min(axis=1) == n_lasers - 1)
    assert np.all(counts.max(axis=1) == n_lasers * (n_lasers - 1))
    assert counts.mean() > unbiased.numpy().sum(axis=-1).mean()
    np.testing.assert_array_equal(root_reach_fractions(biased.numpy(), n_lasers), 1.0)


@pytest.mark.parametrize("n_lasers", [3, 9])
def test_training_topology_density_floor_excludes_sparse_masks(n_lasers):
    n_edges = n_lasers * (n_lasers - 1)
    masks = bernoulli.sample_stratified_rooted_topologies(
        4, 9, n_lasers, np.random.default_rng(31), torch.device("cpu"),
        min_fraction=0.70)
    counts = masks.numpy().sum(axis=-1)
    assert np.all(counts.min(axis=1) == np.ceil(0.70 * n_edges))
    assert np.all(counts.max(axis=1) == n_edges)
    assert np.all(counts >= np.ceil(0.70 * n_edges))
    np.testing.assert_array_equal(root_reach_fractions(masks.numpy(), n_lasers), 1.0)


@pytest.mark.parametrize("n_lasers", [3, 9])
def test_fixed_star_topology_curriculum_reaches_full_range(n_lasers):
    config = replace(previous_topology_config, n_lasers=n_lasers)
    n_edges = n_lasers * (n_lasers - 1)
    receivers, sources = designer.directed_link_indices(n_lasers)
    star = sources == 0
    expected_floor = {
        0: n_edges,
        250: int(np.ceil(0.5 * (n_edges + n_lasers - 1))),
        499: n_lasers,
        500: n_lasers - 1,
        750: n_lasers - 1,
        4999: n_lasers - 1,
    }
    for iteration, minimum in expected_floor.items():
        fraction = bernoulli.topology_min_fraction_at(iteration, config)
        masks = bernoulli.sample_stratified_rooted_topologies(
            2, 9, n_lasers, np.random.default_rng(10), torch.device("cpu"),
            min_fraction=fraction, fixed_star=True)
        counts = masks.sum(dim=-1).numpy()
        assert np.all(counts.min(axis=1) == minimum)
        assert np.all(counts.max(axis=1) == n_edges)
        assert np.all(masks.numpy()[..., star] == 1)
        np.testing.assert_array_equal(root_reach_fractions(masks.numpy(), n_lasers), 1.0)


def test_topology_input_conditions_edge_and_budget_heads_and_enforces_mask():
    config = replace(previous_topology_config, n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    assert not hasattr(policy, "edge_gate_head")
    assert not hasattr(policy, "root_head")
    target = np.array([0.0, 0.4, 1.0])
    detunings = np.array([0.05, -0.05, 0.0])
    features = encode_context(target, detunings, policy)
    tree = np.zeros((3, 3))  # source row, receiver column
    tree[0, 1] = tree[1, 2] = 1.0
    dense = np.ones((3, 3)) - np.eye(3)
    receivers, sources = designer.directed_link_indices(3)
    tree_vector = torch.tensor(tree[sources, receivers][None], dtype=features.dtype)
    dense_vector = torch.tensor(dense[sources, receivers][None], dtype=features.dtype)
    tree_mean, _, _, tree_pooled = policy.parameters_for_edges(
        features, topology_masks=tree_vector, return_pooled=True)
    dense_mean, _, _, dense_pooled = policy.parameters_for_edges(
        features, topology_masks=dense_vector, return_pooled=True)
    assert not torch.allclose(tree_mean, dense_mean, atol=1e-10, rtol=0)
    tree_budget, _ = policy.budget_parameters(features, pooled_states=tree_pooled)
    dense_budget, _ = policy.budget_parameters(features, pooled_states=dense_pooled)
    assert not torch.allclose(tree_budget, dense_budget, atol=1e-10, rtol=0)

    design = infer_designs(policy, target, detunings, config, count=4,
                           topology=tree, seed=5, continuous_mean=True)
    order = np.argsort(detunings)
    sorted_tree = tree[order][:, order]
    np.testing.assert_array_equal(design["kappa_per_ns"] > 0,
                                  np.broadcast_to(sorted_tree.T, (4, 3, 3)))
    assert np.all(design["active_link_counts"] == 2)
    assert np.all(design["redistribution_shortfall_per_ns"] == 0)


def test_topology_input_training_uses_phase_reward_only(monkeypatch):
    config = replace(previous_topology_config, training_n_lasers=(3,), n_lasers=3,
                     n_jobs=1, targets_per_batch=2, candidates_per_target=8,
                     actions_per_topology=2, candidates_per_worker_task=8,
                     targets_per_worker_task=2,
                     de_teacher_enabled=False,
                     coupling_budget_reference_by_m=((3, 10.0),))
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    before = policy.budget_head[-1].weight.detach().clone()

    def fake_chunk(work):
        group, start, stop, item = work
        _, target_start, target_stop, *_ = item
        values = np.linspace(0.2, 0.9, config.candidates_per_target)[start:stop]
        return dict(size_batch_index=group, candidate_start_index=start,
                    candidate_stop_index=stop, start_index=target_start,
                    stop_index=target_stop,
                    phase_reward=np.tile(values, (target_stop - target_start, 1)))

    monkeypatch.setattr(bernoulli.base, "simulate_indexed_target_chunk", fake_chunk)
    metrics = run_training_batch(policy, optimizer, config, np.random.default_rng(9), 0)
    assert metrics["training_reward"] == pytest.approx(metrics["phase_reward"])
    assert metrics["rooted_fraction"] == 1.0
    assert metrics["sampled_budget_median_per_laser"] > 0
    assert not torch.equal(before, policy.budget_head[-1].weight)


def test_worst_topology_selection_merges_duplicate_masks():
    masks = np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                      [0, 1], [0, 1]])
    rewards = np.array([0.1, 0.9, 0.0, 0.1, 0.2, 0.3])
    assert bernoulli.worst_topology_policy_candidate(rewards, masks, 2) == 5


def test_phase_de_teacher_uses_exact_vectorized_budget(monkeypatch):
    config = replace(previous_topology_config, n_lasers=3)
    kappa = np.zeros((3, 3))
    kappa[1:, 0] = 1.0
    phi = np.zeros((3, 3))
    phi[1:, 0] = np.pi
    batch_sizes = []

    def fake_simulate(kappa_cases, phi_cases, _config, **_kwargs):
        batch_sizes.append(len(phi_cases))
        np.testing.assert_array_equal(kappa_cases, np.broadcast_to(kappa, kappa_cases.shape))
        return np.array([0.0]), phi_cases, None, None

    def fake_reward(_time, phi_cases, _targets, _config):
        reward = ((1.0 + np.cos(phi_cases[:, 1:, 0])) / 2.0).mean(axis=1)
        return reward, None, None, None, None, None

    monkeypatch.setattr(bernoulli.base, "simulate_with_vcsel", fake_simulate)
    monkeypatch.setattr(bernoulli.base, "calculate_phase_reward", fake_reward)
    result = bernoulli.phase_de_teacher_search(dict(
        config=config, group=0, target_index=0, candidate_index=0,
        baseline_reward=0.0, target_phases_rad=np.zeros(3),
        detunings_ghz=np.zeros(3), kappa_per_ns=kappa, phi_p_rad=phi,
        seed=7,
    ))
    assert batch_sizes == [5, 5, 5, 5]
    assert result["evaluations"] == 20
    assert result["best_reward"] > result["baseline_reward"]
    np.testing.assert_array_equal(result["phi_p_rad"][kappa == 0], 0.0)


def test_de_teacher_adds_phase_gradient_without_changing_reinforce_samples(monkeypatch):
    config = replace(previous_topology_config, training_n_lasers=(3,), n_lasers=3,
                     n_jobs=1, targets_per_batch=2, candidates_per_target=8,
                     actions_per_topology=2, candidates_per_worker_task=8,
                     targets_per_worker_task=1,
                     de_teacher_enabled=True, pcgrad_task_grouping="array_size",
                     coupling_budget_reference_by_m=((3, 10.0),))
    without_teacher = replace(config, de_teacher_enabled=False)
    torch.manual_seed(21)
    policy = BernoulliSparseGNNPolicy(config)
    baseline_policy = BernoulliSparseGNNPolicy(without_teacher)
    baseline_policy.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    baseline_optimizer = torch.optim.Adam(
        baseline_policy.parameters(), lr=without_teacher.learning_rate)
    selected = []

    def fake_chunk(work):
        group, start, stop, item = work
        _, target_start, target_stop, *_ = item
        values = np.linspace(0.2, 0.9, config.candidates_per_target)[start:stop]
        return dict(size_batch_index=group, candidate_start_index=start,
                    candidate_stop_index=stop, start_index=target_start,
                    stop_index=target_stop,
                    phase_reward=np.tile(values, (target_stop - target_start, 1)))

    def fake_de(work):
        selected.append(work)
        teacher_phi = np.zeros_like(work["kappa_per_ns"])
        teacher_phi[work["kappa_per_ns"] > 0] = np.pi / 2
        return dict(group=work["group"], target_index=work["target_index"],
                    candidate_index=work["candidate_index"],
                    baseline_reward=work["baseline_reward"],
                    best_reward=work["baseline_reward"] + 0.1,
                    phi_p_rad=teacher_phi, evaluations=20)

    monkeypatch.setattr(bernoulli.base, "simulate_indexed_target_chunk", fake_chunk)
    monkeypatch.setattr(bernoulli, "phase_de_teacher_search", fake_de)
    torch.manual_seed(31)
    metrics = run_training_batch(policy, optimizer, config,
                                 np.random.default_rng(9), 500)
    torch.manual_seed(31)
    baseline_metrics = run_training_batch(
        baseline_policy, baseline_optimizer, without_teacher,
        np.random.default_rng(9), 500)
    assert len(selected) == 2
    assert metrics["de_teacher_searches"] == 2
    assert metrics["de_teacher_simulations"] == 40
    assert metrics["de_teacher_improved"] == 2
    assert metrics["de_teacher_loss"] > 0
    assert baseline_metrics["de_teacher_searches"] == 0
    assert not torch.allclose(policy.edge_output.weight[1],
                              baseline_policy.edge_output.weight[1], atol=1e-12)


def test_topology_advantage_uses_shared_running_scale():
    config = replace(previous_topology_config, actions_per_topology=2,
                     candidates_per_target=4, global_reward_std_ema_decay=0.5)
    rewards = [np.array([[0.1, 0.3, 0.8, 0.82]]),
               np.array([[0.2, 0.4, 0.7, 0.72]])]
    state = {}
    advantages = bernoulli._topology_advantages(rewards, config, state)
    scale = np.std(np.concatenate([reward.ravel() for reward in rewards]))
    np.testing.assert_allclose(advantages[0], [[-0.1 / scale, 0.1 / scale,
                                                -0.01 / scale, 0.01 / scale]])
    assert np.sqrt(state["variance"]) == pytest.approx(scale)

    next_rewards = [np.array([[0.4, 0.6, 0.5, 0.7]])]
    next_advantages = bernoulli._topology_advantages(next_rewards, config, state)
    next_variance = 0.5 * scale**2 + 0.5 * np.var(next_rewards[0])
    assert state["variance"] == pytest.approx(next_variance)
    np.testing.assert_allclose(next_advantages[0],
                               np.array([[-0.1, 0.1, -0.1, 0.1]]) / np.sqrt(next_variance))


def test_topology_input_heldout_masks_cover_fixed_full_star_range(monkeypatch):
    config = replace(previous_topology_config, n_lasers=3,
                     coupling_budget_reference_by_m=((3, 10.0),))
    policy = BernoulliSparseGNNPolicy(config)
    observed = []

    def fake_simulation(_targets, kappa, _phi, _config, **_kwargs):
        observed.append((kappa > 0).sum(axis=(-2, -1)).ravel())
        return dict(phase_reward=np.full((len(kappa), 1), 0.7))

    monkeypatch.setattr(bernoulli.base, "simulate_candidates_parallel", fake_simulation)
    targets = np.zeros((64, 3))
    detunings = np.tile([-0.05, 0.0, 0.05], (64, 1))
    first = evaluate_policy(policy, targets, detunings, config, seed=42)
    second = evaluate_policy(policy, targets, detunings, config, seed=42)
    np.testing.assert_array_equal(observed[0], observed[1])
    levels, frequencies = np.unique(observed[0], return_counts=True)
    np.testing.assert_array_equal(levels, np.arange(2, 7))
    assert frequencies.max() - frequencies.min() <= 1
    assert first["phase_reward_by_density"] == {
        "sparse": pytest.approx(0.7),
        "middle": pytest.approx(0.7),
        "dense": pytest.approx(0.7),
    }
    assert first["training_reward"] == first["phase_reward"] == pytest.approx(0.7)
    assert first["retained_link_fraction"] == second["retained_link_fraction"]


def test_topology_input_checkpoint_round_trip(tmp_path):
    config = replace(previous_topology_config, n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    path = tmp_path / "topology_input.pt"
    save_checkpoint(path, policy, optimizer, config, 17, 0.7)
    loaded, saved_config, metadata = load_checkpoint(path)
    assert saved_config.topology_input_mode
    assert saved_config.objective_mode == "phase_only"
    assert saved_config.budget_exploration_mode == "local"
    assert saved_config.budget_wide_exploration_fraction == pytest.approx(0.0)
    assert saved_config.training_topology_min_fraction == pytest.approx(0.0)
    assert saved_config.training_topology_density_bias == pytest.approx(1.0)
    assert saved_config.topology_star_backbone
    assert saved_config.topology_curriculum_start_iteration == 0
    assert saved_config.topology_curriculum_end_iteration == 500
    assert saved_config.strength_exploration_fraction == 0.0
    assert saved_config.phase_exploration_fraction == pytest.approx(0.25)
    assert not saved_config.de_teacher_enabled
    assert saved_config.pcgrad_task_grouping == "array_size_topology_density"
    assert saved_config.de_teacher_population == 5
    assert saved_config.de_teacher_generations == 3
    assert metadata["iteration"] == 17
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    mask = features.new_ones((1, 6))
    original = policy.parameters_for_edges(features, topology_masks=mask)[0]
    restored = loaded.parameters_for_edges(features, topology_masks=mask)[0]
    torch.testing.assert_close(original, restored)


def test_topology_input_progress_figure_shows_phase_not_duplicate_reward():
    history = dict(training_reward=[0.3, 0.6], phase_reward=[0.3, 0.6],
                   retained_link_fraction=[0.5, 0.5],
                   held_out_by_m={3: [0.2, 0.4]}, held_out_reward=[0.2, 0.4])
    figure = training_figure(history, previous_topology_config)
    try:
        assert len(figure.axes) == 2
        assert figure.axes[1].get_title() == "Star-to-dense held-out targets"
        labels = [line.get_label() for line in figure.axes[0].lines]
        assert "phase reward" in labels
        assert "total reward" not in labels
    finally:
        plt.close(figure)


def test_constructed_validation_simulates_every_candidate(monkeypatch):
    config = replace(run_config, n_lasers=3, construct_rooted_backbone=True,
                     require_rooted_connectivity=False,
                     objective_mode="weighted",
                         dense_reference_feasibility=False,
                     coupling_budget_reference_by_m=((3, 20.0),))
    policy = BernoulliSparseGNNPolicy(config)
    with torch.no_grad():
        policy.edge_gate_head.weight.zero_()
        policy.edge_gate_head.bias.fill_(-100.0)
    calls = []
    def fake_simulation(targets, kappa, phi, _config, **_kwargs):
        calls.append(len(targets))
        assert np.all(kappa.sum(axis=(-2, -1)) > 0)
        return {"phase_reward": np.full((len(targets), 1), 0.9)}
    monkeypatch.setattr(bernoulli.base, "simulate_candidates_parallel", fake_simulation)
    results = evaluate_policy(policy, np.zeros((2, 3)),
                              np.tile([-0.05, 0, 0.05], (2, 1)), config)
    assert calls == [2]
    assert results["rooted_fraction"] == 1.0
    assert results["training_reward"] > 0


def test_nonrooted_reward_is_negative_and_ignores_budget():
    config = legacy_config
    phase = np.array([1.0, 0.8, 0.4])
    reach = np.array([0.0, 0.5, 1.0])
    score = reward_function(phase, np.array([0.0, 0.2, 0.4]),
                            np.array([0.0, 2.0, 0.8]), config,
                            root_reach_fraction=reach)
    np.testing.assert_allclose(score[:2], [-1.0, -0.5])
    assert score[2] > 0
    with pytest.raises(ValueError, match="root_reach_fraction"):
        reward_function(phase, np.zeros(3), np.zeros(3), config)


def test_rooted_chunk_simulates_only_valid_cases(monkeypatch):
    config = replace(legacy_config, n_lasers=3)
    targets = np.zeros((2, 3))
    detunings = np.zeros((2, 3))
    kappa = np.zeros((2, 2, 3, 3))
    phi = np.zeros_like(kappa)
    rooted = np.array([[True, False], [False, True]])
    calls = []

    def fake_simulate(case_kappa, case_phi, _config, **kwargs):
        calls.append((case_kappa.shape, case_phi.shape,
                      kwargs["detuning_distribution_ghz"].shape))
        return np.array([0.0]), np.empty((2, 9, 1)), None, None

    def fake_phase(_time, _states, case_targets, _config):
        assert case_targets.shape == (2, 3)
        return (np.array([0.8, 0.6]), None, None, None, np.zeros((2, 3)))

    monkeypatch.setattr(bernoulli.base, "simulate_with_vcsel", fake_simulate)
    monkeypatch.setattr(bernoulli.base, "calculate_phase_reward", fake_phase)
    work_item = (0, 0, 2, targets, kappa, phi, detunings, config)
    result = simulate_rooted_indexed_chunk((1, 0, 2, work_item, rooted))
    assert calls == [((2, 3, 3), (2, 3, 3), (2, 3))]
    np.testing.assert_allclose(result["phase_reward"], [[0.8, 0], [0, 0.6]])
    calls.clear()
    result = simulate_rooted_indexed_chunk((1, 0, 2, work_item,
                                            np.zeros_like(rooted)))
    assert calls == []
    np.testing.assert_allclose(result["phase_reward"], 0)


def test_validation_skips_all_off_masks(monkeypatch):
    config = replace(legacy_config, n_lasers=3,
                     coupling_budget_reference_by_m=((3, 20.0),))
    policy = BernoulliSparseGNNPolicy(config)
    with torch.no_grad():
        policy.edge_gate_head.weight.zero_()
        policy.edge_gate_head.bias.fill_(-100.0)

    def unexpected_simulation(*_args, **_kwargs):
        raise AssertionError("non-rooted validation masks must not be simulated")

    monkeypatch.setattr(bernoulli.base, "simulate_candidates_parallel",
                        unexpected_simulation)
    results = evaluate_policy(policy, np.zeros((2, 3)),
                              np.tile([-0.05, 0, 0.05], (2, 1)), config)
    assert results["rooted_fraction"] == 0
    assert results["phase_reward"] == 0
    assert results["training_reward"] == -1


def test_coupling_cost_waits_until_2000_then_ramps_for_500():
    config = replace(legacy_config, coupling_cost_start_iteration=2000,
                     coupling_cost_ramp_iterations=500)
    assert coupling_cost_weight_at(1999, config) == 0.0
    assert coupling_cost_weight_at(2000, config) == 0.0
    assert coupling_cost_weight_at(2250, config) == pytest.approx(0.075)
    assert coupling_cost_weight_at(2500, config) == pytest.approx(0.15)
    assert coupling_cost_weight_at(3000, config) == pytest.approx(0.15)
    phase, rho, budget = np.array([1.0]), np.array([0.5]), np.array([1.0])
    rooted = np.ones_like(phase)
    before = reward_function(phase, rho, budget, config, iteration=2000,
                             root_reach_fraction=rooted)
    middle = reward_function(phase, rho, budget, config, iteration=2250,
                             root_reach_fraction=rooted)
    after = reward_function(phase, rho, budget, config, iteration=2500,
                            root_reach_fraction=rooted)
    validation = reward_function(phase, rho, budget, config,
                                 root_reach_fraction=rooted)
    assert before[0] == pytest.approx(1 / (1 + 0.25 * 0.5))
    assert middle[0] == pytest.approx(1 / (1 + 0.25 * 0.5 + 0.075))
    assert after[0] == pytest.approx(validation[0])


def test_frozen_budget_reference_depends_on_m():
    config = replace(run_config, training_n_lasers=(3, 4),
                     budget_reference_calibration_targets=4)
    policy = BernoulliSparseGNNPolicy(config)
    calibrated = calibrate_budget_references(policy, config)
    assert len(calibrated.coupling_budget_reference_by_m) == 2
    assert config.coupling_budget_reference_by_m == ()
    reference_3 = reference_budget_for_m(calibrated, 3)
    reference_4 = reference_budget_for_m(calibrated, 4)
    assert reference_4 > reference_3 > 0
    assert relative_coupling_budget(np.full((3, 3), reference_3 / 9),
                                    replace(calibrated, n_lasers=3)) == pytest.approx(1)
    # Outside training, preserve the physical decoder's size normalization.
    factor = (lambda size: size if calibrated.normalize_incoming_coupling_by_degree
              else size * (size - 1))
    assert reference_budget_for_m(calibrated, 100) / factor(100) == pytest.approx(
        np.median([reference_3 / factor(3), reference_4 / factor(4)])
    )


def test_training_batches_two_targets_per_size_job():
    config = replace(
        legacy_config, training_n_lasers=(3, 4), targets_per_batch=8,
        targets_per_worker_task=2, candidates_per_target=2,
        candidates_per_worker_task=2,
        coupling_budget_reference_by_m=((3, 1.0), (4, 1.0)),
    )
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    class RecordingPool:
        def __init__(self):
            self.work = []

        def imap_unordered(self, _function, work):
            self.work = list(work)
            for group, candidate_start, candidate_stop, item, rooted in reversed(self.work):
                _, target_start, target_stop, *_ = item
                assert rooted.shape == (target_stop - target_start,
                                        candidate_stop - candidate_start)
                yield {
                    "size_batch_index": group,
                    "candidate_start_index": candidate_start,
                    "candidate_stop_index": candidate_stop,
                    "start_index": target_start,
                    "stop_index": target_stop,
                    "phase_reward": np.tile([0.2, 0.8], (target_stop - target_start, 1)),
                }

    pool = RecordingPool()
    run_training_batch(policy, optimizer, config, np.random.default_rng(7), 1001,
                       simulation_pool=pool)
    assert len(pool.work) == 4
    for _, candidate_start, candidate_stop, item, rooted in pool.work:
        _, target_start, target_stop, targets, kappa, phi, detunings, local = item
        assert target_stop - target_start == 2
        assert candidate_stop - candidate_start == 2
        assert targets.shape == detunings.shape == (2, local.n_lasers)
        assert kappa.shape == phi.shape == (2, 2, local.n_lasers, local.n_lasers)
        assert rooted.shape == (2, 2)


def test_training_simulates_own_all_on_reference_for_each_target(monkeypatch):
    config = replace(run_config, training_n_lasers=(3,), n_lasers=3,
                     objective_mode="weighted",
                     targets_per_batch=1, candidates_per_target=2,
                     candidates_per_worker_task=2, targets_per_worker_task=1,
                     coupling_budget_reference_by_m=((3, 10.0),),
                     dense_reference_feasibility=True)
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def fake_dense(_policy, targets, _detunings, local):
        matrices = np.zeros((len(targets), 1, local.n_lasers, local.n_lasers))
        return matrices, matrices.copy()

    monkeypatch.setattr(bernoulli, "dense_reference_designs", fake_dense)

    class RecordingPool:
        def __init__(self):
            self.ranges = []

        def imap_unordered(self, _worker, work):
            for group, start, stop, item in reversed(list(work)):
                self.ranges.append((start, stop))
                _, target_start, target_stop, *_ = item
                values = np.array([0.98, 0.80, 0.95])[start:stop]
                yield dict(size_batch_index=group, candidate_start_index=start,
                           candidate_stop_index=stop, start_index=target_start,
                           stop_index=target_stop,
                           phase_reward=values[None].repeat(target_stop - target_start, axis=0))

    pool = RecordingPool()
    metrics = run_training_batch(policy, optimizer, config, np.random.default_rng(7),
                                 1001, simulation_pool=pool)
    assert sorted(pool.ranges) == [(0, 2), (2, 3)]
    assert metrics["dense_reference_phase_reward"] == pytest.approx(0.95)
    assert metrics["feasible_candidate_fraction"] == pytest.approx(0.5)
    assert metrics["targets_with_feasible_candidate"] == pytest.approx(1.0)


def test_previous_topology_run_configuration_remains_supported():
    config = previous_topology_config
    assert config.training_n_lasers == tuple(range(3, 10))
    assert config.targets_per_batch == 14
    assert config.targets_per_worker_task == 1
    assert config.candidates_per_target == 72
    assert config.actions_per_topology == 8
    assert config.topology_advantage_scale == "global_running"
    assert config.n_jobs == 14
    assert config.topology_input_mode
    assert not config.construct_rooted_backbone
    assert config.objective_mode == "phase_only"
    assert config.sparsity_reward_mode == "links_per_laser"
    assert config.link_cost_weight == 0.0
    assert config.coupling_cost_weight == 0.0
    assert config.coupling_cost_start_iteration == 0
    assert config.coupling_cost_ramp_iterations == 0
    np.testing.assert_allclose(
        reward_function(np.ones(2), np.full(2, 0.5), np.array([0.0, 100.0]), config),
        np.full(2, reward_function(np.ones(1), np.full(1, 0.5), np.zeros(1), config)[0]),
    )
    assert config.structured_target_fraction == 0.0
    assert not config.require_rooted_connectivity
    assert config.uncapped_redistribution
    assert config.strength_action_transform == "uncapped_softplus"
    assert config.softplus_strength_scale_per_ns == pytest.approx(1.0)
    assert config.budget_action_transform == "log_budget"
    assert config.budget_exploration_mode == "local"
    assert config.budget_wide_exploration_fraction == 0.0
    assert config.strength_exploration_fraction == 0.0
    assert config.phase_exploration_fraction == pytest.approx(0.25)
    assert config.phase_exploration_start_std == pytest.approx(np.pi)
    assert config.phase_exploration_start_iteration == 500
    assert config.phase_exploration_end_iteration == 4000
    assert config.training_topology_min_fraction == pytest.approx(0.0)
    assert config.training_topology_density_bias == pytest.approx(1.0)
    assert config.topology_star_backbone
    assert config.topology_curriculum_start_iteration == 0
    assert config.topology_curriculum_end_iteration == 500
    assert config.log_budget_initial_per_laser == pytest.approx(0.5)
    assert config.budget_exploration_end_std == pytest.approx(0.35)
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.lexicographic_maximum_phase_error_deg == pytest.approx(3.0)
    assert config.lexicographic_initial_phase_error_deg == pytest.approx(30.0)
    assert config.lexicographic_phase_error_anneal_iterations == 0
    assert config.lexicographic_minimum_phase_reward == pytest.approx(0.98)
    assert config.lexicographic_initial_minimum_phase_reward == pytest.approx(
        0.5 * (1.0 + np.cos(np.deg2rad(30.0))))
    assert config.current_checkpoint_file.endswith(
        "conditional_coupling_variable_m_3to9_lasers_current_star_wide_phase_topology_pcgrad_no_de.pt")
    assert not config.de_teacher_enabled
    assert config.pcgrad_task_grouping == "array_size_topology_density"
    assert config.de_teacher_population == 5
    assert config.de_teacher_generations == 3
    assert config.de_teacher_interval == 1
    assert not config.dense_reference_feasibility
    assert not config.plot_coupling_reference
    assert config.detuning_curriculum_warmup_iterations == 0
    assert config.detuning_curriculum_iterations == 500
    assert config.detuning_curriculum_initial_half_span_ghz == pytest.approx(0.005)
    assert designer.detuning_curriculum_half_span_at(0, config) == pytest.approx(0.005)
    assert designer.detuning_curriculum_half_span_at(1, config) > 0.005
    assert designer.detuning_curriculum_half_span_at(500, config) == pytest.approx(2.0)
    assert designer.detuning_curriculum_half_span_at(1000, config) == pytest.approx(2.0)


def test_star_curriculum_run_keeps_phase_only_objective_and_separate_checkpoints():
    config = previous_topology_config
    assert config.topology_input_mode
    assert config.objective_mode == "phase_only"
    assert config.topology_advantage_scale == "global_running"
    assert config.actions_per_topology == 8
    assert config.candidates_per_target == 72
    assert config.budget_action_transform == "log_budget"
    assert config.budget_exploration_mode == "local"
    assert config.training_topology_min_fraction == pytest.approx(0.0)
    assert config.training_topology_density_bias == pytest.approx(1.0)
    assert config.topology_star_backbone
    assert config.topology_curriculum_start_iteration == 0
    assert config.topology_curriculum_end_iteration == 500
    assert config.strength_exploration_fraction == 0.0
    assert config.phase_exploration_fraction == pytest.approx(0.25)
    assert config.current_checkpoint_file.endswith("_star_wide_phase_topology_pcgrad_no_de.pt")
    assert not config.current_checkpoint_file.endswith("_global_advantage.pt")


def test_previous_topology_run_widens_only_two_phase_actions_per_topology():
    config = previous_topology_config
    log_std = torch.zeros((1, config.actions_per_topology, 12))
    early = bernoulli._topology_action_std(log_std, config, 0, False)
    held = bernoulli._topology_action_std(log_std, config, 500, False)
    late = bernoulli._topology_action_std(log_std, config, 4000, False)
    np.testing.assert_allclose(early, held)
    np.testing.assert_allclose(early[..., :6], 1.0)
    np.testing.assert_allclose(early[0, :2, 6:], np.sqrt(1.0 + np.pi**2))
    np.testing.assert_allclose(early[0, 2:, 6:], 1.0)
    np.testing.assert_allclose(late, 1.0)
    np.testing.assert_allclose(
        bernoulli._topology_action_std(log_std, config, None, False), 1.0)


def test_wide_strength_sampling_preserves_local_actions_and_phase_variance():
    config = replace(previous_topology_config, strength_exploration_fraction=0.5,
                     strength_exploration_start_std=5.0,
                     phase_exploration_fraction=0.0)
    log_std = torch.zeros((1, 16, 12))
    early = bernoulli._topology_action_std(log_std, config, 0, False)
    wide = np.r_[np.arange(4), np.arange(8, 12)]
    local = np.r_[np.arange(4, 8), np.arange(12, 16)]
    np.testing.assert_allclose(early[0, wide, :6], np.sqrt(26.0))
    np.testing.assert_allclose(early[0, local, :6], 1.0)
    np.testing.assert_allclose(early[..., 6:], 1.0)
    np.testing.assert_allclose(
        bernoulli._topology_action_std(log_std, config, 4000, False), 1.0)
    np.testing.assert_allclose(
        bernoulli._topology_action_std(log_std, config, None, False), 1.0)


def test_wide_phase_sampling_repeats_per_topology_and_preserves_strength_variance():
    config = previous_topology_config
    log_std = torch.zeros((1, 16, 12))
    early = bernoulli._topology_action_std(log_std, config, 0, False)
    wide = np.r_[np.arange(2), np.arange(8, 10)]
    local = np.r_[np.arange(2, 8), np.arange(10, 16)]
    np.testing.assert_allclose(early[..., :6], 1.0)
    np.testing.assert_allclose(early[0, wide, 6:], np.sqrt(1.0 + np.pi**2))
    np.testing.assert_allclose(early[0, local, 6:], 1.0)
    np.testing.assert_allclose(
        bernoulli._topology_action_std(log_std, config, 4000, False), 1.0)
    np.testing.assert_allclose(
        bernoulli._topology_action_std(log_std, config, None, False), 1.0)


def test_wide_budget_sampling_is_centered_and_low_rewarded_draw_pulls_mean_down():
    config = replace(previous_topology_config, budget_exploration_mode="mixed_std",
                     budget_wide_exploration_fraction=0.5,
                     budget_exploration_start_iteration=500,
                     budget_exploration_end_iteration=4000,
                     budget_exploration_start_std=1.4,
                     budget_exploration_end_std=0.1)
    mean = torch.zeros((1, 16))
    log_std = torch.full_like(mean, np.log(0.1))
    early_mean, early_std = bernoulli._budget_action_parameters(
        mean, log_std, config, 0, None, 8)
    wide = np.r_[np.arange(4), np.arange(8, 12)]
    local = np.r_[np.arange(4, 8), np.arange(12, 16)]
    np.testing.assert_allclose(early_mean, 0.0)
    np.testing.assert_allclose(early_std[0, wide], 1.4, rtol=1e-6)
    np.testing.assert_allclose(early_std[0, local], 0.1)
    _, late_std = bernoulli._budget_action_parameters(
        mean, log_std, config, 4000, None, 8)
    _, inference_std = bernoulli._budget_action_parameters(
        mean, log_std, config, None, None, 8)
    np.testing.assert_allclose(late_std, 0.1)
    np.testing.assert_allclose(inference_std, 0.1)

    predicted_log_budget = torch.tensor(0.0, requires_grad=True)
    low_budget_draw = torch.tensor(-np.log(10.0), dtype=torch.float32)
    learned_mean, sampled_std = bernoulli._budget_action_parameters(
        predicted_log_budget.expand(1, 8), torch.full((1, 8), np.log(0.1)),
        config, 0, None, 8)
    loss = -torch.distributions.Normal(learned_mean[0, 0], sampled_std[0, 0]).log_prob(
        low_budget_draw)
    loss.backward()
    assert predicted_log_budget.grad > 0  # gradient descent lowers the predicted budget


def test_mixed_budget_sampling_does_not_use_offsets_and_trains_both_heads(monkeypatch):
    config = replace(previous_topology_config, n_lasers=3,
                     budget_exploration_mode="mixed_std",
                     budget_wide_exploration_fraction=0.5)
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    masks = torch.ones((1, 16, 6), dtype=torch.float32)
    monkeypatch.setattr(bernoulli, "_budget_offsets", lambda *_: pytest.fail("Offsets are disabled"))
    actions, budgets, log_prob = bernoulli.sample_topology_conditioned_candidates(
        policy, features, masks, iteration=0, rng=np.random.default_rng(4))
    assert actions.shape == (1, 16, 12)
    assert budgets.shape == log_prob.shape == (1, 16)
    assert torch.isfinite(log_prob).all()
    (-log_prob.mean()).backward()
    assert policy.budget_head[-1].weight.grad.abs().sum() > 0
    assert policy.edge_output.weight.grad[0].abs().sum() > 0


def test_new_run_budget_starts_small_and_uses_only_local_sampling(monkeypatch):
    config = replace(previous_topology_config, n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    mask = torch.ones((1, 6))
    _, _, _, pooled = policy.parameters_for_edges(
        features, topology_masks=mask, return_pooled=True)
    mean, log_std = policy.budget_parameters(features, pooled_states=pooled)
    assert float(mean.exp().detach()[0]) == pytest.approx(1.5, rel=0.05)
    assert float(log_std.exp().detach()[0]) == pytest.approx(0.35)
    monkeypatch.setattr(bernoulli, "_budget_offsets", lambda *_: pytest.fail("No offsets"))
    repeated_mean = mean[:, None].expand(1, 8)
    repeated_std = log_std[:, None].expand(1, 8)
    early = bernoulli._budget_action_parameters(
        repeated_mean, repeated_std, config, 0, np.random.default_rng(4), 8)
    late = bernoulli._budget_action_parameters(
        repeated_mean, repeated_std, config, 4000, np.random.default_rng(4), 8)
    torch.testing.assert_close(early[0], repeated_mean)
    torch.testing.assert_close(early[1], repeated_std.exp())
    torch.testing.assert_close(late[0], early[0])
    torch.testing.assert_close(late[1], early[1])


def test_legacy_log_budget_offsets_still_cover_each_topology_group(monkeypatch):
    config = replace(previous_topology_config, n_lasers=3,
                     budget_exploration_mode="stratified_offsets")
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    masks = torch.ones((1, 16, 6), dtype=torch.float32)
    original = bernoulli._budget_offsets
    calls = []

    def recording_offsets(targets, candidates, width, rng, device):
        calls.append((targets, candidates))
        offsets = original(targets, candidates, width, rng, device)
        np.testing.assert_allclose(offsets.min(dim=1).values, -width)
        np.testing.assert_allclose(offsets.max(dim=1).values, width)
        return offsets

    monkeypatch.setattr(bernoulli, "_budget_offsets", recording_offsets)
    bernoulli.sample_topology_conditioned_candidates(
        policy, features, masks, iteration=0, rng=np.random.default_rng(4))
    assert calls == [(2, 8)]


def test_training_budget_ratio_has_its_own_axis():
    history = dict(
        training_reward=[0.4, 0.7], phase_reward=[0.5, 0.9],
        retained_link_fraction=[0.5, 0.3], rooted_fraction=[0.6, 0.9],
        normalized_coupling_budget=[1.0, 3.5],
        held_out_by_m={3: [0.3, 0.6]}, held_out_reward=[0.3, 0.6],
        held_out_rooted_fraction=[0.5, 0.8],
    )
    figure = training_figure(history, replace(run_config, objective_mode="weighted",
                                              dense_reference_feasibility=False,
                                              plot_coupling_reference=True))
    try:
        reward_axis = next(axis for axis in figure.axes
                           if axis.get_title() == "REINFORCE training")
        budget_axis = next(axis for axis in figure.axes
                           if axis.get_ylabel() == "coupling / initial dense reference")
        assert reward_axis.get_ylim()[1] == pytest.approx(1.05)
        assert budget_axis.get_ylim()[1] > 3.5
    finally:
        plt.close(figure)


def test_training_figure_omits_budget_axis_for_next_run():
    history = dict(
        training_reward=[0.4, 0.7], phase_reward=[0.5, 0.9],
        retained_link_fraction=[0.5, 0.3], normalized_coupling_budget=[1.0, 3.5],
        held_out_by_m={3: [0.3, 0.6]}, held_out_reward=[0.3, 0.6],
    )
    figure = training_figure(history, replace(run_config, objective_mode="weighted"))
    try:
        assert len(figure.axes) == 2
        assert all(axis.get_ylabel() != "coupling / initial dense reference"
                   for axis in figure.axes)
    finally:
        plt.close(figure)


def test_star_curriculum_figure_shows_fixed_heldout_density_bands():
    history = dict(
        training_reward=[0.4, 0.7], phase_reward=[0.5, 0.9],
        retained_link_fraction=[1.0, 0.6],
        held_out_by_m={3: [0.3, 0.6]}, held_out_reward=[0.3, 0.6],
        held_out_by_density={
            "sparse": [0.1, 0.3], "middle": [0.3, 0.6],
            "dense": [0.8, 0.95],
        },
    )
    figure = training_figure(history, previous_topology_config)
    try:
        labels = {line.get_label() for line in figure.axes[1].lines}
        assert {"sparse topology", "middle topology", "dense topology", "overall"} <= labels
    finally:
        plt.close(figure)


def test_lexicographic_training_figure_has_two_panes():
    history = dict(
        training_reward=[0.4, 0.5], phase_reward=[0.5, 0.8],
        retained_link_fraction=[0.6, 0.5],
        feasible_candidate_fraction=[0.1, 0.3],
        targets_with_feasible_candidate=[0.3, 0.7],
        held_out_by_m={3: [0.5, 0.8]}, held_out_reward=[0.5, 0.8],
        held_out_feasible_fraction=[0.1, 0.5],
        sampled_budget_p10_per_laser=[0.1, 0.4],
        sampled_budget_median_per_laser=[1.0, 0.9],
        sampled_budget_p90_per_laser=[10.0, 2.0],
        selected_feasible_budget_median_per_laser=[3.0, 1.5],
        held_out_budget_per_laser=[2.0, 1.2],
    )
    figure = training_figure(history, run_config)
    try:
        assert len(figure.axes) == 2
        assert any(line.get_label() == "targets with feasible sample"
                   for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_plotted_phase_reward_tail_mean_matches_training_reward():
    config = replace(run_config, n_lasers=3)
    time_seconds = np.linspace(0.0, config.simulation_time_seconds, 101)
    states = np.zeros((1, 3 * config.n_lasers, len(time_seconds)))
    states[0, 5] = np.linspace(0.1, 0.5, len(time_seconds))
    states[0, 8] = np.linspace(-0.5, -0.1, len(time_seconds))
    targets = np.zeros((1, config.n_lasers))
    reward, _, _, _, _, orientation = designer.calculate_phase_reward(
        time_seconds, states, targets, config)
    relative_phases = states[:, 2::3] - states[:, 2:3]
    errors = orientation[:, None, None] * relative_phases - targets[:, :, None]
    trace, tail_start = phase_reward_trace(time_seconds, errors, config)
    assert trace.shape == (1, len(time_seconds))
    assert np.all((0.0 <= trace) & (trace <= 1.0))
    np.testing.assert_allclose(trace[:, tail_start:].mean(axis=1), reward, atol=1e-12)


def test_lexicographic_order_changes_only_feasible_resource_comparison():
    phase = np.array([[0.99, 0.995, 0.5, 0.999]])
    error = np.array([[1.0, 2.0, 30.0, 5.0]])
    links = np.array([[3, 4, 2, 2]])
    budgets = np.array([[100.0, 10.0, 1.0, 1.0]])
    lex_config = replace(run_config, objective_mode="lexicographic",
                         lexicographic_order="sparsity_budget")
    sparse_score, feasible = lexicographic_candidate_scores(
        phase, error, links, budgets, lex_config)
    budget_score, _ = lexicographic_candidate_scores(
        phase, error, links, budgets,
        replace(lex_config, lexicographic_order="budget_sparsity"))
    np.testing.assert_array_equal(feasible, [[True, True, False, False]])
    assert np.argmax(sparse_score, axis=1).tolist() == [0]
    assert np.argmax(budget_score, axis=1).tolist() == [1]
    assert sparse_score[0, 2] < sparse_score[0, 3] < sparse_score[0, 1]


def test_phase_error_curriculum_only_relaxes_training_feasibility():
    config = replace(run_config, lexicographic_phase_error_anneal_iterations=3500)
    assert lexicographic_phase_error_limit_at(0, config) == pytest.approx(30.0)
    assert lexicographic_phase_error_limit_at(1000, config) == pytest.approx(30.0 - 27.0 * 1000 / 3500)
    assert lexicographic_phase_error_limit_at(3500, config) == pytest.approx(3.0)
    assert lexicographic_phase_error_limit_at(5000, config) == pytest.approx(3.0)
    assert lexicographic_phase_reward_limit_at(0, config) == pytest.approx(
        0.5 * (1.0 + np.cos(np.deg2rad(30.0))))
    assert lexicographic_phase_reward_limit_at(1000, config) == pytest.approx(
        config.lexicographic_initial_minimum_phase_reward
        + (0.98 - config.lexicographic_initial_minimum_phase_reward) * 1000 / 3500)
    assert lexicographic_phase_reward_limit_at(3500, config) == pytest.approx(0.98)
    assert lexicographic_phase_reward_limit_at(5000, config) == pytest.approx(0.98)
    phase = np.array([[0.94, 0.99, 0.90]])
    errors = np.array([[20.0, 2.0, 2.0]])
    links = np.array([[2, 3, 1]])
    budgets = np.array([[20.0, 10.0, 5.0]])
    early_scores, early_feasible = lexicographic_candidate_scores(
        phase, errors, links, budgets, config,
        maximum_phase_error_deg=lexicographic_phase_error_limit_at(0, config),
        minimum_phase_reward=lexicographic_phase_reward_limit_at(0, config))
    final_scores, final_feasible = lexicographic_candidate_scores(
        phase, errors, links, budgets, config)
    np.testing.assert_array_equal(early_feasible, [[True, True, False]])
    np.testing.assert_array_equal(final_feasible, [[False, True, False]])
    assert early_scores.argmax(axis=1).tolist() == [0]
    assert final_scores.argmax(axis=1).tolist() == [1]


def test_lexicographic_budget_head_is_sampled_and_preserved_across_masks():
    config = replace(run_config, n_lasers=3, objective_mode="lexicographic")
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    budgets, log_prob = sample_budget_actions(policy, features, 2)
    assert budgets.shape == log_prob.shape == (1, 2)
    actions = torch.zeros((1, 2, 12))
    gates = torch.tensor([[[1., 1., 0., 0., 0., 0.],
                           [1., 1., 1., 1., 1., 1.]]])
    same_budget = torch.zeros((1, 2))
    kappa, _, counts, _, shortfall = decode_samples(
        actions, gates, config, budget_actions=same_budget)
    assert counts.tolist() == [[2, 6]]
    np.testing.assert_allclose(kappa.sum(axis=(-2, -1))[0],
                               [3 * np.log(2), 3 * np.log(2)])
    np.testing.assert_allclose(shortfall, 0.0)
    grad = torch.autograd.grad(log_prob.sum(), policy.budget_head[-1].weight)[0]
    assert torch.isfinite(grad).all()


def test_softplus_lexicographic_budget_transform():
    config = replace(run_config, n_lasers=3)
    actions = torch.zeros((1, 1, 12))
    gates = torch.ones((1, 1, 6))
    kappa = decode_samples(actions, gates, config,
                           budget_actions=torch.zeros((1, 1)))[0]
    assert kappa.sum() == pytest.approx(3 * config.softplus_strength_scale_per_ns
                                        * np.log(2))


def test_log_budget_initialization_and_scale_cancellation():
    config = replace(run_config, n_lasers=3, budget_action_transform="log_budget",
                     log_budget_initial_per_laser=1e-3)
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    mean, log_std = policy.budget_parameters(features)
    assert float(mean.exp().detach()[0]) == pytest.approx(0.003, rel=0.05)
    assert float(log_std.exp().detach()[0]) == pytest.approx(config.budget_exploration_end_std)
    actions = torch.zeros((1, 1, 12))
    gates = torch.tensor([[[1., 0., 1., 0., 0., 0.]]])
    log_budget = torch.tensor([[np.log(17.0)]], dtype=torch.float32)
    kappa_a = decode_samples(actions, gates, config, budget_actions=log_budget)[0]
    kappa_b = decode_samples(
        actions, gates, replace(config, softplus_strength_scale_per_ns=25.0),
        budget_actions=log_budget)[0]
    np.testing.assert_allclose(kappa_a, kappa_b, rtol=1e-6)
    assert kappa_a.sum() == pytest.approx(17.0, rel=1e-6)


def test_budget_exploration_covers_range_then_anneals():
    config = replace(run_config, budget_action_transform="log_budget",
                     budget_exploration_mode="stratified_offsets",
                     budget_exploration_start_factor=100.0,
                     budget_exploration_end_factor=1.2)
    width_early, std_early = budget_exploration_at(0, config)
    width_curriculum_end, _ = budget_exploration_at(
        config.budget_exploration_start_iteration, config)
    width_late, std_late = budget_exploration_at(
        config.budget_exploration_end_iteration, config)
    assert np.exp(width_early) == pytest.approx(100.0)
    assert width_curriculum_end == pytest.approx(width_early)
    assert np.exp(width_late) == pytest.approx(1.2)
    assert std_early == pytest.approx(config.budget_exploration_start_std)
    assert std_late == pytest.approx(config.budget_exploration_end_std)
    policy = BernoulliSparseGNNPolicy(replace(config, n_lasers=3))
    features = encode_context(np.zeros(3), np.array([-0.05, 0.0, 0.05]), policy)
    torch.manual_seed(7)
    actions, log_prob = sample_budget_actions(
        policy, features, 36, iteration=0, rng=np.random.default_rng(7))
    assert actions.shape == log_prob.shape == (1, 36)
    assert torch.isfinite(log_prob).all()
    assert float(actions.max() - actions.min()) > 7.0
    gradient = torch.autograd.grad(log_prob.sum(), policy.budget_head[-1].weight)[0]
    assert torch.isfinite(gradient).all()


def test_softplus_budget_samples_are_unbounded_and_train_head():
    config = replace(run_config, training_n_lasers=(3,), n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    features = encode_context(np.zeros(3), np.array([-0.005, 0.0, 0.005]), policy)
    _, log_std = policy.budget_parameters(features)
    assert float(log_std.detach()[0]) == pytest.approx(config.maximum_log_std)
    actions, log_prob = sample_budget_actions(
        policy, features, 36, iteration=0, rng=np.random.default_rng(8))
    budgets = 3 * config.softplus_strength_scale_per_ns * np.logaddexp(
        0.0, actions.detach().numpy())
    assert actions.shape == log_prob.shape == (1, 36)
    assert budgets.min() > 0
    assert budgets.max() > budgets.min()
    gradient = torch.autograd.grad(log_prob.sum(), policy.budget_head[-1].weight)[0]
    assert torch.isfinite(gradient).all()
    very_large_budget = decode_samples(
        torch.zeros((1, 1, 12)), torch.ones((1, 1, 6)), config,
        budget_actions=torch.full((1, 1), 100.0),
    )[0].sum()
    assert very_large_budget == pytest.approx(300.0)


def test_old_adaptive_checkpoint_still_loads_for_inference(tmp_path):
    config = replace(run_config, n_lasers=3, budget_action_transform="log_budget",
                     log_budget_initial_per_laser=1e-3)
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    path = tmp_path / "old_adaptive.pt"
    save_checkpoint(path, policy, optimizer, config, 7, 0.8)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"].pop("log_budget_initial_per_laser")
    checkpoint["config"].update(budget_exploration_mode="adaptive_frontier",
                                budget_search_floor_per_laser=1e-4,
                                budget_search_initial_ceiling_per_laser=1e-2,
                                budget_search_growth_factor=2.0,
                                budget_search_update_targets=8,
                                budget_search_span_bands=3,
                                budget_search_chaos_margin=0.05,
                                budget_search_target_feasible_fraction=0.8)
    torch.save(checkpoint, path)
    loaded_policy, loaded_config, metadata = load_checkpoint(path)
    assert loaded_config.log_budget_initial_per_laser == pytest.approx(1e-3)
    assert metadata["iteration"] == 7
    features = encode_context(np.zeros(3), np.array([-0.005, 0.0, 0.005]), loaded_policy)
    mean, _ = loaded_policy.budget_parameters(features)
    assert float(mean.exp().detach()[0]) == pytest.approx(0.003, rel=0.05)


def test_maximum_phase_error_wraps_at_pi():
    aligned = np.array([[[0.0, -np.pi + np.deg2rad(1.0)]]])
    target = np.array([[[0.0, np.pi]]])
    np.testing.assert_allclose(maximum_phase_error_deg(aligned, target), [[1.0]])


def test_lexicographic_training_batch_uses_aligned_phase_and_budget_head():
    config = replace(run_config, training_n_lasers=(3,), n_lasers=3,
                     objective_mode="lexicographic",
                     targets_per_batch=1, targets_per_worker_task=1,
                     candidates_per_target=3, candidates_per_worker_task=3,
                     coupling_budget_reference_by_m=((3, 50.0),))
    policy = BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    class FakePool:
        def imap_unordered(self, _worker, work):
            for group, candidate_start, candidate_stop, item in work:
                _, start, stop, targets, *_ = item
                target_relative = designer.make_relative_to_laser_1(targets)
                aligned = np.repeat(target_relative[:, None], candidate_stop - candidate_start, axis=1)
                yield dict(size_batch_index=group,
                           candidate_start_index=candidate_start,
                           candidate_stop_index=candidate_stop,
                           start_index=start, stop_index=stop,
                           phase_reward=np.full((stop - start, candidate_stop - candidate_start), 0.999),
                           aligned_achieved_phases_rad=aligned)

    metrics = run_training_batch(policy, optimizer, config, np.random.default_rng(12),
                                 1, simulation_pool=FakePool())
    assert metrics["feasible_candidate_fraction"] == pytest.approx(1.0)
    assert (0 < metrics["sampled_budget_p10_per_laser"]
            <= metrics["sampled_budget_median_per_laser"]
            <= metrics["sampled_budget_p90_per_laser"])
    assert metrics["selected_feasible_budget_median_per_laser"] > 0
    assert np.isfinite(metrics["gradient_norm"])


def test_targets_per_worker_task_must_be_positive():
    with pytest.raises(ValueError, match="targets_per_worker_task"):
        replace(run_config, targets_per_worker_task=0)
