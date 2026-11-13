"""Verify the prepared stratified RMS run without starting a training run."""

from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from rl import conditional_coupling_bernoulli_sparse as sparse
from rl import conditional_coupling_designer_variable_m as base
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import (
    MODEL_SUFFIX, config as run_config,
)


def phase_case(errors, *, targets=None, conjugate=False):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim == 1:
        errors = np.repeat(errors[:, None], 40, axis=1)
    size, steps = errors.shape
    config = replace(run_config, n_lasers=size, allow_global_phase_conjugate=conjugate)
    times = np.linspace(0, 500e-9, steps)
    states = np.zeros((1, 3 * size, steps))
    states[:, 1::3] = 1.
    states[:, 2::3] = errors
    target = np.zeros(size) if targets is None else np.asarray(targets)
    return times, states, target, config


def test_rms_reward_uses_mean_and_worst_nonreference_lasers():
    args = phase_case(np.deg2rad([0., 20., 40.]))
    metrics = base.calculate_phase_rms_metrics(*args)
    np.testing.assert_allclose(metrics["mean_phase_rms_error_deg"], 30.)
    np.testing.assert_allclose(metrics["worst_phase_rms_error_deg"], 40.)
    np.testing.assert_allclose(metrics["phase_rms_reward"], 1 - 35 / 180)
    assert not metrics["phase_rms_success"][0]


def test_rms_reward_does_not_hide_oscillations_behind_circular_mean():
    alternating = np.tile(np.deg2rad([30., -30.]), 20)
    args = phase_case(np.stack([np.zeros(40), alternating, alternating]))
    metrics = base.calculate_phase_rms_metrics(*args)
    np.testing.assert_allclose(metrics["worst_phase_rms_error_deg"], 30.)
    np.testing.assert_allclose(metrics["phase_rms_reward"], 1 - 30 / 180)


def test_rms_reward_wraps_errors_and_handles_global_conjugacy():
    target = np.array([0., .7, 1.2])
    args = phase_case(-target + 4 * np.pi, targets=target, conjugate=True)
    metrics = base.calculate_phase_rms_metrics(*args)
    np.testing.assert_allclose(metrics["phase_rms_reward"], 1., atol=1e-12)
    assert metrics["phase_rms_success"][0]
    assert metrics["rms_orientation"][0] == -1.
    # A separate sign per laser would incorrectly give this pattern reward 1.
    mixed = base.calculate_phase_rms_metrics(*phase_case(
        [0., -.7, 1.2], targets=target, conjugate=True))
    assert mixed["phase_rms_reward"][0] < .9
    assert not mixed["phase_rms_success"][0]


def test_rms_metrics_ignore_transients_and_reject_nonfinite_tail():
    times, states, target, config = phase_case([0., .01, -.01])
    states[:, 2::3, :10] = np.nan
    metrics = base.calculate_phase_rms_metrics(times, states, target, config)
    assert metrics["phase_rms_success"][0]
    states[:, 1, -1] = np.nan  # Invalid photon state, even with finite phases.
    metrics = base.calculate_phase_rms_metrics(times, states, target, config)
    assert metrics["phase_rms_reward"][0] == -1.
    assert np.isinf(metrics["worst_phase_rms_error_deg"][0])
    assert not metrics["phase_rms_success"][0]


def test_worker_vectorizes_five_contexts_and_preserves_rms_metrics(monkeypatch):
    config = replace(run_config, n_lasers=3, n_jobs=1)
    targets = np.array([[0., .1 * i, -.1 * i] for i in range(5)])
    detunings = np.array([[-.01 * (i + 1), 0., .01 * (i + 1)] for i in range(5)])
    kappa = np.ones((5, 10, 3, 3))
    phi = np.zeros_like(kappa)
    calls = []
    def simulate(kappa_flat, phi_flat, local, **kwargs):
        calls.append(kappa_flat.shape)
        np.testing.assert_allclose(kwargs["detuning_distribution_ghz"],
                                   np.repeat(detunings, 10, axis=0))
        times = np.linspace(0, 500e-9, 40)
        states = np.zeros((50, 9, 40))
        states[:, 1::3] = 1.
        states[:, 2::3] = np.repeat(targets, 10, axis=0)[:, :, None]
        return times, states, None, None
    monkeypatch.setattr(base, "simulate_with_vcsel", simulate)
    metrics = base.simulate_candidates_parallel(
        targets, kappa, phi, config, detuning_distributions_ghz=detunings)
    assert calls == [(50, 3, 3)]
    for key in base.PHASE_RMS_METRIC_KEYS:
        assert metrics[key].shape == (5, 10)
    np.testing.assert_allclose(metrics["phase_rms_reward"], 1.)
    assert metrics["phase_rms_success"].all()


@pytest.mark.parametrize("size", range(3, 10))
def test_density_strata_and_twenty_detuning_bins_cover_full_range_at_iteration_zero(size):
    counts_by_seed = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        masks = sparse.sample_stratified_rooted_topologies(
            1, 20, size, rng, torch.device("cpu"),
            min_fraction=sparse.topology_min_fraction_at(0, run_config),
            fixed_star=True, evenly_spaced=True)[0]
        counts = masks.sum(-1).numpy()
        expected = np.rint(np.linspace(size - 1, size * (size - 1), 20))
        np.testing.assert_array_equal(np.sort(counts), expected)
        groups = sparse.topology_stratum_group_ids(counts, 5)
        np.testing.assert_array_equal(np.bincount(groups), [4] * 5)
        _, sources = base.directed_link_indices(size)
        assert torch.all(masks[:, sources == 0] == 1)
        np.testing.assert_array_equal(sparse.root_reach_fractions(masks.numpy(), size), 1.)
        widths = np.ptp(base.sample_detuning_distributions(
            20, rng, replace(run_config, n_lasers=size), iteration=0), axis=1)
        np.testing.assert_array_equal(np.sort((widths / .2).astype(int)), np.arange(20))
        counts_by_seed.append(counts)
    assert not np.array_equal(*counts_by_seed)


def test_full_batch_has_28_vectorized_jobs_and_35_balanced_rms_tasks(monkeypatch):
    # Preserve coverage of the preceding independent-context run.
    config = replace(run_config, targets_per_batch=140, candidates_per_target=10,
                     topology_sampling_mode="stratified_contexts", pcgrad_topology_groups=5,
                     node_hidden_sizes=(8,), node_embedding_dim=8,
                     gnn_message_hidden_size=8, gnn_message_passing_steps=1,
                     edge_hidden_sizes=(16,),
                     coupling_budget_reference_by_m=tuple((m, float(m)) for m in range(3, 10)))
    torch.manual_seed(19)
    policy = sparse.BernoulliSparseGNNPolicy(config)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.)
    jobs = []
    class Pool:
        def imap_unordered(self, worker, work):
            for group, start, stop, item in reversed(work):
                _, ts, te, targets, kappa, phi, detuning, local = item
                assert (te - ts, stop - start) == (5, 10)
                assert np.all(np.count_nonzero(kappa, axis=(-2, -1)) ==
                              np.count_nonzero(kappa, axis=(-2, -1))[:, :1])
                jobs.append(dict(size=local.n_lasers, ts=ts, te=te,
                                 counts=np.count_nonzero(kappa, axis=(-2, -1))[:, 0],
                                 widths=np.ptp(detuning, axis=1)))
                shape = (5, 10)
                rms_reward = np.broadcast_to(.2 + .05 * np.arange(10), shape)
                yield dict(size_batch_index=group, candidate_start_index=start,
                    candidate_stop_index=stop, start_index=ts, stop_index=te,
                    phase_reward=np.full(shape, .95), phase_rms_reward=rms_reward,
                    mean_phase_rms_error_deg=180 * (1 - rms_reward),
                    worst_phase_rms_error_deg=180 * (1 - rms_reward),
                    phase_rms_success=np.zeros(shape, dtype=bool), rms_orientation=np.ones(shape))
    combine = base.combine_task_gradients_pcgrad
    def capture(gradients, clip, rng, **kwargs):
        assert gradients.shape[0] == 35
        np.testing.assert_allclose(kwargs["task_weights"], np.full(35, 1/35))
        return combine(gradients, clip, rng, **kwargs)
    monkeypatch.setattr(base, "combine_task_gradients_pcgrad", capture)
    monkeypatch.setattr(sparse, "phase_de_teacher_search", lambda _: pytest.fail("DE is disabled"))
    metrics = sparse.run_training_batch(policy, optimizer, config, np.random.default_rng(2),
                                         0, simulation_pool=Pool())
    assert len(jobs) == 28
    assert sum((j["te"] - j["ts"]) * 10 for j in jobs) == 1400
    assert metrics["pcgrad_task_count"] == 35
    assert metrics["pcgrad_task_context_counts"] == [4] * 35
    assert metrics["training_reward"] == pytest.approx(.425)
    assert metrics["phase_reward"] == pytest.approx(.95)
    assert metrics["gradient_norm"] > 0  # Cosine-only advantages would all be zero.
    assert metrics["rooted_fraction"] == 1.
    assert metrics["retained_link_fraction"] == pytest.approx(.5949263038548753)
    for size in config.training_n_lasers:
        local = sorted((j for j in jobs if j["size"] == size), key=lambda j: j["ts"])
        counts = np.concatenate([j["counts"] for j in local])
        np.testing.assert_array_equal(np.sort(counts), np.rint(np.linspace(size - 1, size*(size-1), 20)))
        widths = np.concatenate([j["widths"] for j in local])
        np.testing.assert_array_equal(np.sort((widths/.2).astype(int)), np.arange(20))


def test_new_checkpoint_round_trip_and_legacy_defaults(tmp_path):
    policy = sparse.BernoulliSparseGNNPolicy(run_config)
    path = tmp_path / "new.pt"
    sparse.save_checkpoint(path, policy, torch.optim.Adam(policy.parameters()), run_config, 1, .5)
    loaded, config, _ = sparse.load_checkpoint(path)
    assert asdict(config) == asdict(run_config)
    for first, second in zip(policy.parameters(), loaded.parameters()):
        torch.testing.assert_close(first, second)
    saved = torch.load(path, weights_only=False)
    for key in ("topology_sampling_mode", "phase_reward_mode", "pcgrad_topology_groups"):
        saved["config"].pop(key)
    saved["config"]["pcgrad_task_grouping"] = "array_size_topology_density"
    torch.save(saved, path)
    _, old, _ = sparse.load_checkpoint(path)
    assert old.phase_reward_mode == "cosine"
    assert old.topology_sampling_mode == "per_target"
    assert old.pcgrad_topology_groups == 3


def test_validation_reports_both_rewards_and_strict_rms_success(monkeypatch):
    config = replace(run_config, n_lasers=3, n_jobs=1,
                     coupling_budget_reference_by_m=((3, 1.),))
    policy = sparse.BernoulliSparseGNNPolicy(config)
    cosine = np.full((6, 1), .99)
    rms_reward = np.linspace(.5, .9, 6)[:, None]
    worst = np.array([1., 2., 4., 8., 10., 12.])[:, None]
    def simulate(*args, **kwargs):
        return dict(phase_reward=cosine, phase_rms_reward=rms_reward,
                    mean_phase_rms_error_deg=worst / 2,
                    worst_phase_rms_error_deg=worst,
                    phase_rms_success=worst <= 3.)
    monkeypatch.setattr(base, "simulate_candidates_parallel", simulate)
    result = sparse.evaluate_policy(policy, np.zeros((6, 3)),
        np.repeat([[-.5, 0., .5]], 6, axis=0), config, seed=3)
    assert result["phase_reward"] == pytest.approx(.99)
    assert result["training_reward"] == pytest.approx(.7)
    assert result["worst_phase_rms_error_deg"] == pytest.approx(worst.mean())
    assert result["phase_rms_success_fraction"] == pytest.approx(2/6)
    assert set(result["rms_reward_by_density"]) == set(sparse.DENSITY_BANDS)


def test_progress_plot_distinguishes_cosine_and_rms_rewards():
    history = dict(training_reward=[.3, .6], phase_reward=[.8, .9],
        retained_link_fraction=[.595, .595], held_out_reward=[.4, .7],
        held_out_phase_reward=[.85, .95], held_out_by_m={3: [.85, .95]})
    figure = sparse.training_figure(history, run_config)
    try:
        assert {line.get_label() for line in figure.axes[0].lines} >= {
            "cosine phase reward", "RMS training reward"}
        overall = next(line for line in figure.axes[1].lines if line.get_label() == "overall cosine")
        np.testing.assert_allclose(overall.get_ydata(), [.85, .95])
    finally:
        plt.close(figure)


def test_prepared_schedule_has_new_paths_and_shortened_annealing():
    assert MODEL_SUFFIX == "_star_rms_pairs5_topo4_wide180_3k"
    assert run_config.training_iterations == 3000
    assert run_config.n_jobs == 14
    assert run_config.targets_per_worker_task == 5
    assert run_config.targets_per_batch == 35
    assert run_config.candidates_per_target == 40
    assert run_config.actions_per_topology == 10
    assert run_config.topology_sampling_mode == "paired_contexts"
    assert run_config.pcgrad_topology_groups == 4
    assert run_config.candidates_per_worker_task == 10
    assert run_config.phase_reward_mode == "wrapped_rms"
    assert not run_config.de_teacher_enabled
    assert run_config.phase_exploration_fraction == pytest.approx(.2)
    assert run_config.phase_exploration_start_std == pytest.approx(np.deg2rad(180.))
    assert run_config.strength_exploration_fraction == 0
    assert run_config.topology_curriculum_start_iteration is None
    assert run_config.topology_curriculum_end_iteration is None
    assert run_config.detuning_curriculum_iterations == 0
    assert run_config.allow_global_phase_conjugate
    assert run_config.learning_rate_decay_start_iteration == 1999
    # Printed 999 is the last full-width iteration; both bounds first change
    # at 1000 and reach their final values exactly at printed 1500.
    assert base.minimum_log_std_at(998, run_config) == -1.5
    assert base.maximum_log_std_at(998, run_config) == .75
    for printed in (1000, 1250, 1499, 1500):
        progress = (printed - 999) / 501
        assert base.minimum_log_std_at(printed - 1, run_config) == pytest.approx(-1.5 - 1.5 * progress)
        assert base.maximum_log_std_at(printed - 1, run_config) == pytest.approx(.75 - 2.25 * progress)
        assert sparse.phase_exploration_at(printed - 1, run_config) == pytest.approx(
            np.deg2rad(180.) * .5 * (1 + np.cos(np.pi * progress)))
    assert base.minimum_log_std_at(1499, run_config) == -3.
    assert base.maximum_log_std_at(1499, run_config) == -1.5
    assert sparse.phase_exploration_at(998, run_config) == pytest.approx(np.deg2rad(180.))
    for iteration in (0, 998, 999, 1249, 1499, 1999, 2999):
        std = sparse._topology_action_std(torch.zeros((1, 10, 12)), run_config, iteration, False)
        torch.testing.assert_close(std[..., :6], torch.ones_like(std[..., :6]))
        torch.testing.assert_close(std[:, 2:, 6:], torch.ones_like(std[:, 2:, 6:]))
        torch.testing.assert_close(std[:, :2, 6:], torch.full_like(
            std[:, :2, 6:], np.sqrt(1 + sparse.phase_exploration_at(iteration, run_config)**2)))
        if iteration >= 1499:
            assert sparse.phase_exploration_at(iteration, run_config) == 0.
    for kind in ("current", "best", "final"):
        assert Path(getattr(run_config, f"{kind}_checkpoint_file")).stem.endswith(MODEL_SUFFIX)
