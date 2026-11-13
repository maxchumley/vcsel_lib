"""Shared pairs must align density tasks without increasing physical work."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl import conditional_coupling_bernoulli_sparse as sparse
from rl import conditional_coupling_designer_variable_m as base
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import config as run_config


@pytest.mark.parametrize("size", range(3, 10))
def test_paired_sampler_spans_all_densities_with_random_assignments(size):
    draws = []
    for seed in (7, 8):
        rng = np.random.default_rng(seed)
        masks = sparse.sample_paired_rooted_topologies(
            5, 4, size, rng, torch.device("cpu"), fixed_star=True)
        assert masks.shape == (5, 4, size * (size - 1))
        counts = masks.sum(-1).numpy()
        expected = np.rint(np.linspace(size - 1, size * (size - 1), 20)).reshape(4, 5)
        for group in range(4):
            np.testing.assert_array_equal(np.sort(counts[:, group]), expected[group])
        _, sources = base.directed_link_indices(size)
        assert masks[..., sources == 0].all()
        np.testing.assert_array_equal(sparse.root_reach_fractions(masks.numpy(), size), 1.)
        widths = np.ptp(base.sample_detuning_distributions(
            5, rng, replace(run_config, n_lasers=size), iteration=0), axis=1)
        np.testing.assert_array_equal(np.sort((widths / .8).astype(int)), np.arange(5))
        draws.append(masks)
    assert not torch.equal(*draws)
    repeated = sparse.sample_paired_rooted_topologies(
        5, 4, size, np.random.default_rng(7), torch.device("cpu"), fixed_star=True)
    torch.testing.assert_close(draws[0], repeated)


def test_paired_batch_shares_five_pairs_across_28_balanced_tasks(monkeypatch):
    config = replace(run_config, node_hidden_sizes=(8,), node_embedding_dim=8,
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
                assert (ts, te, stop - start) == (0, 5, 10)
                counts = np.count_nonzero(kappa, axis=(-2, -1))
                np.testing.assert_array_equal(counts, np.repeat(counts[:, :1], 10, axis=1))
                assert np.all(kappa[:, :, 1:, 0] > 0)
                jobs.append(dict(size=local.n_lasers, band=start // 10,
                                 targets=targets.copy(), detuning=detuning.copy(),
                                 counts=counts[:, 0]))
                shape = (5, 10)
                rms = np.broadcast_to(.2 + .05 * np.arange(10), shape)
                yield dict(size_batch_index=group, candidate_start_index=start,
                    candidate_stop_index=stop, start_index=ts, stop_index=te,
                    phase_reward=np.full(shape, .95), phase_rms_reward=rms,
                    mean_phase_rms_error_deg=180 * (1 - rms),
                    worst_phase_rms_error_deg=180 * (1 - rms),
                    phase_rms_success=np.zeros(shape, dtype=bool), rms_orientation=np.ones(shape))

    combine = base.combine_task_gradients_pcgrad
    def capture(gradients, clip, rng, **kwargs):
        assert gradients.shape[0] == 28
        np.testing.assert_allclose(kwargs["task_weights"], np.full(28, 1 / 28))
        return combine(gradients, clip, rng, **kwargs)
    monkeypatch.setattr(base, "combine_task_gradients_pcgrad", capture)
    monkeypatch.setattr(sparse, "phase_de_teacher_search", lambda _: pytest.fail("DE is disabled"))
    metrics = sparse.run_training_batch(policy, optimizer, config, np.random.default_rng(2),
                                        0, simulation_pool=Pool())
    assert len(jobs) == 28
    assert len(jobs) * 50 == 1400
    assert metrics["pcgrad_task_count"] == 28
    assert metrics["pcgrad_task_context_counts"] == [5] * 28
    assert len(set(metrics["pcgrad_task_labels"])) == 28
    assert metrics["training_reward"] == pytest.approx(.425)
    assert metrics["phase_reward"] == pytest.approx(.95)
    assert metrics["gradient_norm"] > 0
    for size in config.training_n_lasers:
        local = sorted((j for j in jobs if j["size"] == size), key=lambda j: j["band"])
        counts = np.concatenate([j["counts"] for j in local])
        np.testing.assert_array_equal(np.sort(counts), np.rint(np.linspace(size - 1, size*(size-1), 20)))
        for j in local:
            np.testing.assert_array_equal(j["targets"], local[0]["targets"])
            np.testing.assert_array_equal(j["detuning"], local[0]["detuning"])
            bins = (np.ptp(j["detuning"], axis=1) / .8).astype(int)
            np.testing.assert_array_equal(np.sort(bins), np.arange(5))


def test_wide_actions_repeat_twice_in_each_ten_action_mask():
    std = sparse._topology_action_std(torch.zeros((5, 40, 12)), run_config, 0, False)
    torch.testing.assert_close(std[..., :6], torch.ones_like(std[..., :6]))
    for slot in range(40):
        expected = np.sqrt(1 + np.pi**2) if slot % 10 < 2 else 1.
        torch.testing.assert_close(std[:, slot, 6:], torch.full_like(std[:, slot, 6:], expected))
    reward = np.tile(np.repeat([.1, .3, .6, .8], 10) + np.tile(np.arange(10)*.01, 4), (5, 1))
    advantage = sparse._topology_advantages([reward], run_config)[0].reshape(5, 4, 10)
    np.testing.assert_allclose(advantage.mean(-1), 0., atol=1e-14)


def test_paired_config_rejects_topology_count_mismatches():
    with pytest.raises(ValueError, match="one topology per density group"):
        replace(run_config, candidates_per_target=30)
    with pytest.raises(ValueError, match="equal pairs per size"):
        replace(run_config, targets_per_batch=36)
