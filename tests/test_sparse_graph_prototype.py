"""Prototype acceptance: topology, policy gradients, compatibility and scaling."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.archive import conditional_coupling_sparse_graph_tolerance as graph
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_graph_scale_aware import make_config


def small_config(**changes):
    settings = dict(n_lasers=5, training_n_lasers=(3, 5), training_iterations=20,
        targets_per_batch=2, candidates_per_target=6, dense_replay_candidates=2,
        candidates_per_worker_task=6, n_jobs=1, held_out_target_count=1,
        node_hidden_sizes=(8,), node_embedding_dim=8, edge_hidden_sizes=(16,),
        gnn_message_hidden_size=8, gnn_message_passing_steps=1,
        detuning_curriculum_warmup_iterations=1, detuning_curriculum_iterations=4,
        sparse_stage_start_iteration=4, full_sparse_stage_start_iteration=8,
        refinement_stage_start_iteration=16, max_extra_edges_per_node=2,
        optional_candidates_per_node=4, simulation_time_seconds=5e-9,
        coupling_ramp_start_delays=0.1, coupling_ramp_rise_delays=0.5)
    settings.update(changes)
    return replace(make_config(), **settings)


def infer(policy, m, q=None):
    return graph.infer_sparse_graph(policy, np.arange(m)*2*np.pi/m,
        np.linspace(-0.05, 0.05, m), 20, q_override=q)


@pytest.mark.parametrize("m", [2, 3, 5, 20, 1000])
def test_exact_budget_backbone_connectivity_and_caps(m):
    policy = graph.SparseGraphPolicy(small_config())
    for q in [0.0, 0.1, 0.5, 1.0]:
        design = infer(policy, m, q)
        r, s = design["receiver_indices"], design["source_indices"]
        # The entire forward chain remains present, ensuring a directed root
        # reaches every node as well as weak connectivity.
        np.testing.assert_array_equal(s[:m-1], np.arange(m-1))
        np.testing.assert_array_equal(r[:m-1], np.arange(1, m))
        assert len(set(zip(r, s))) == len(r)
        assert np.all(r != s)
        maximum = design["sparse_edge_budget"]
        assert len(r) == m-1+int(np.floor(q*(maximum-(m-1))+0.5))
        assert np.bincount(r, minlength=m).max() <= policy.config.max_extra_edges_per_node+1
        assert np.all(design["kappa_per_ns"] > 0)
        assert np.all(np.abs(design["phi_p_rad"]) <= np.pi+1e-6)
        incoming = np.bincount(r, weights=design["kappa_per_ns"], minlength=m)
        assert incoming.max() <= policy.config.maximum_kappa_per_ns+1e-6


def test_candidate_construction_is_bounded_and_deterministic():
    for m in (2, 3, 5, 1000):
        candidates = graph.candidate_sources(m, 8)
        assert candidates.shape == (m, min(m-1, 8))
        for r, row in enumerate(candidates):
            valid = row[row >= 0]
            assert len(set(valid)) == len(valid)
            assert len(valid) == min(8, m-1-(r > 0))
            assert r not in valid
            assert r == 0 or r-1 not in valid
        np.testing.assert_array_equal(candidates, graph.candidate_sources(m, 8))


def test_automatic_and_manual_q_match_and_original_indices_survive():
    policy = graph.SparseGraphPolicy(small_config())
    first = infer(policy, 7)
    torch.manual_seed(42)
    second = infer(policy, 7, first["q"])
    for key in ("receiver_indices", "source_indices", "kappa_per_ns", "phi_p_rad"):
        np.testing.assert_array_equal(first[key], second[key])
    target, detuning = np.array([1.0, 2.0, 3.0]), np.array([5.0, -1.0, 2.0])
    sorted_design = graph.infer_sparse_graph(policy, target, detuning, 20, q_override=0)
    np.testing.assert_array_equal(sorted_design["sorted_to_original"], [1, 2, 0])
    np.testing.assert_array_equal(sorted_design["target_phases_rad"], [2, 3, 1])
    np.testing.assert_allclose(sorted_design["detuning_distribution_ghz"], [-3, 0, 3])


class AllocationAudit(TorchDispatchMode):
    def __init__(self, m):
        super().__init__()
        self.m, self.maximum_numel = m, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if isinstance(value, torch.Tensor):
                assert tuple(value.shape).count(self.m) < 2, (func, value.shape)
                assert self.m*(self.m-1) not in value.shape, (func, value.shape)
                self.maximum_numel = max(self.maximum_numel, value.numel())
        return output


def test_m1000_has_no_all_pairs_tensors_and_allocations_scale_linearly(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense policy/simulation helper was called")
    monkeypatch.setattr(graph.base, "directed_link_indices", forbidden)
    monkeypatch.setattr(graph, "dense_matrices", forbidden)
    policy = graph.SparseGraphPolicy(small_config())
    maxima = []
    for m in (500, 1000):
        with AllocationAudit(m) as audit:
            for q in (0, 1, None):
                infer(policy, m, q)
        maxima.append(audit.maximum_numel)
    assert maxima[1] <= 2.1*maxima[0]


def test_dense_adapter_orientation_and_large_m_guard():
    design = infer(graph.SparseGraphPolicy(small_config()), 5, 0)
    kappa, phi = graph.dense_matrices(design, 5)
    np.testing.assert_array_equal(kappa[design["receiver_indices"], design["source_indices"]], design["kappa_per_ns"])
    np.testing.assert_array_equal(phi[design["receiver_indices"], design["source_indices"]], design["phi_p_rad"])
    assert np.count_nonzero(kappa) == 4
    with pytest.raises(ValueError, match="100"):
        graph.dense_matrices({}, 1000)


def test_dense_adapter_uses_identical_existing_physics():
    config = small_config(n_lasers=3, noise_amplitude=0)
    design = infer(graph.SparseGraphPolicy(config), 3, 0.5)
    targets, detunings = design["target_phases_rad"][None], design["detuning_distribution_ghz"][None]
    kappa, phi = graph.dense_matrices(design, 3)
    actual = graph.simulate_graph_batch(targets, detunings, [[design]], config)
    reference = graph.base.simulate_candidates_parallel(targets, kappa[None, None], phi[None, None],
        config, detuning_distributions_ghz=detunings)
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-12)


def test_stochastic_joint_log_probability_trains_all_heads():
    torch.manual_seed(13)
    config = small_config(max_extra_edges_per_node=2, optional_candidates_per_node=4)
    policy = graph.SparseGraphPolicy(config)
    features = graph.legacy.encode_sparse_tolerance_context(np.zeros(5),
        np.linspace(-1, 1, 5), 20, policy, config)[0]
    draw = policy.draw(policy.encode(features), minimum_q=0.7)
    assert torch.isfinite(draw["latent_entropy"])
    (-draw["log_probability"]).backward()
    for head in (policy.node_encoder, policy.q_head, policy.topology_head, policy.edge_head):
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    assert torch.isfinite(policy.log_std.grad).all()
    assert torch.isfinite(policy.q_log_std.grad)


@pytest.mark.parametrize("iteration", [0, 5, 10, 18])
def test_small_physical_training_step_each_stage(iteration):
    torch.manual_seed(7)
    config = small_config()
    policy = graph.SparseGraphPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    before = policy.edge_head[-1].weight.detach().clone()
    metrics = graph.train_step(policy, optimizer, np.random.default_rng(7), iteration)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, policy.edge_head[-1].weight)
    assert metrics["q"] == 1 if iteration == 0 else metrics["q"] < 1


def test_checkpoint_roundtrip_and_family_guard(tmp_path):
    config = small_config()
    policy = graph.SparseGraphPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    filename = tmp_path / "graph.pt"
    graph.save_checkpoint(filename, policy, optimizer, iteration=3, metrics={"phase_reward": 0.5})
    restored, _, metadata = graph.load_checkpoint(filename)
    assert metadata["iteration"] == 3
    for key in ("source_indices", "receiver_indices", "kappa_per_ns", "phi_p_rad"):
        np.testing.assert_array_equal(infer(policy, 5)[key], infer(restored, 5)[key])
    torch.save({"format": graph.legacy.SPARSE_TOLERANCE_CHECKPOINT_FORMAT}, filename)
    with pytest.raises(ValueError, match="incompatible"):
        graph.load_checkpoint(filename)


def test_next_run_schedule_and_paths_are_separate():
    config = make_config()
    assert (config.detuning_curriculum_warmup_iterations, config.detuning_curriculum_iterations,
            config.sparse_stage_start_iteration) == (100, 600, 1100)
    assert config.detuning_span_ghz == 10
    for kind in ("best", "current", "final"):
        assert "sparse_graph_prototype" in getattr(config, f"{kind}_checkpoint_file")


def test_training_loop_saves_new_family_and_progress(tmp_path):
    config = small_config(training_iterations=2,
        detuning_curriculum_warmup_iterations=0, detuning_curriculum_iterations=0,
        sparse_stage_start_iteration=0, full_sparse_stage_start_iteration=0,
        refinement_stage_start_iteration=0, validation_interval=1,
        **{f"{kind}_checkpoint_file": str(tmp_path / f"{kind}.pt")
           for kind in ("best", "current", "final")})
    policy, _, history = graph.train(graph.SparseGraphPolicy(config),
                                    progress_file=tmp_path / "progress.json")
    assert len(history) == 2
    for kind in ("best", "current", "final"):
        loaded, _, metadata = graph.load_checkpoint(tmp_path / f"{kind}.pt")
        assert 1 <= metadata["iteration"] <= 2
    assert metadata["iteration"] == 2
    np.testing.assert_array_equal(infer(loaded, 5)["kappa_per_ns"], infer(policy, 5)["kappa_per_ns"])
    assert (tmp_path / "progress.json").is_file()


def test_large_m_inspector_skips_simulator_and_uses_only_small_plot_bins(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from rl.archive import inspect_saved_coupling_model_variable_m_sparse_graph_scale_aware as inspect
    config = small_config()
    policy = graph.SparseGraphPolicy(config)
    filename = tmp_path / "graph.pt"
    graph.save_checkpoint(filename, policy, torch.optim.Adam(policy.parameters()), iteration=0, metrics={})
    def forbidden(*args, **kwargs):
        raise AssertionError("Large-M inspector must not densify")
    monkeypatch.setattr(inspect, "dense_matrices", forbidden)
    monkeypatch.setattr(plt, "show", lambda: None)
    with plt.rc_context({"text.usetex": False}):
        result = inspect.main(filename, n_lasers=1000, run_validation=True)
        plt.gcf().canvas.draw()
    assert result["simulation"] is None
    plt.close("all")
