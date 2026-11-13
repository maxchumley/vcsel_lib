"""Learned-tree topology, scaling, gradients and unchanged dense-physics adapter."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.archive import conditional_coupling_learned_tree_tolerance as tree
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_learned_tree_scale_aware import make_config


def small_config(**changes):
    settings = dict(n_lasers=5, training_n_lasers=(3, 5), training_iterations=20,
        targets_per_batch=2, candidates_per_target=6, dense_replay_candidates=2,
        candidates_per_worker_task=6, n_jobs=1, held_out_target_count=1,
        node_hidden_sizes=(8,), node_embedding_dim=8, edge_hidden_sizes=(16,),
        gnn_message_hidden_size=8, gnn_message_passing_steps=1,
        detuning_curriculum_warmup_iterations=1, detuning_curriculum_iterations=4,
        sparse_stage_start_iteration=4, full_sparse_stage_start_iteration=8,
        refinement_stage_start_iteration=16, simulation_time_seconds=5e-9,
        coupling_ramp_start_delays=0.1, coupling_ramp_rise_delays=0.5)
    settings.update(changes)
    return replace(make_config(), **settings)


def infer(policy, m, q):
    return tree.infer_learned_tree(policy, np.arange(m) * 2 * np.pi / m,
        np.linspace(-0.05, 0.05, m), 20.0, q_override=q)


def connected(m, sources, receivers):
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for source, receiver in zip(sources, receivers):
        parent[find(source)] = find(receiver)
    return len({find(i) for i in range(m)}) == 1


@pytest.mark.parametrize("m", [3, 10, 1000])
@pytest.mark.parametrize("q", [0.0, 0.5, 0.8, 1.0])
def test_exact_count_unique_edges_and_learned_backbone(m, q):
    if m == 1000 and q == 1.0:
        pytest.skip("All-to-all output is intentionally dense; its count is checked separately")
    policy = tree.LearnedTreePolicy(small_config())
    design = infer(policy, m, q)
    s, r = design["source_indices"], design["receiver_indices"]
    assert len(s) == tree.q_to_edge_count(q, m)
    assert len(set(zip(s, r))) == len(s)
    assert np.all(s != r)
    assert connected(m, s[:m - 1], r[:m - 1]) if q < 1 else connected(m, s, r)
    incoming = np.bincount(r, weights=design["kappa_per_ns"], minlength=m)
    assert incoming.max() <= policy.config.maximum_kappa_per_ns + 1e-6


def test_q_count_scaling_and_schedule():
    assert tree.q_to_edge_count(0, 1000) == 999
    assert tree.q_to_edge_count(0.5, 1000) == 1996
    assert tree.q_to_edge_count(1, 1000) == 999000
    config = make_config()
    assert config.detuning_span_ghz == 4.0
    assert (config.detuning_curriculum_warmup_iterations,
            config.detuning_curriculum_iterations,
            config.sparse_stage_start_iteration) == (100, 600, 1100)


def test_deterministic_replay_and_root_is_context_dependent():
    policy = tree.LearnedTreePolicy(small_config())
    design = infer(policy, 9, 0.5)
    repeated = infer(policy, 9, 0.5)
    for key in ("source_indices", "receiver_indices", "kappa_per_ns", "phi_p_rad"):
        np.testing.assert_array_equal(design[key], repeated[key])
    assert 0 <= design["root_index"] < 9


class AllocationAudit(TorchDispatchMode):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if isinstance(value, torch.Tensor):
                assert tuple(value.shape).count(self.m) < 2, (func, value.shape)
                assert self.m * (self.m - 1) not in value.shape, (func, value.shape)
        return output


def test_m1000_sparse_inference_allocates_no_pair_grid():
    policy = tree.LearnedTreePolicy(small_config())
    with AllocationAudit(1000):
        infer(policy, 1000, 0.5)


def test_stochastic_log_probability_reaches_topology_and_value_heads():
    torch.manual_seed(5)
    config = small_config()
    policy = tree.LearnedTreePolicy(config)
    features = tree.legacy.encode_sparse_tolerance_context(
        np.zeros(5), np.linspace(-1, 1, 5), 20.0, policy, config)[0]
    draw = policy.draw(policy.encode(features), q_override=0.7)
    assert torch.isfinite(draw["log_probability"])
    (-draw["log_probability"]).backward()
    for head in (policy.root_head, policy.join_head, policy.parent_head,
                 policy.receiver_head, policy.source_head, policy.direction_head,
                 policy.edge_head):
        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_reused_training_step_and_dense_simulator():
    config = small_config()
    policy = tree.LearnedTreePolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    for iteration in (0, 9):
        metrics = tree.train_step(policy, optimizer, np.random.default_rng(7), iteration)
        assert all(np.isfinite(value) for value in metrics.values())


def test_all_sizes_are_submitted_to_pool_together():
    class RecordingPool:
        batches = []

        def imap_unordered(self, function, work_items):
            self.batches.append(list(work_items))
            return map(function, self.batches[-1])

    config = small_config(n_jobs=4, targets_per_batch=4)
    policy = tree.LearnedTreePolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    pool = RecordingPool()
    metrics = tree.train_step(policy, optimizer, np.random.default_rng(7), 0,
                              simulation_pool=pool)
    assert np.isfinite(metrics["reward"])
    assert len(pool.batches) == 1
    assert len(pool.batches[0]) == config.targets_per_batch
    assert {item[3][-1].n_lasers for item in pool.batches[0]} == {3, 5}


def test_checkpoint_roundtrip(tmp_path):
    config = small_config()
    policy = tree.LearnedTreePolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    path = tmp_path / "learned_tree.pt"
    tree.save_checkpoint(path, policy, optimizer, iteration=3, metrics={"reward": 0.5})
    loaded, _, metadata = tree.load_checkpoint(path)
    assert metadata["iteration"] == 3
    np.testing.assert_array_equal(infer(policy, 5, 0.5)["kappa_per_ns"],
                                  infer(loaded, 5, 0.5)["kappa_per_ns"])


def test_training_entry_point_saves_separate_checkpoints(tmp_path):
    config = small_config(training_iterations=2,
        detuning_curriculum_warmup_iterations=0, detuning_curriculum_iterations=0,
        sparse_stage_start_iteration=0, full_sparse_stage_start_iteration=0,
        refinement_stage_start_iteration=0, validation_interval=1,
        **{f"{kind}_checkpoint_file": str(tmp_path / f"{kind}.pt")
           for kind in ("best", "current", "final")})
    policy, _, history = tree.train(tree.LearnedTreePolicy(config),
                                    figure_directory=tmp_path)
    assert len(history) == 2
    assert not list(tmp_path.glob("*.json"))
    assert (tmp_path / "current_training_progress.png").is_file()
    for kind in ("best", "current", "final"):
        restored, _, metadata = tree.load_checkpoint(tmp_path / f"{kind}.pt")
        assert 1 <= metadata["iteration"] <= 2
        assert isinstance(restored, tree.LearnedTreePolicy)


def test_inspector_inference_only_path(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from rl.archive import inspect_saved_coupling_model_variable_m_learned_tree_scale_aware as inspect
    config = small_config()
    policy = tree.LearnedTreePolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    checkpoint = tmp_path / "inspect.pt"
    tree.save_checkpoint(checkpoint, policy, optimizer, iteration=4, metrics={})
    monkeypatch.setattr(inspect, "INFERENCE_DEVICE", "cpu")
    monkeypatch.setattr(inspect, "FIGURE_DIRECTORY", tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    result = inspect.main(checkpoint, n_lasers=5, run_validation=False)
    assert result["simulation"] is None
    assert result["design"]["root_index"] in range(5)
    assert len(list(tmp_path.glob("inferred_summary_*.png"))) == 1
    plt.close("all")


@pytest.mark.parametrize("n_lasers", [5, 101])
def test_inspector_uses_standard_four_panel_and_dynamics_plots(
    tmp_path, monkeypatch, n_lasers,
):
    import matplotlib.pyplot as plt
    from rl.archive import inspect_saved_coupling_model_variable_m_learned_tree_scale_aware as inspect
    config = small_config()
    policy = tree.LearnedTreePolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    checkpoint = tmp_path / "inspect.pt"
    tree.save_checkpoint(checkpoint, policy, optimizer, iteration=4, metrics={})
    seen = {}

    trajectory = (np.array([0.0, 1.0]), np.zeros((1, 3 * n_lasers, 2)),
                  np.zeros((1, n_lasers, 2)), np.zeros((1, n_lasers)))

    def fake_simulation(kappa, phi, run_config, **kwargs):
        assert run_config.n_lasers == n_lasers
        assert kappa.shape == phi.shape == (1, n_lasers, n_lasers)
        assert kwargs["progress"] is True
        seen["simulation_calls"] = seen.get("simulation_calls", 0) + 1
        return trajectory

    def fake_reward(time, states, targets, run_config):
        assert time is trajectory[0]
        assert states is trajectory[1]
        return (np.ones(1), None, None, None, targets, None)

    def fake_four_panel(design, run_config, **kwargs):
        seen["design"] = design
        seen["plot_options"] = kwargs
        figure, _ = plt.subplots(1, 4)
        return figure

    def fake_dynamics(design, run_config, **kwargs):
        seen["dynamics_options"] = kwargs
        return {"plotted": True}

    monkeypatch.setattr(inspect, "INFERENCE_DEVICE", "cpu")
    monkeypatch.setattr(inspect, "FIGURE_DIRECTORY", tmp_path)
    monkeypatch.setattr(inspect, "simulate_with_vcsel", fake_simulation)
    monkeypatch.setattr(inspect, "calculate_phase_reward", fake_reward)
    monkeypatch.setattr(inspect, "plot_selected_design", fake_four_panel)
    monkeypatch.setattr(inspect, "simulate_and_plot_best_design", fake_dynamics)
    monkeypatch.setattr(plt, "show", lambda: None)
    result = inspect.main(checkpoint, n_lasers=n_lasers, run_validation=True)
    assert result["simulation"] == {"plotted": True}
    assert seen["design"]["kappa_per_ns"].shape == (n_lasers, n_lasers)
    assert seen["design"]["aligned_achieved_phases_rad"].shape == (n_lasers,)
    assert seen["plot_options"]["topology_detuning_arc_layout"] is True
    assert seen["plot_options"]["disabled_links_black"] is True
    assert seen["dynamics_options"]["far_field_panel"] is True
    assert seen["dynamics_options"]["precomputed_simulation"] is trajectory
    assert seen["simulation_calls"] == 1
    assert list(tmp_path.glob("selected_design_learned_tree_*.png"))
    plt.close("all")


def test_notebook_entry_point_uses_configured_action(monkeypatch):
    from rl.archive import conditional_coupling_run_variable_m_gnn_pcgrad_learned_tree_scale_aware as run
    calls = []
    monkeypatch.setattr(run, "_running_in_notebook", lambda: True)
    monkeypatch.setattr(run, "NOTEBOOK_ACTION", "train")
    monkeypatch.setattr(run, "train", lambda policy: calls.append(policy))
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "kernel.json"])
    run.main()
    assert len(calls) == 1
    assert isinstance(calls[0], tree.LearnedTreePolicy)


def test_progress_figure_is_displayed_inline_in_interactive_kernel(tmp_path, monkeypatch):
    import IPython
    import IPython.display
    config = small_config(jupyter_mode=True)
    shown = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "clear_output", lambda **kwargs: shown.append("clear"))
    monkeypatch.setattr(IPython.display, "display", lambda figure: shown.append("figure"))
    filename = tree.save_training_figure([
        dict(iteration=1, reward=0.7, q=1.0, max_budget_phase=0.8)
    ], config, tmp_path)
    assert filename.is_file()
    assert shown == ["clear", "figure"]


def test_progress_curves_are_lines_without_point_markers():
    import matplotlib.pyplot as plt
    config = small_config()
    figure = tree.update_training_figure([
        dict(iteration=1, reward=0.7, q=1.0, max_budget_phase=0.8),
        dict(iteration=2, reward=0.8, q=0.9, max_budget_phase=0.85),
    ], config)
    assert all(line.get_marker() == "None" for axis in figure.axes for line in axis.lines)
    plt.close(figure)
