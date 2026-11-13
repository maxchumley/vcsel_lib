"""Regression checks for the topology-inspection trajectory summary and plot."""

from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from rl import inspect_saved_coupling_model_variable_m_bernoulli_sparse as inspector
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import (
    config as training_config,
)
from rl.conditional_coupling_bernoulli_sparse import (
    BernoulliSparseGNNPolicy, infer_designs,
)
from rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse import (
    _resolve_topology_input_override,
    plot_topology_ensemble,
    select_ensemble_topology,
    summarize_topology_trajectories,
)


@pytest.mark.parametrize("mode,expected", [("best", 0), ("sparsest", 2), ("min_budget", 3)])
def test_ensemble_selection_balances_reward_and_link_count(mode, expected):
    selected = select_ensemble_topology(
        np.array([.99, .85, .92, .97]), np.array([5, 2, 2, 4]), mode,
        coupling_budgets=np.array([7., 6., 8., 2.]))
    assert selected == expected


@pytest.mark.parametrize("mode", ["best", "sparsest", "min_budget"])
def test_ensemble_selection_excludes_invalid_simulations(mode):
    selected = select_ensemble_topology(
        np.array([np.nan, 1.0, .9, .8]), np.array([1, 1, 2, 2]), mode,
        valid_topologies=np.array([True, False, True, True]),
        coupling_budgets=np.array([1., 1., 3., 4.]))
    assert selected == 2
    with pytest.raises(ValueError, match="No valid ensemble topology"):
        select_ensemble_topology(np.array([.9]), np.array([2]), mode,
                                 valid_topologies=np.array([False]),
                                 coupling_budgets=np.array([1.]))


def test_min_budget_selection_filters_nonfinite_budgets_and_breaks_ties():
    selected = select_ensemble_topology(
        np.array([1., .9, .9, .8]), np.array([1, 5, 2, 1]), "min_budget",
        coupling_budgets=np.array([np.nan, 2., 2., 2.]))
    assert selected == 2  # Equal budgets: higher reward, then fewer links.
    with pytest.raises(ValueError, match="requires coupling budgets"):
        select_ensemble_topology(np.array([.9]), np.array([2]), "min_budget")
    with pytest.raises(ValueError, match="budgets must match"):
        select_ensemble_topology(np.array([.9]), np.array([2]), "min_budget",
                                 coupling_budgets=np.array([1., 2.]))
    with pytest.raises(ValueError, match="No valid ensemble topology"):
        select_ensemble_topology(np.array([.9]), np.array([2]), "min_budget",
                                 coupling_budgets=np.array([np.inf]))


@pytest.mark.parametrize("mode,expected", [("best", 1), ("sparsest", 0), ("min_budget", 2)])
def test_inspector_reuses_selected_ensemble_simulation(monkeypatch, tmp_path, mode, expected):
    checkpoint = tmp_path / f"checkpoint{inspector.TRAINING_MODEL_SUFFIX}.pt"
    checkpoint.touch()
    config = replace(training_config, n_lasers=3, current_checkpoint_file=checkpoint,
                     coupling_budget_reference_by_m=((3, 6.),))
    settings = dict(
        CHECKPOINT_KIND="current", MODEL_FILE_SUFFIX=inspector.TRAINING_MODEL_SUFFIX,
        training_config=config, N_LASERS=3, MANUAL_TARGET_PHASES_PI=[0., 0., 0.],
        DETUNING_MODE="manual", MANUAL_DETUNINGS_GHZ=[-.1, 0., .1],
        TOPOLOGY_MATRIX_SOURCE_RECEIVER=None, TARGET_ACTIVE_FRACTION=None,
        RUN_VALIDATION=True, PLOT_TOPOLOGY_ENSEMBLE=True,
        ENSEMBLE_SELECTED_TOPOLOGY=mode, ENSEMBLE_TRAJECTORY_MODE="mean_sd",
        NUMBER_OF_CANDIDATES=3, NOISE_ITERATIONS=2, PRINT_GATE_LOGITS=False,
        INFERENCE_DEVICE="cpu", OVERRIDE_COUPLING_RAMP=False,
        FIGURE_DIRECTORY=tmp_path, PLOT_STYLE={"legend.fontsize": 12, "text.usetex": False},
    )
    for key, value in settings.items():
        monkeypatch.setattr(inspector, key, value)
    monkeypatch.setattr(inspector.coupling_plots, "figure_directory", tmp_path, raising=False)
    monkeypatch.setattr(inspector, "load_checkpoint", lambda *args, **kwargs:
                        (torch.nn.Linear(1, 1), config, {"iteration": 42}))
    kappas = np.zeros((3, 3, 3))
    kappas[:, 1:, 0] = 1.
    kappas[1:, 1, 2] = 1.
    kappas[2, 2, 1] = 1.
    kappas[2] *= .1  # Lowest budget despite more links and worse phase reward.
    counts = np.count_nonzero(kappas, axis=(1, 2))
    receivers, sources = inspector.designer.directed_link_indices(3)
    inference = dict(
        kappa_per_ns=kappas, phi_p_rad=np.zeros_like(kappas),
        target_phases_rad=np.zeros(3), detuning_distribution_ghz=np.array([-.1, 0., .1]),
        active_link_counts=counts, retained_link_fractions=counts / 6,
        gates=(kappas[:, receivers, sources] > 0).astype(float),
        expected_active_fraction=float(np.mean(counts / 6)),
        raw_total_coupling_per_ns=kappas.sum(axis=(1, 2)),
        redistribution_shortfall_per_ns=np.zeros(3),
    )
    monkeypatch.setattr(inspector, "infer_designs", lambda *args, **kwargs: inference)
    times = np.linspace(0, 500e-9, 40)
    states = np.zeros((3, 9, len(times)))
    states[:, 1::3] = 1.
    states[:, 5::3] = np.deg2rad([10., 0., 30.])[:, None, None]
    states = np.repeat(states, 2, axis=0)
    simulation = (times, states, np.zeros((6, 3, len(times))),
                  np.zeros((6, 3, len(times) * config.save_every)))
    simulation_calls = []

    def simulate(*args, **kwargs):
        simulation_calls.append(True)
        return simulation

    monkeypatch.setattr(inspector, "simulate_with_vcsel", simulate)
    figure = SimpleNamespace(savefig=lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector, "plot_selected_design", lambda *args, **kwargs: figure)
    ensemble_counts = []

    def plot_ensemble(summary, *args, **kwargs):
        ensemble_counts.append(summary["n_valid_topologies"])
        return figure

    monkeypatch.setattr(inspector, "plot_topology_ensemble", plot_ensemble)
    monkeypatch.setattr(inspector, "simulate_and_plot_best_design", lambda *args, **kwargs:
                        dict(precomputed=kwargs["precomputed_simulation"],
                             desired_state=kwargs["desired_state"]))
    monkeypatch.setattr(plt, "show", lambda: None)
    result = inspector.main()
    assert result["selected_candidate_index"] == expected
    assert result["ensemble_selection_mode"] == mode
    np.testing.assert_array_equal(result["design"]["kappa_per_ns"], kappas[expected])
    np.testing.assert_array_equal(result["simulation"]["precomputed"][1],
                                  states[expected * 2:(expected + 1) * 2])
    assert f"_{mode}_topology" in result["simulation"]["desired_state"]
    assert len(simulation_calls) == 1
    assert ensemble_counts == [3]


def test_near_unity_gate_override_becomes_all_on_topology():
    config = replace(training_config, n_lasers=10)
    topology, fraction = _resolve_topology_input_override(config, None, 0.999999999)
    assert fraction is None
    assert topology.shape == (10, 10)
    assert np.all(np.diag(topology) == 0)
    assert topology.sum() == 90


def test_fixed_topology_samples_continuous_designs():
    config = replace(training_config, n_lasers=3)
    policy = BernoulliSparseGNNPolicy(config)
    topology = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    designs = infer_designs(
        policy, np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
        np.array([-0.05, 0.0, 0.05]), config, count=4,
        seed=12345, continuous_mean=False,
        topology=topology,
    )
    kappas = designs["kappa_per_ns"]
    assert kappas.shape == (4, 3, 3)
    assert np.all((kappas > 0) == topology.T)
    assert not np.allclose(kappas[0], kappas[1])


def test_ensemble_side_profile_uses_every_candidate_laser(tmp_path):
    config = replace(training_config, n_lasers=3)
    target = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    time = np.linspace(0, 500e-9, 40)
    states = np.zeros((4, 9, len(time)))
    states[:, 1::3] = 1.0
    wrapped_tail = np.deg2rad(np.where(np.arange(len(time)) % 2, 179, -179))
    states[:2, 5, :] = wrapped_tail
    frequencies_ghz = np.array([
        [0.01, 0.02, 0.03], [0.01, 0.02, 0.03],
        [0.04, 0.05, 0.06], [0.04, 0.05, 0.06],
    ])
    frequencies_nd = (
        frequencies_ghz[:, :, None]
        * (2 * np.pi * config.photon_lifetime_seconds * 1e9)
        * np.ones(len(time))
    )
    initial = np.repeat(frequencies_ghz[:, :, None],
                        len(time) * config.save_every, axis=2)
    summary = summarize_topology_trajectories(
        (time, states, frequencies_nd, initial), target, config,
        n_topologies=2, noise_iterations=2, orientations=np.ones(4),
        include_trajectories=True,
    )
    np.testing.assert_allclose(
        summary["settled_frequencies_ghz"],
        [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], atol=1e-12)
    assert np.isclose(abs(summary["settled_phases_rad"][0, 1]), np.pi)
    assert summary["settled_powers_mw"].shape == (2, 3)
    assert summary["settled_total_powers_mw"].shape == (2,)
    assert summary["frequency_trajectories"].shape == (2, 3, len(time))
    assert summary["phase_trajectories"].shape == (2, 3, len(time))
    assert summary["reward_trajectories"].shape == (2, len(time))
    assert summary["power_trajectories"].shape == (2, 3, len(time))
    assert summary["total_power_trajectories"].shape == (2, len(time))
    assert np.isclose(summary["free_running_end_ns"],
                      config.coupling_ramp_start_delays * config.delay_seconds * 1e9)
    with plt.rc_context({"text.usetex": False}):
        figure = plot_topology_ensemble(summary, target)
        assert len(figure.axes) == 7
        for axis in figure.axes[:4]:
            assert axis.patches  # Gray free-running interval.
        for axis in figure.axes:
            assert any(line.get_visible() for line in axis.get_xgridlines())
        for trajectory_axis, distribution_axis in (
            (figure.axes[0], figure.axes[4]),
            (figure.axes[1], figure.axes[5]),
            (figure.axes[3], figure.axes[6]),
        ):
            assert distribution_axis.get_shared_y_axes().joined(
                distribution_axis, trajectory_axis)
            assert np.isclose(
                max(float(np.max(line.get_xdata()))
                    for line in distribution_axis.lines), 1.0)
        guides = [line for line in figure.axes[5].lines
                  if line.get_linestyle() == "--"]
        assert len(guides) == 3
        np.testing.assert_allclose(
            sorted(float(line.get_ydata()[0]) for line in guides),
            [-2 * np.pi / 3, 0.0, 2 * np.pi / 3], atol=1e-12)
        figure.savefig(tmp_path / "ensemble.png")
        assert (tmp_path / "ensemble.png").is_file()
        plt.close(figure)

        all_figure = plot_topology_ensemble(summary, target, trajectory_mode="all")
        assert "all 2 candidate trajectories" in all_figure.axes[0].get_title()
        assert len(all_figure.axes[0].lines) == 6  # Two candidates per laser.
        assert len(all_figure.axes[2].lines) == 2  # Candidate phase rewards.
        assert len(all_figure.axes[3].lines) == 8  # Individual and total power.
        assert not all_figure.axes[2].collections  # No mean +/- SD band.
        all_figure.savefig(tmp_path / "all_candidates.png")
        assert (tmp_path / "all_candidates.png").is_file()
        plt.close(all_figure)
