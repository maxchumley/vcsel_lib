import numpy as np

from rl import reinforcement_matrix_design as design


def test_decode_actions_bounds_wraps_and_respects_topology():
    adjacency = np.array([[0.0, 1.0], [0.0, 0.0]])
    actions = np.array(
        [
            [-100.0, 4.0 * np.pi + 0.25],
            [100.0, -4.0 * np.pi - 0.50],
        ]
    )

    kappa_ns, phi_p = design.decode_actions(actions, adjacency, 20.0)

    assert kappa_ns.shape == (2, 2, 2)
    assert phi_p.shape == (2, 2, 2)
    assert np.all((kappa_ns >= 0.0) & (kappa_ns <= 20.0))
    assert np.allclose(kappa_ns[:, adjacency == 0], 0.0)
    assert np.allclose(phi_p[:, adjacency == 0], 0.0)
    assert np.allclose(phi_p[:, 0, 1], [0.25, -0.50])


def test_reward_prefers_requested_phase_and_frequency_lock(monkeypatch):
    n_lasers = 2
    n_time = 20
    states = np.zeros((2, 3 * n_lasers, n_time))
    states[:, 1::3, :] = 1.0
    states[0, 5, :] = -np.pi
    states[1, 5, :] = 0.0

    frequencies_ghz = np.zeros((2, n_lasers, n_time))
    frequencies_ghz[1, 0, :] = -1.0
    frequencies_ghz[1, 1, :] = 1.0
    monkeypatch.setattr(design, "N_lasers", n_lasers)
    monkeypatch.setattr(design, "target_phases", np.array([0.0, np.pi]))

    (
        rewards,
        phase_scores,
        mean_phase_scores,
        worst_phase_scores,
        frequency_scores,
        _,
    ) = design.trajectory_rewards(states, frequencies_ghz, tail_start=10)

    assert phase_scores[0] > 0.999
    assert mean_phase_scores[0] > 0.999
    assert worst_phase_scores[0] > 0.999
    assert frequency_scores[0] > 0.999
    assert phase_scores[1] < 0.001
    assert mean_phase_scores[1] < 0.001
    assert worst_phase_scores[1] < 0.001
    assert frequency_scores[1] < 0.001
    assert rewards[0] > rewards[1]


def test_reward_penalizes_one_bad_laser_even_when_phase_means_match(monkeypatch):
    n_lasers = 3
    n_time = 12
    states = np.zeros((2, 3 * n_lasers, n_time))
    states[:, 1::3, :] = 1.0

    # Candidate 1 has per-laser scores [1, 0]. Candidate 2 has [0.5, 0.5].
    # Their means match, but candidate 2 has the better worst laser.
    states[0, 5, :] = 0.0
    states[0, 8, :] = np.pi
    states[1, 5, :] = np.pi / 2.0
    states[1, 8, :] = np.pi / 2.0
    frequencies_ghz = np.zeros((2, n_lasers, n_time))

    monkeypatch.setattr(design, "N_lasers", n_lasers)
    monkeypatch.setattr(design, "target_phases", np.zeros(n_lasers))
    monkeypatch.setattr(design, "worst_laser_phase_weight", 0.5)

    (
        rewards,
        phase_scores,
        mean_phase_scores,
        worst_phase_scores,
        _,
        _,
    ) = design.trajectory_rewards(states, frequencies_ghz, tail_start=0)

    assert np.allclose(mean_phase_scores, [0.5, 0.5])
    assert np.allclose(worst_phase_scores, [0.0, 0.5])
    assert phase_scores[1] > phase_scores[0]
    assert rewards[1] > rewards[0]


def test_reward_accepts_a_different_phase_target_for_each_case(monkeypatch):
    n_lasers = 2
    states = np.zeros((2, 3 * n_lasers, 8))
    states[:, 1::3, :] = 1.0
    states[0, 5, :] = np.pi
    states[1, 5, :] = 0.0
    frequencies_ghz = np.zeros((2, n_lasers, 8))
    config = {
        "target_phases": np.array([[0.0, np.pi], [0.0, 0.0]]),
        "worst_laser_phase_weight": 0.5,
        "phase_weight": 0.7,
        "frequency_lock_weight": 0.3,
        "frequency_tolerance_ghz": 0.1,
    }

    rewards, phase_scores, _, _, _, _ = design.trajectory_rewards(
        states, frequencies_ghz, tail_start=0, reward_config=config
    )

    assert np.all(phase_scores > 0.999)
    assert np.all(rewards > 0.999)


def test_short_batched_training_writes_best_matrices(tmp_path, monkeypatch):
    monkeypatch.setattr(design, "n_restarts", 2)
    monkeypatch.setattr(design, "parallel_restarts", 1)
    monkeypatch.setattr(design, "n_iterations", 1)
    monkeypatch.setattr(design, "population_size", 2)
    monkeypatch.setattr(design, "Tmax", 10.0 * design.tau)
    monkeypatch.setattr(design, "save_every", 5)
    output_file = tmp_path / "best_design.npz"
    monkeypatch.setattr(design, "output_file", output_file)
    monkeypatch.setattr(design, "show_inline_plot", False)
    captured = {"kappa_shapes": [], "ramps": []}
    original_integrate = design.VCSEL.integrate

    def capture_training_ramp(self, *args, **kwargs):
        nd_run = kwargs["nd"]
        captured["kappa_shapes"].append(nd_run["kappa"].shape)
        captured["ramps"].append(nd_run["kappa_ramp"].copy())
        return original_integrate(self, *args, **kwargs)

    monkeypatch.setattr(design.VCSEL, "integrate", capture_training_ramp)

    kappa_ns, phi_p, reward = design.run_training()

    assert kappa_ns.shape == (design.N_lasers, design.N_lasers)
    assert phi_p.shape == (design.N_lasers, design.N_lasers)
    assert np.isfinite(reward)
    assert np.allclose(np.diag(kappa_ns), 0.0)
    assert np.allclose(np.diag(phi_p), 0.0)
    assert len(captured["kappa_shapes"]) == design.n_restarts
    assert captured["kappa_shapes"] == design.n_restarts * [(
        design.population_size,
        design.N_lasers,
        design.N_lasers,
    )]
    assert captured["ramps"][0].shape == (int(design.Tmax / design.dt),)
    assert captured["ramps"][0][0] == 0.0
    assert captured["ramps"][0][-1] == 1.0
    assert np.array_equal(captured["ramps"][0], captured["ramps"][1])
    with np.load(output_file) as saved:
        assert saved["kappa_ns"].shape == (design.N_lasers, design.N_lasers)
        assert saved["phi_p"].shape == (design.N_lasers, design.N_lasers)
        assert saved["mean_reward_history"].shape == (2,)
        assert saved["restart_best_reward_history"].shape == (2,)
        assert saved["best_reward_history"].shape == (2,)
        assert np.all(np.diff(saved["best_reward_history"]) >= 0.0)
        assert np.array_equal(saved["restart_index_history"], [1, 2])
        assert np.array_equal(saved["restart_seeds"], [7, 8])
        assert saved["n_restarts"] == design.n_restarts
        assert saved["parallel_restarts"] == 1
        assert 1 <= saved["best_restart"] <= design.n_restarts
        assert 0.0 <= saved["worst_phase_score"] <= 1.0
        assert saved["worst_laser_phase_weight"] == 0.5
        assert saved["coupling_ramp_start_tau"] == design.coupling_ramp_start_tau
        assert saved["coupling_ramp_time_tau"] == design.coupling_ramp_time_tau


def test_final_design_cell_simulates_saved_matrices(tmp_path, monkeypatch):
    n_lasers = 2
    output_file = tmp_path / "best_design.npz"
    np.savez(
        output_file,
        kappa_s=np.array([[0.0, 5e9], [5e9, 0.0]]),
        phi_p=np.zeros((n_lasers, n_lasers)),
    )

    monkeypatch.setattr(design, "N_lasers", n_lasers)
    monkeypatch.setattr(
        design,
        "delta",
        np.linspace(-0.5, 0.5, n_lasers)
        * design.detuning_span_ghz
        * 2.0
        * np.pi
        * 1e9,
    )
    monkeypatch.setattr(design, "save_every", 20)
    monkeypatch.setattr(design, "output_file", output_file)
    monkeypatch.setattr(design.plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)

    time, states, frequencies = design.simulate_final_design()

    assert time.ndim == 1
    assert states.shape[0:2] == (1, 3 * n_lasers)
    assert frequencies.shape[0:2] == (1, n_lasers)
    assert states.shape[2] == time.size == frequencies.shape[2]
    assert (tmp_path / "reinforcement_matrix_design_validation.png").exists()
