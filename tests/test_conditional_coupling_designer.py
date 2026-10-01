from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parents[1]))
from rl import conditional_coupling_designer as conditional


def small_config(**changes):
    config = conditional.DesignerConfig(
        training_iterations=1,
        targets_per_batch=2,
        candidates_per_target=4,
        hidden_sizes=(16, 16),
        detuning_hidden_sizes=(8,),
        edge_hidden_sizes=(16, 16),
        simulation_time_seconds=30.0e-9,
        held_out_target_count=3,
        validation_interval=1,
        magnitude_symmetry_weight=0.0,
        jupyter_mode=False,
    )
    return replace(config, **changes)


def test_standalone_module_has_no_old_optimizer_dependency():
    source = open(conditional.__file__, encoding="utf-8").read()
    assert "import reinforcement_matrix_design" not in source
    assert "base." not in source


def test_n_jobs_must_be_positive():
    with pytest.raises(ValueError, match="n_jobs"):
        small_config(n_jobs=0)


def test_detuning_curriculum_grows_random_span_after_warmup():
    config = small_config(
        detuning_span_ghz=10.0,
        detuning_curriculum_initial_half_span_ghz=0.05,
        detuning_curriculum_warmup_iterations=2,
        detuning_curriculum_iterations=6,
    )
    warmup = conditional.sample_detuning_distributions(
        2, np.random.default_rng(5), config, iteration=2
    )
    partial = conditional.sample_detuning_distributions(
        2, np.random.default_rng(5), config, iteration=4
    )
    full = conditional.sample_detuning_distributions(
        2, np.random.default_rng(5), config, iteration=6
    )

    assert np.max(np.abs(warmup)) <= 0.05
    np.testing.assert_allclose(partial, 50.5 * warmup)
    np.testing.assert_allclose(full, 100.0 * warmup)
    np.testing.assert_allclose(warmup.mean(axis=1), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(partial.mean(axis=1), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(full.mean(axis=1), 0.0, atol=1.0e-12)
    assert np.all(np.diff(warmup, axis=1) >= 0.0)
    assert np.all(np.diff(partial, axis=1) >= 0.0)
    assert np.all(np.diff(full, axis=1) >= 0.0)


def test_detuning_curriculum_warmup_cannot_exceed_end_iteration():
    with pytest.raises(ValueError, match="warmup"):
        small_config(
            detuning_curriculum_warmup_iterations=7,
            detuning_curriculum_iterations=6,
        )


def test_initial_detuning_half_span_cannot_exceed_full_range():
    with pytest.raises(ValueError, match="initial_half_span"):
        small_config(
            detuning_span_ghz=10.0,
            detuning_curriculum_initial_half_span_ghz=5.1,
        )


def test_default_checkpoint_names_include_laser_count():
    config = conditional.DesignerConfig(n_lasers=7)

    assert config.best_checkpoint_file == str(
        conditional.MODEL_DIR / "conditional_coupling_edge_7_lasers_best.pt"
    )
    assert config.current_checkpoint_file == str(
        conditional.MODEL_DIR / "conditional_coupling_edge_7_lasers_current.pt"
    )
    assert config.final_checkpoint_file == str(
        conditional.MODEL_DIR / "conditional_coupling_edge_7_lasers_final.pt"
    )

    custom = conditional.DesignerConfig(
        n_lasers=7,
        best_checkpoint_file="custom_best.pt",
        current_checkpoint_file="custom_current.pt",
        final_checkpoint_file="custom_final.pt",
    )
    assert custom.best_checkpoint_file == "custom_best.pt"
    assert custom.current_checkpoint_file == "custom_current.pt"
    assert custom.final_checkpoint_file == "custom_final.pt"


def test_exploration_cap_anneals_linearly():
    config = small_config(
        maximum_log_std=0.75,
        enable_exploration_annealing=True,
        exploration_anneal_start_iteration=2,
        exploration_anneal_iterations=4,
        final_maximum_log_std=-1.5,
    )

    assert conditional.maximum_log_std_at(1, config) == pytest.approx(0.75)
    assert conditional.maximum_log_std_at(2, config) == pytest.approx(0.1875)
    assert conditional.maximum_log_std_at(5, config) == pytest.approx(-1.5)
    disabled = replace(config, enable_exploration_annealing=False)
    assert conditional.maximum_log_std_at(5, disabled) == pytest.approx(0.75)


def test_learning_rate_must_be_positive():
    with pytest.raises(ValueError, match="learning_rate"):
        small_config(learning_rate=0.0)


def test_relative_phase_and_encoding_ignore_global_phase():
    phases = np.array([0.4, -2.0, 0.8, 2.2, -1.1])
    relative = conditional.make_relative_to_laser_1(phases)
    encoded = conditional.encode_target_phases(phases)
    shifted = conditional.encode_target_phases(phases + 7.3)

    assert relative[0] == pytest.approx(0.0)
    assert encoded.shape == (8,)
    assert encoded.dtype == np.float32
    assert np.allclose(encoded, shifted)


def test_global_conjugates_have_one_policy_representation():
    target = np.array([0.0, -0.7, 1.4, -2.1, 2.8])
    direct = conditional.canonicalize_global_phase_conjugate(target)
    conjugate = conditional.canonicalize_global_phase_conjugate(-target)
    assert np.allclose(direct, conjugate)


def test_policy_sampling_shapes_and_log_probability_gradient():
    config = small_config()
    policy = conditional.PolicyNetwork(config)
    targets = conditional.sample_target_phases(
        2, np.random.default_rng(4)
    )
    encoded = conditional.encode_for_policy(targets, policy, config)
    actions, log_probabilities = conditional.sample_coupling_designs(
        policy, encoded, candidates_per_target=4
    )

    assert encoded.shape == (2, 13)
    assert actions.shape == (2, 4, 40)
    assert log_probabilities.shape == (2, 4)
    (-log_probabilities.mean()).backward()
    assert policy.phase_encoder[0].weight.grad is not None
    assert policy.detuning_encoder[0].weight.grad is not None
    assert policy.global_fusion[0].weight.grad is not None
    assert policy.edge_decoder[-1].weight.grad is not None


def test_edge_decoder_shares_one_rule_across_directed_links():
    config = small_config()
    policy = conditional.PolicyNetwork(config)
    encoded = conditional.encode_for_policy(
        np.zeros((1, config.n_lasers)), policy, config
    )

    mean = policy(encoded)[0]
    magnitude_logits = mean[: policy.n_directed_links]
    coupling_phases = mean[policy.n_directed_links :]

    # Every in-phase link has the same local [cos(delta), sin(delta)] = [1, 0]
    # and therefore receives the same result from the shared decoder.
    assert torch.allclose(
        magnitude_logits,
        magnitude_logits[0].expand_as(magnitude_logits),
    )
    assert torch.allclose(
        coupling_phases,
        coupling_phases[0].expand_as(coupling_phases),
    )
    assert policy.link_receivers.shape == (20,)
    assert policy.link_sources.shape == (20,)


def test_edge_decoder_receives_local_phase_and_detuning_features():
    config = small_config()
    policy = conditional.PolicyNetwork(config)
    target = np.array([[0.0, 0.3, -0.8, 1.4, -2.0]])
    detunings_ghz = np.array([[-5.0, -2.5, 0.0, 2.5, 5.0]])
    encoded = conditional.encode_for_policy(
        target, policy, config, detunings_ghz
    )
    captured = {}

    def capture_edge_inputs(_module, inputs):
        captured["edge_inputs"] = inputs[0].detach().cpu().numpy()

    hook = policy.edge_decoder[0].register_forward_pre_hook(
        capture_edge_inputs
    )
    policy(encoded)
    hook.remove()

    relative = conditional.make_relative_to_laser_1(target)[0]
    receivers, sources = conditional.directed_link_indices(config.n_lasers)
    differences = relative[sources] - relative[receivers]
    expected_phase = np.column_stack(
        [np.cos(differences), np.sin(differences)]
    )
    normalized_detunings = conditional.encode_detuning_distributions(
        detunings_ghz, config
    )[0]
    signed_detuning = 0.5 * (
        normalized_detunings[sources] - normalized_detunings[receivers]
    )
    expected_detuning = np.column_stack(
        [signed_detuning, np.abs(signed_detuning)]
    )
    expected = np.column_stack((expected_phase, expected_detuning))

    assert captured["edge_inputs"].shape == (
        config.n_lasers * (config.n_lasers - 1),
        policy.global_embedding_width + 4,
    )
    np.testing.assert_allclose(
        captured["edge_inputs"][:, -4:],
        expected,
        atol=1.0e-6,
    )


def test_decode_action_bounds_order_and_zero_diagonal():
    n_lasers = 5
    n_links = n_lasers * (n_lasers - 1)
    actions = np.zeros(
        (1, conditional.action_size_for(n_lasers))
    )
    actions[0, 0] = 30.0
    actions[0, n_links] = np.pi / 3.0
    kappa_per_ns, phi_p_rad = conditional.decode_action(
        actions, 100.0, n_lasers
    )

    assert kappa_per_ns.shape == (1, 5, 5)
    assert phi_p_rad.shape == (1, 5, 5)
    assert np.allclose(np.diagonal(kappa_per_ns[0]), 0.0)
    assert np.allclose(np.diagonal(phi_p_rad[0]), 0.0)
    assert np.all((kappa_per_ns >= 0.0) & (kappa_per_ns <= 100.0))
    assert np.all((phi_p_rad >= -np.pi) & (phi_p_rad <= np.pi))
    # First directed link is source 1 -> receiver 0.
    assert kappa_per_ns[0, 0, 1] > 99.9
    assert phi_p_rad[0, 0, 1] == pytest.approx(np.pi / 3.0)


def test_symmetric_kappa_policy_keeps_directed_coupling_phases():
    config = small_config(force_symmetric_kappa=True)
    policy = conditional.PolicyNetwork(config)
    unique_links = config.n_lasers * (config.n_lasers - 1) // 2
    directed_links = config.n_lasers * (config.n_lasers - 1)
    actions = np.linspace(
        -2.0,
        2.0,
        conditional.action_size_for(config.n_lasers, True),
    )[None, :]

    kappa_per_ns, phi_p_rad = conditional.decode_action(
        actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
    )

    assert policy.n_magnitude_links == unique_links
    assert policy.n_directed_links == directed_links
    assert policy.action_size == unique_links + directed_links
    np.testing.assert_allclose(kappa_per_ns, kappa_per_ns.transpose(0, 2, 1))
    assert not np.allclose(phi_p_rad, phi_p_rad.transpose(0, 2, 1))
    np.testing.assert_allclose(np.diagonal(kappa_per_ns[0]), 0.0)
    np.testing.assert_allclose(np.diagonal(phi_p_rad[0]), 0.0)


def test_kappa_and_phi_p_can_both_be_forced_symmetric():
    config = small_config(
        force_symmetric_kappa=True,
        force_symmetric_phi_p=True,
    )
    policy = conditional.PolicyNetwork(config)
    unique_links = config.n_lasers * (config.n_lasers - 1) // 2
    actions = np.linspace(-1.5, 1.5, 2 * unique_links)[None, :]

    kappa_per_ns, phi_p_rad = conditional.decode_action(
        actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
    )

    assert policy.n_magnitude_links == unique_links
    assert policy.n_phase_links == unique_links
    assert policy.action_size == 2 * unique_links
    np.testing.assert_allclose(kappa_per_ns, kappa_per_ns.transpose(0, 2, 1))
    np.testing.assert_allclose(phi_p_rad, phi_p_rad.transpose(0, 2, 1))


def test_phase_reward_accepts_one_global_conjugate():
    config = small_config(allow_global_phase_conjugate=True)
    target = np.array([[0.0, 0.7, -1.4, 2.1, -2.8]])
    states = np.zeros((1, 15, 40))
    states[:, 2::3, :] = -target[:, :, None]
    time_seconds = np.linspace(0.0, 30.0e-9, states.shape[2])

    reward = conditional.calculate_phase_reward(
        time_seconds, states, target, config
    )

    assert reward[0][0] == pytest.approx(1.0)
    assert reward[5][0] == -1.0
    assert np.allclose(reward[4][0], target[0])


def test_magnitude_symmetry_penalty_is_normalized():
    symmetric = np.array(
        [[[0.0, 40.0], [40.0, 0.0]]], dtype=float
    )
    one_way = np.array(
        [[[0.0, 100.0], [0.0, 0.0]]], dtype=float
    )
    assert conditional.normalized_magnitude_symmetry_penalty(
        symmetric, 100.0
    )[0] == pytest.approx(0.0)
    assert conditional.normalized_magnitude_symmetry_penalty(
        one_way, 100.0
    )[0] == pytest.approx(1.0)


def test_real_vcsel_integration_uses_case_dependent_matrices():
    config = small_config()
    actions = np.zeros(
        (2, conditional.action_size_for(config.n_lasers))
    )
    kappa_per_ns, phi_p_rad = conditional.decode_action(
        actions, 100.0, config.n_lasers
    )
    time_seconds, states, frequencies, initial_frequency = (
        conditional.simulate_with_vcsel(
            kappa_per_ns, phi_p_rad, config
        )
    )

    assert time_seconds.ndim == 1
    assert states.shape[:2] == (2, 15)
    assert states.shape[2] == len(time_seconds)
    assert frequencies.shape == (2, 5, len(time_seconds))
    assert initial_frequency.shape[:2] == (2, 5)


@pytest.mark.parametrize(
    ("target_count", "n_jobs", "expected_sizes"),
    [
        (8, 4, [2, 2, 2, 2]),
        (10, 3, [4, 3, 3]),
        (2, 8, [1, 1]),
    ],
)
def test_simulation_split_preserves_targets_and_all_candidates(
    target_count, n_jobs, expected_sizes
):
    config = small_config(n_jobs=n_jobs)
    candidates = 3
    targets = np.arange(target_count * 5, dtype=float).reshape(
        target_count, 5
    )
    kappa = np.arange(
        target_count * candidates * 25, dtype=float
    ).reshape(target_count, candidates, 5, 5)
    phi_p = -kappa

    work_items = conditional.split_simulation_batch(
        targets, kappa, phi_p, config
    )

    assert [item[2] - item[1] for item in work_items] == expected_sizes
    assert all(item[4].shape[1] == candidates for item in work_items)
    assert np.array_equal(
        np.concatenate([item[3] for item in work_items]), targets
    )
    assert np.array_equal(
        np.concatenate([item[4] for item in work_items]), kappa
    )
    assert np.array_equal(
        np.concatenate([item[5] for item in work_items]), phi_p
    )
    covered_indices = np.concatenate(
        [np.arange(item[1], item[2]) for item in work_items]
    )
    assert np.array_equal(covered_indices, np.arange(target_count))


def test_target_chunk_calls_vectorized_simulator_once(monkeypatch):
    config = small_config(n_jobs=1)
    target_count = 3
    candidates = 4
    targets = np.arange(target_count * 5, dtype=float).reshape(
        target_count, 5
    )
    kappa = np.zeros((target_count, candidates, 5, 5))
    phi_p = np.zeros_like(kappa)
    simulator_calls = []

    def fake_simulator(flat_kappa, flat_phi_p, worker_config):
        simulator_calls.append((flat_kappa.shape, flat_phi_p.shape))
        systems = len(flat_kappa)
        return (
            np.array([0.0]),
            np.zeros((systems, 15, 1)),
            np.empty(0),
            np.empty(0),
        )

    def fake_reward(time_seconds, states, repeated_targets, worker_config):
        expected_targets = np.repeat(targets, candidates, axis=0)
        assert np.array_equal(repeated_targets, expected_targets)
        systems = len(repeated_targets)
        scalar = np.arange(systems, dtype=float)
        phases = np.repeat(scalar[:, None], 5, axis=1)
        return scalar, scalar, scalar, phases, phases, np.ones(systems)

    monkeypatch.setattr(
        conditional, "simulate_with_vcsel", fake_simulator
    )
    monkeypatch.setattr(
        conditional, "calculate_phase_reward", fake_reward
    )
    work_item = conditional.split_simulation_batch(
        targets, kappa, phi_p, config
    )[0]

    result = conditional.simulate_target_chunk(work_item)

    assert simulator_calls == [((12, 5, 5), (12, 5, 5))]
    assert result["phase_reward"].shape == (3, 4)
    assert result["achieved_phases_rad"].shape == (3, 4, 5)


def test_serial_and_spawn_parallel_simulation_match():
    serial_config = small_config(
        n_lasers=3,
        targets_per_batch=3,
        candidates_per_target=2,
        n_jobs=1,
    )
    parallel_config = replace(serial_config, n_jobs=2)
    rng = np.random.default_rng(22)
    targets = conditional.sample_target_phases(
        3, rng, n_lasers=3
    )
    actions = rng.normal(
        size=(3 * 2, conditional.action_size_for(3))
    )
    flat_kappa, flat_phi_p = conditional.decode_action(
        actions, serial_config.maximum_kappa_per_ns, 3
    )
    kappa = flat_kappa.reshape(3, 2, 3, 3)
    phi_p = flat_phi_p.reshape(3, 2, 3, 3)

    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=2,
        initializer=conditional._initialize_simulation_worker,
    )
    try:
        parallel = conditional.simulate_candidates_parallel(
            targets,
            kappa,
            phi_p,
            parallel_config,
            simulation_pool=pool,
        )
    finally:
        pool.close()
        pool.join()
    serial = conditional.simulate_candidates_parallel(
        targets, kappa, phi_p, serial_config
    )

    assert serial["phase_reward"].shape == (3, 2)
    for key in serial:
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=0.0, atol=0.0
        )


def test_three_laser_configuration_derives_every_dimension():
    config = small_config(n_lasers=3)
    policy = conditional.PolicyNetwork(config)
    targets = conditional.sample_target_phases(
        2,
        np.random.default_rng(8),
        n_lasers=config.n_lasers,
    )
    encoded = conditional.encode_for_policy(targets, policy, config)
    actions, _ = conditional.sample_coupling_designs(
        policy, encoded, candidates_per_target=4
    )
    flat_actions = actions.numpy().reshape(-1, policy.action_size)
    kappa_per_ns, phi_p_rad = conditional.decode_action(
        flat_actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
    )
    time_seconds, states, frequencies, _ = (
        conditional.simulate_with_vcsel(
            kappa_per_ns[:1], phi_p_rad[:1], config
        )
    )

    assert targets.shape == (2, 3)
    assert encoded.shape == (2, 7)
    assert policy.n_directed_links == 6
    assert policy.action_size == 12
    assert actions.shape == (2, 4, 12)
    assert kappa_per_ns.shape == (8, 3, 3)
    assert np.allclose(np.diagonal(kappa_per_ns, axis1=1, axis2=2), 0.0)
    assert states.shape == (1, 9, len(time_seconds))
    assert frequencies.shape == (1, 3, len(time_seconds))


def test_real_reinforce_batch_changes_policy_parameters():
    config = small_config(
        enable_exploration_annealing=True,
        exploration_anneal_start_iteration=0,
        exploration_anneal_iterations=0,
        final_maximum_log_std=-1.5,
    )
    rng = conditional.set_random_seed(config.random_seed)
    policy = conditional.PolicyNetwork(config)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate
    )
    before = torch.cat(
        [parameter.detach().flatten() for parameter in policy.parameters()]
    ).clone()

    metrics = conditional.run_training_batch(
        policy, optimizer, config, rng, iteration=0
    )
    after = torch.cat(
        [parameter.detach().flatten() for parameter in policy.parameters()]
    )

    assert np.isfinite(list(metrics.values())).all()
    assert metrics["gradient_norm"] > 0.0
    assert metrics["maximum_log_std"] == pytest.approx(-1.5)
    assert torch.max(policy.log_std).item() <= -1.5
    assert not torch.equal(before, after)


def test_parallel_reinforce_batch_changes_policy_parameters():
    config = small_config(
        n_lasers=3,
        targets_per_batch=2,
        candidates_per_target=2,
        n_jobs=2,
    )
    rng = conditional.set_random_seed(config.random_seed)
    policy = conditional.PolicyNetwork(config)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate
    )
    before = torch.cat(
        [parameter.detach().flatten() for parameter in policy.parameters()]
    ).clone()

    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=2,
        initializer=conditional._initialize_simulation_worker,
    )
    try:
        metrics = conditional.run_training_batch(
            policy,
            optimizer,
            config,
            rng,
            iteration=0,
            simulation_pool=pool,
        )
    finally:
        pool.close()
        pool.join()
    after = torch.cat(
        [parameter.detach().flatten() for parameter in policy.parameters()]
    )

    assert np.isfinite(list(metrics.values())).all()
    assert metrics["gradient_norm"] > 0.0
    assert not torch.equal(before, after)


def test_train_loop_reuses_and_closes_spawn_pool(tmp_path):
    config = small_config(
        n_lasers=3,
        training_iterations=2,
        targets_per_batch=2,
        candidates_per_target=2,
        held_out_target_count=2,
        n_jobs=2,
        best_checkpoint_file=str(tmp_path / "best.pt"),
        current_checkpoint_file=str(tmp_path / "current.pt"),
        final_checkpoint_file=str(tmp_path / "final.pt"),
    )
    children_before = {child.pid for child in mp.active_children()}

    _, _, history = conditional.train_reinforce(
        config, live_plot=False
    )

    children_after = {child.pid for child in mp.active_children()}
    assert len(history["phase_reward"]) == 2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "current.pt").exists()
    assert (tmp_path / "final.pt").exists()
    assert children_after == children_before


def test_train_loop_updates_one_progress_png(
    tmp_path, monkeypatch
):
    config = small_config(
        training_iterations=2,
        validation_interval=2,
        best_checkpoint_file=str(tmp_path / "best.pt"),
        final_checkpoint_file=str(tmp_path / "final.pt"),
    )

    def fake_training_batch(*_args, **_kwargs):
        return {
            "loss": 0.0,
            "phase_reward": 0.5,
            "magnitude_symmetry_penalty": 0.0,
            "training_reward": 0.5,
            "gradient_norm": 0.0,
            "maximum_log_std": 0.75,
            "mean_action_std": 0.5,
        }

    def fake_evaluation(*_args, **_kwargs):
        return {"phase_reward": np.full(config.held_out_target_count, 0.5)}

    monkeypatch.setattr(
        conditional, "run_training_batch", fake_training_batch
    )
    monkeypatch.setattr(conditional, "evaluate_policy", fake_evaluation)
    monkeypatch.setattr(
        conditional,
        "save_checkpoint",
        lambda filename, *_args, **_kwargs: Path(filename),
    )
    monkeypatch.setattr(conditional.plt, "pause", lambda _seconds: None)
    monkeypatch.setattr(conditional, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.chdir(tmp_path)

    conditional.train_reinforce(config, live_plot=True)

    progress_directory = tmp_path / "results" / "training" / "progress"
    progress_file = progress_directory / (
        f"{Path(config.current_checkpoint_file).stem}_training_progress.png"
    )
    assert progress_file.exists()
    assert len(list(progress_directory.glob("*.png"))) == 1
    conditional.plt.close("all")


def test_new_checkpoint_round_trip_and_old_format_rejection(tmp_path):
    config = small_config()
    policy = conditional.PolicyNetwork(config)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate
    )
    path = tmp_path / "simple.pt"
    conditional.save_checkpoint(
        path,
        policy,
        optimizer,
        config,
        iteration=3,
        held_out_reward=0.7,
    )
    loaded_policy, _, loaded_config, metadata = (
        conditional.load_checkpoint(path)
    )

    assert loaded_config.maximum_kappa_per_ns == 100.0
    assert metadata == {"iteration": 3, "held_out_reward": 0.7}
    for expected, actual in zip(
        policy.parameters(), loaded_policy.parameters()
    ):
        assert torch.equal(expected, actual)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["format"] == conditional.CHECKPOINT_FORMAT

    legacy_schedule_path = tmp_path / "legacy_schedule.pt"
    saved["config"].update(
        {
            "learning_rate_decay_iteration": 400,
            "learning_rate_after_decay": 3.0e-4,
            "final_learning_rate_iteration": 1500,
            "final_learning_rate": 1.0e-4,
        }
    )
    torch.save(saved, legacy_schedule_path)
    _, legacy_optimizer, legacy_config, _ = conditional.load_checkpoint(
        legacy_schedule_path
    )
    assert legacy_config.learning_rate == pytest.approx(1.0e-3)
    assert legacy_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    plain_path = tmp_path / "plain.pt"
    torch.save({"format": "conditional_coupling_simple_v1"}, plain_path)
    with pytest.raises(ValueError, match="plain 40-output policy"):
        conditional.load_checkpoint(plain_path)

    archived_path = tmp_path / "archived.pt"
    torch.save(
        {"format": "conditional_coupling_policy_v1"}, archived_path
    )
    with pytest.raises(ValueError, match="another policy architecture"):
        conditional.load_checkpoint(archived_path)
