"""Locking gates, physical correction labels and the opt-in training boundary."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from rl.rl_api import CouplingModel, Experiment, PhaseCorrectionConfig, ValidationConfig
from rl.rl_api._compat import core
from rl.rl_api.experiments import train_mirrored_root_encoder_1000 as previous
from rl.rl_api.experiments import train_phase_feedback_1000 as run
from rl.rl_api.models import build_policy
from rl.rl_api.phase_correction import (collect_measurements, locked, measure_locking,
    prepare_teachers, teacher_losses, verify_teachers, wrap)
from rl.rl_api.sampling import sample_band_random_reachable_topologies
from rl.rl_api.training_batch import prepare_size


def test_recipe_changes_only_correction_settings_and_output_name(tmp_path):
    old, new = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert old.pop('phase_correction')['weight'] == 0.
    assert new.pop('phase_correction') == dict(weight=1., frequency_tolerance_mhz=1.,
                                               phase_jitter_tolerance_deg=1., mode='sampled_joint')
    assert old == new
    assert asdict(previous.CONFIG.to_backend(tmp_path)) == asdict(run.CONFIG.to_backend(tmp_path))
    assert run.RUN_NAME != previous.RUN_NAME
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['correction_verification_simulations_max'] == 175
    assert summary['root_selection'] == 'uniform per topology'
    assert not (tmp_path/'run').exists()
    with pytest.raises(ValueError, match='nonnegative'):
        PhaseCorrectionConfig(weight=-1.)


def test_locking_gate_ignores_target_bias_but_rejects_drift_jitter_and_nonfinite(tmp_path):
    cfg = replace(run.CONFIG.to_backend(tmp_path), n_lasers=3)
    times = np.linspace(0., 500e-9, 1001)
    states = np.zeros((4, 9, len(times)))
    states[:, 1::3] = 1.
    # A shared 2 GHz frequency and arbitrary relative offsets are locked.
    states[:, 2::3] = 2*np.pi*2e9*times+np.array([0., 1.4, -2.])[None, :, None]
    states[1, 5] += 2*np.pi*2e6*times
    states[2, 8] += .1*np.sin(2*np.pi*200e6*times)
    states[3, 1, -1] = np.nan
    measurements = measure_locking(times, states, cfg)
    np.testing.assert_array_equal(locked(measurements, run.CONFIG.phase_correction),
                                  [True, False, False, False])
    np.testing.assert_allclose(measurements['locking_achieved_phases_rad'][0], [0., 1.4, -2.], atol=1e-11)


def test_measured_correction_preserves_delayed_coupling_on_feedback_graphs():
    rng = np.random.default_rng(12)
    size = 5
    receiver, source = core.directed_link_indices(size)
    now, delayed = rng.normal(size=(2, size))
    target, achieved = rng.uniform(-np.pi, np.pi, (2, size))
    beta = wrap(target-achieved)
    phase = rng.uniform(-np.pi, np.pi, (size, size))
    corrected = wrap(phase+beta[None, :]-beta[:, None])
    old_argument = delayed[source]-now[receiver]-phase[receiver, source]
    new_argument = delayed[source]+beta[source]-now[receiver]-beta[receiver]-corrected[receiver, source]
    np.testing.assert_allclose(np.exp(1j*new_argument), np.exp(1j*old_argument), atol=2e-15)


def prepared_for(tmp_path, size=4):
    config = run.CONFIG.to_backend(tmp_path)
    torch.manual_seed(7)
    policy = build_policy(config)
    item = prepare_size(policy, config, np.random.default_rng(99), size, 2, 0,
                        sample_band_random_reachable_topologies)
    shape = item['counts'].shape
    item['locking'] = dict(locking_finite=np.ones(shape, dtype=bool),
        locking_frequency_spread_mhz=np.zeros(shape), locking_phase_jitter_deg=np.zeros(shape),
        locking_achieved_phases_rad=np.zeros(shape+(size,)),
        phase_rms_reward=np.tile(np.arange(shape[1])/100., (shape[0], 1)),
        rms_orientation=np.ones(shape))
    return policy, item


def test_teacher_selection_keeps_strengths_masks_and_best_action_and_verifies_each_once(tmp_path):
    _, item = prepared_for(tmp_path)
    # Reject the first topology; keep the remaining nine context/topology blocks.
    item['locking']['locking_phase_jitter_deg'][0, :8] = 10.
    before = item['kappa'].copy()
    jobs, diagnostics = prepare_teachers([item], run.CONFIG.phase_correction)
    assert diagnostics['phase_correction_attempted'] == 9
    assert {teacher['column'] % 8 for teacher in item['phase_teachers']} == {7}
    outputs = []
    receiver, source = core.directed_link_indices(item['n_lasers'])
    for group, _, _, job in jobs:
        _, start, stop, targets, strengths, phases, detunings, config = job
        for index, teacher in enumerate(item['phase_teachers'][start:stop]):
            row, column = teacher['row'], teacher['column']
            np.testing.assert_array_equal(strengths[index, 0], item['kappa'][row, column])
            np.testing.assert_array_equal(targets[index], item['targets'][row])
            np.testing.assert_array_equal(detunings[index], item['detunings'][row])
            enabled = np.zeros((config.n_lasers, config.n_lasers), dtype=bool)
            enabled[receiver, source] = item['gates'][row, column].numpy().astype(bool)
            assert np.all(phases[index, 0][~enabled] == 0.)
        shape = (stop-start, 1)
        outputs.append(dict(size_batch_index=group, start_index=start, stop_index=stop,
            locking_finite=np.ones(shape, dtype=bool), locking_frequency_spread_mhz=np.zeros(shape),
            locking_phase_jitter_deg=np.zeros(shape), phase_rms_reward=np.ones(shape)))
    # A corrected case that loses locking and one that loses reward are excluded.
    outputs[0]['locking_phase_jitter_deg'][0, 0] = 2.
    outputs[0]['phase_rms_reward'][1, 0] = -1.
    verify_teachers([item], reversed(outputs), run.CONFIG.phase_correction, diagnostics)
    assert diagnostics['phase_correction_accepted'] == 7
    np.testing.assert_array_equal(item['kappa'], before)
    with pytest.raises(RuntimeError, match='every teacher exactly once'):
        verify_teachers([item], outputs[:-1], run.CONFIG.phase_correction, diagnostics)


def test_circular_teacher_loss_uses_physical_root_reference_and_only_active_phase_means(tmp_path):
    policy, item = prepared_for(tmp_path)
    row, column = np.argwhere(item['roots'] != 0)[0]
    teacher = dict(row=int(row), column=int(column), accepted=True)
    row, column = teacher['row'], teacher['column']
    mask = item['gates'][row:row+1, column]
    roots = item['roots'][row:row+1, column]
    from rl.rl_api.root_context import condition_on_roots
    from rl.rl_api.phase_reference import apply_root_reference
    with torch.no_grad():
        mean, *_ = condition_on_roots(policy, roots).parameters_for_edges(
            item['features'][row:row+1], topology_masks=mask)
    receiver, source = core.directed_link_indices(item['n_lasers'])
    phase = np.zeros((1, 1, item['n_lasers'], item['n_lasers']))
    phase[0, 0, receiver, source] = mean[0, len(receiver):].numpy()
    phase = apply_root_reference(phase, mask[:, None], item['detunings'][row:row+1],
                                roots[:, None], item['config'])[0, 0]
    enabled = np.zeros_like(phase, dtype=bool)
    enabled[receiver, source] = mask[0].numpy().astype(bool)
    teacher['phase'] = phase.copy()
    teacher['phase'][~enabled] += 1.1
    item['phase_teachers'] = [teacher]
    _, losses = teacher_losses(policy, item)
    assert losses.item() == pytest.approx(0., abs=1e-7)
    teacher['phase'][enabled] += .2+2*np.pi
    _, losses = teacher_losses(policy, item)
    assert losses.item() == pytest.approx(1-np.cos(.2), abs=1e-7)
    losses.mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.parameters())
    assert all(p.grad is None for name,p in policy.named_parameters() if name.startswith('budget_'))


def test_only_the_verified_teachers_density_group_gets_the_extra_pcgrad_gradient(tmp_path, monkeypatch):
    from rl.rl_api._compat import engine
    from rl.rl_api.training_batch import pcgrad_update
    policy, item = prepared_for(tmp_path)
    shape = item['counts'].shape
    item['phase_teachers'] = [dict(row=0, column=8, accepted=True,
                                   phase=wrap(item['phi'][0, 8]+.7))]
    gradients = []
    combine = core.combine_task_gradients_pcgrad
    def capture(task_gradients, *args, **kwargs):
        gradients.append(task_gradients.clone())
        return combine(task_gradients, *args, **kwargs)
    monkeypatch.setattr(core, 'combine_task_gradients_pcgrad', capture)
    optimizer = torch.optim.SGD(policy.parameters(), lr=.01)
    result = pcgrad_update(policy, optimizer, policy.config, np.random.default_rng(9),
        [item], [np.zeros(shape)], phase_correction=run.CONFIG.phase_correction)
    bands = engine.topology_density_band_ids(item['counts'], item['n_lasers'])
    teacher_band = bands[0, 8]
    occupied = sorted(set(bands.ravel()))
    assert result['pcgrad_task_count'] == len(occupied)
    nonzero = torch.linalg.vector_norm(gradients[0], dim=1) > 0.
    np.testing.assert_array_equal(nonzero.numpy(), np.array(occupied) == teacher_band)
    assert result['phase_correction_loss'] > 0.


def test_real_vcsel_rhs_is_unchanged_by_the_teacher_rephasing(tmp_path):
    from rl.rl_api.simulation import prepare_vcsel
    size = 4
    rng = np.random.default_rng(78)
    config = replace(run.CONFIG.to_backend(tmp_path), n_lasers=size)
    kappa = rng.uniform(.1, 2., (1, size, size))
    np.fill_diagonal(kappa[0], 0.)
    phi = rng.uniform(-np.pi, np.pi, kappa.shape)
    delta = np.array([[-1., -.2, .4, .8]])
    vcsel, _, _ = prepare_vcsel(kappa, phi, config, detunings_ghz=delta)
    vcsel.nd['kappa_ramp'] = 1.
    states = np.empty((3, 1, 3*size))
    states[:, :, 0::3] = rng.uniform(-.1, .1, (3, 1, size))
    states[:, :, 1::3] = rng.uniform(.1, 1., (3, 1, size))
    states[:, :, 2::3] = rng.uniform(-3., 3., (3, 1, size))
    beta = rng.uniform(-np.pi, np.pi, size)
    shifted = states.copy()
    shifted[:, :, 2::3] += beta
    corrected = wrap(phi+beta[None, None, :]-beta[None, :, None])
    old = vcsel.f_nd(*states, 0, phi, vcsel.nd)
    new = vcsel.f_nd(*shifted, 0, corrected, vcsel.nd)
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)


def test_workers_restore_locking_metrics_from_reversed_chunks(tmp_path):
    _, item = prepared_for(tmp_path)
    expected = item.pop('locking')
    item['results'] = []
    for start in range(0, 40, 8):
        result = dict(start_index=0, stop_index=2, candidate_start_index=start, candidate_stop_index=start+8)
        result.update({key:value[:, start:start+8] for key,value in expected.items()})
        item['results'].insert(0, result)
    collect_measurements(item)
    for key,value in expected.items():
        np.testing.assert_array_equal(item['locking'][key], value)


def test_new_training_path_verifies_teachers_and_keeps_validation_and_final_batch_unassisted(
        tmp_path, monkeypatch, fake_physics):
    from rl.rl_api import simulation, training
    schedule = replace(run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                       learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(run.CONFIG, sizes=(3,), schedules=schedule,
        optimization=replace(run.CONFIG.optimization, iterations=3, workers=1),
        validation=ValidationConfig(pairs_per_size=3))
    worker_calls = []
    original = simulation.simulate_phase_model_chunk
    def record(work, **options):
        worker_calls.append((work[4].shape[:2], options.get('collect_locking', False)))
        return original(work, **options)
    monkeypatch.setattr(simulation, 'simulate_phase_model_chunk', record)
    result = Experiment('phase-feedback-test', config=public, output_directory=tmp_path/'run').train(
        live_plot=False, verify_reference=False)
    assert sum(locking and shape[1] == 8 for shape,locking in worker_calls) == 15
    assert any(locking and shape[1] == 1 for shape,locking in worker_calls)
    assert any(not locking for _,locking in worker_calls)  # Fixed validation stays uncorrected.
    history = result.history
    assert sum(history['phase_correction_accepted']) > 0
    assert history['phase_correction_attempted'][-1] == 0
    assert history['phase_correction_loss'][-1] == 0.
    assert max(history['phase_correction_attempted']) <= 25
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert all(int(state['step']) == 2 for state in saved['optimizer_state_dict']['state'].values())
    assert saved['sampling_recipe']['phase_correction']['weight'] == 1.
    model = CouplingModel.load(result.checkpoint('final'))
    assert model.backend_config.encode_root
    assert model.backend_config.phase_parameterization == 'target_root_delay_residual'
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['phase_correction']['weight'] == 1.
    recorded = json.loads((result.directory/'current_history.json').read_text())
    assert len(recorded['history']['phase_correction_loss']) == 3


def test_mean_recipe_changes_only_correction_mode_and_separate_run_name(tmp_path):
    from rl.rl_api.experiments import train_mean_phase_feedback_1000 as mean_run
    old, new = asdict(run.CONFIG), asdict(mean_run.CONFIG)
    assert old['phase_correction'].pop('mode') == 'sampled_joint'
    assert new['phase_correction'].pop('mode') == 'mean_phase_only'
    assert old == new
    assert asdict(run.CONFIG.to_backend(tmp_path)) == asdict(mean_run.CONFIG.to_backend(tmp_path))
    assert run.RUN_NAME != mean_run.RUN_NAME
    summary = mean_run.CONFIG.summary()
    assert summary['correction_mean_simulations_max'] == 175
    assert summary['simulations_per_iteration_max'] == 1750
    assert summary['root_selection'] == 'uniform per topology'


def test_restricted_phase_step_preserves_all_other_weights_strengths_and_adam_state(tmp_path):
    from rl.rl_api.mean_phase_correction import mean_designs, phase_output_step
    policy, item = prepared_for(tmp_path)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.zero_grad()
    sum(p.square().sum() for p in policy.parameters()).backward()
    optimizer.step()  # Populate ordinary RL's Adam momentum before the phase-only step.
    means = mean_designs(policy, [item])
    means[0]['phase_teachers'] = [dict(row=0, column=1, accepted=True,
                                     phase=wrap(means[0]['phi'][0, 1]+.2))]
    before = {key:value.clone() for key,value in policy.state_dict().items()}
    import copy
    adam_before = copy.deepcopy(optimizer.state_dict())
    result = phase_output_step(policy, means, learning_rate=1e-4, weight=1., gradient_clip=1.)
    assert result['phase_correction_update_applied'] == 1
    assert result['phase_correction_loss'] < result['phase_correction_loss_before']
    for key,value in policy.state_dict().items():
        if key in {'edge_output.weight', 'edge_output.bias'}:
            torch.testing.assert_close(value[[0,2,3]], before[key][[0,2,3]], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    after = mean_designs(policy, [item])
    np.testing.assert_array_equal(after[0]['kappa'], means[0]['kappa'])
    for param,state in adam_before['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(optimizer.state_dict()['state'][param][key], value, rtol=0, atol=0)
    # A non-improving proposal is restored exactly, without moving shared weights.
    before = {key:value.clone() for key,value in policy.state_dict().items()}
    rejected = phase_output_step(policy, means, learning_rate=1e3, weight=1., gradient_clip=1.)
    assert rejected['phase_correction_update_applied'] == 0
    for key,value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_mean_teacher_rollouts_follow_rl_and_use_identical_post_update_strengths(tmp_path, monkeypatch, fake_physics):
    from rl.rl_api import mean_phase_correction as mean_module, training_batch
    from rl.rl_api.experiments import train_mean_phase_feedback_1000 as mean_run
    cfg = replace(mean_run.CONFIG.to_backend(tmp_path), training_n_lasers=(3,4), targets_per_batch=2)
    torch.manual_seed(7)
    policy = build_policy(cfg)
    from rl.rl_api.root_context import calibrate_budget_references
    cfg = calibrate_budget_references(policy, cfg)
    policy.config = cfg
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    state = dict(rl_done=False, mean_batches=[], mean_jobs=[], verification_jobs=[])
    original_update = training_batch.pcgrad_update
    def rl_update(*args, **options):
        assert 'phase_correction' not in options  # Correction is outside RL/PCGrad.
        result = original_update(*args, **options)
        with torch.no_grad():policy.budget_head[-1].bias.add_(.03)
        state['rl_done'] = True
        return result
    monkeypatch.setattr(training_batch, 'pcgrad_update', rl_update)
    original_means = mean_module.mean_designs
    def means(*args):
        assert state['rl_done']
        result = original_means(*args);state['mean_batches'].append(result)
        return result
    monkeypatch.setattr(mean_module, 'mean_designs', means)
    original_jobs = mean_module.mean_simulation_jobs
    def mean_jobs(items):
        result = original_jobs(items);state['mean_jobs'].extend(result)
        return result
    monkeypatch.setattr(mean_module, 'mean_simulation_jobs', mean_jobs)
    original_teachers = mean_module.prepare_teachers
    def teachers(items, *args, **options):
        result = original_teachers(items, *args, **options)
        state['verification_jobs'].extend(result[0])
        return result
    monkeypatch.setattr(mean_module, 'prepare_teachers', teachers)
    metrics = training_batch.run_training_batch(policy, optimizer, cfg, np.random.default_rng(9), 0,
        mask_sampler=sample_band_random_reachable_topologies, schedules=mean_run.CONFIG.schedules,
        phase_correction=mean_run.CONFIG.phase_correction)
    assert metrics['phase_correction_mean_simulations'] == 10
    assert metrics['phase_correction_accepted'] > 0
    assert metrics['phase_correction_strength_change_max'] == 0.
    assert metrics['phase_correction_loss'] <= metrics['phase_correction_loss_before']
    assert len(state['mean_jobs']) == 10
    before, after = state['mean_batches']
    for first,second in zip(before,after):
        np.testing.assert_array_equal(first['kappa'], second['kappa'])
    for group,_,_,job in state['verification_jobs']:
        selected = before[group]['phase_teachers']
        for local,index in enumerate(range(job[1],job[2])):
            teacher=selected[index]
            np.testing.assert_array_equal(job[4][local,0],
                before[group]['kappa'][teacher['row'],teacher['column']])


def test_mean_verification_rejects_improved_but_inaccurate_teachers(tmp_path):
    _, item = prepared_for(tmp_path)
    jobs, diagnostics = prepare_teachers([item], run.CONFIG.phase_correction)
    results = []
    for group,_,_,job in jobs:
        shape=(job[2]-job[1],1)
        results.append(dict(size_batch_index=group,start_index=job[1],stop_index=job[2],
            locking_finite=np.ones(shape,dtype=bool),locking_frequency_spread_mhz=np.zeros(shape),
            locking_phase_jitter_deg=np.zeros(shape),phase_rms_reward=np.ones(shape),
            phase_rms_success=np.zeros(shape,dtype=bool)))
    verify_teachers([item],results,run.CONFIG.phase_correction,diagnostics,require_phase_success=True)
    assert diagnostics['phase_correction_accepted'] == 0


def test_mean_mode_full_workflow_reloads_and_final_mean_batch_skips_extra_work(tmp_path, fake_physics):
    from rl.rl_api.experiments import train_mean_phase_feedback_1000 as mean_run
    schedule = replace(mean_run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                       learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(mean_run.CONFIG, sizes=(3,), schedules=schedule,
        optimization=replace(mean_run.CONFIG.optimization, iterations=3, workers=1),
        validation=ValidationConfig(pairs_per_size=2))
    result = Experiment('mean-phase-feedback-test', config=public, output_directory=tmp_path/'run').train(
        live_plot=False, verify_reference=False)
    assert result.history['phase_correction_mean_simulations'] == [25,25,0]
    assert result.history['phase_correction_strength_change_max'] == [0.,0.,0.]
    assert result.history['phase_correction_attempted'][-1] == 0
    assert np.isnan(result.history['phase_correction_locked_fraction'][-1])
    assert result.history['phase_correction_update_applied'][-1] == 0
    assert any(result.history['phase_correction_update_applied'])
    for before,after in zip(result.history['phase_correction_loss_before'],result.history['phase_correction_loss']):
        assert after <= before
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert all(int(state['step']) == 2 for state in saved['optimizer_state_dict']['state'].values())
    assert saved['sampling_recipe']['phase_correction']['mode'] == 'mean_phase_only'
    model = CouplingModel.load(result.checkpoint('final'))
    assert set(model.policy.state_dict()) == set(build_policy(public.to_backend(tmp_path)).state_dict())
    assert model.backend_config.encode_root
