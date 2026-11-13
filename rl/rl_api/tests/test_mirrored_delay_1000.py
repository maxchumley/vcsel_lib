"""Only the known delay-phase offset changes the mirrored 1k experiment."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import CouplingModel, Experiment, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api.experiments import train_mirrored_delay_1000 as run, train_mirrored_1000 as previous
from rl.rl_api.exploration import refinement_config, sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.models.policy import TargetDelayResidualPhasePolicy, delay_phase_offsets
from rl.rl_api.sampling import sample_band_random_topologies
from rl.rl_api.tests.test_mirrored_1000 import features_and_masks, gaussian_parameters
from rl.rl_api import training, training_batch


def paired_policies(directory, size=3, printed=1, conjugate=True):
    result = []
    for public in (previous.CONFIG, run.CONFIG):
        config = replace(public.to_backend(directory), n_lasers=size,
                         allow_global_phase_conjugate=conjugate)
        torch.manual_seed(7)
        policy = build_policy(config)
        policy.config = refinement_config(config, public.schedules, printed-1)
        policy.minimum_log_std = core.minimum_log_std_at(printed-1, policy.config)
        policy.maximum_log_std = core.maximum_log_std_at(printed-1, policy.config)
        result.append(policy)
    return result


def test_training_configuration_diff_is_only_the_phase_convention(tmp_path):
    old, new = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert old['model'].pop('phase_parameterization') == 'target_residual'
    assert new['model'].pop('phase_parameterization') == 'target_delay_residual'
    assert old == new
    old, new = asdict(previous.CONFIG.to_backend(tmp_path)), asdict(run.CONFIG.to_backend(tmp_path))
    old.pop('phase_parameterization')
    assert new.pop('phase_parameterization') == 'target_delay_residual'
    assert old == new
    for key in ('START_TRAINING', 'RESTART_EXISTING_RUN', 'LIVE_PLOT', 'OUTPUT_DIRECTORY'):
        assert getattr(run, key) == getattr(previous, key)
    assert run.RUN_NAME != previous.RUN_NAME
    # Launch-only exception: the reviewed integrator intentionally differs
    # from the historical source fingerprint. Keep that manifest unchanged.
    assert run.VERIFY_REFERENCE is False
    assert run.CONFIG.optimization.iterations == 1000
    assert run.CONFIG.summary()['simulations_per_iteration'] == 1400
    assert run.CONFIG.summary()['pcgrad_tasks'] == 21
    assert not Experiment('preview', config=run.CONFIG, output_directory=tmp_path/'preview').directory.exists()


def test_identical_initial_weights_architecture_and_rng(tmp_path):
    states = []
    for public in (previous.CONFIG, run.CONFIG):
        torch.manual_seed(7)
        policy = build_policy(public.to_backend(tmp_path))
        states.append((policy.state_dict(), torch.get_rng_state().clone()))
        assert sum(p.numel() for p in policy.parameters()) == 127304
    assert states[0][0].keys() == states[1][0].keys()
    for name in states[0][0]:
        torch.testing.assert_close(states[0][0][name], states[1][0][name], rtol=0, atol=0)
    assert torch.equal(states[0][1], states[1][1])


@pytest.mark.parametrize('size', [3, 6, 9])
@pytest.mark.parametrize('conjugate', [False, True])
def test_delay_offset_matches_asymmetric_centered_detunings(size, conjugate, tmp_path):
    old, new = paired_policies(tmp_path, size=size, conjugate=conjugate)
    rng = np.random.default_rng(271 + size)
    target = rng.uniform(-np.pi, np.pi, (2, size))
    detuning = np.sort(rng.uniform(-2., 2., (2, size)), axis=1)
    detuning -= detuning.mean(axis=1, keepdims=True)
    features = engine.encode_context(target, detuning, new)
    masks = sample_band_random_topologies(2, 5, size, rng, 'cpu').reshape(-1, size*(size-1))
    repeated = features.repeat_interleave(5, dim=0)
    a = old.parameters_for_edges(repeated, topology_masks=masks, return_pooled=True)
    b = new.parameters_for_edges(repeated, topology_masks=masks, return_pooled=True)
    edges = size*(size-1)
    tau = int(new.config.delay_seconds/new.config.time_step_seconds)*new.config.time_step_seconds
    expected = np.repeat(-2*np.pi*detuning.min(axis=1)*1e9*tau, 5)
    np.testing.assert_allclose(delay_phase_offsets(repeated, new.config).detach().numpy()[:, 0], expected, atol=3e-6, rtol=0)
    np.testing.assert_allclose((b[0][:, edges:]-a[0][:, edges:]).detach().numpy(),
                               np.broadcast_to(expected[:, None], (10, edges)), atol=3e-6, rtol=0)
    torch.testing.assert_close(a[0][:, :edges], b[0][:, :edges], rtol=0, atol=0)
    for x, y in zip(a[1:], b[1:]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(old.budget_parameters(repeated, pooled_states=a[3]),
                    new.budget_parameters(repeated, pooled_states=b[3])):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_zero_residual_cancels_the_delayed_reference_phase(tmp_path):
    _, policy = paired_policies(tmp_path, size=4, conjugate=False)
    target = np.array([[0., -.7, 1.3, 2.1]])
    detuning = np.array([[-1.7, -.2, .3, 1.6]])  # mean zero, asymmetric
    with torch.no_grad():
        policy.edge_output.weight[1].zero_()
        policy.edge_output.bias[1].zero_()
    features = engine.encode_context(target, detuning, policy)
    masks = torch.ones((1, 12))
    actions = policy.parameters_for_edges(features, topology_masks=masks)[0][:, None]
    _, phase, _, _, _ = engine.decode_samples(actions, masks[:, None], policy.config)
    receivers, sources = core.directed_link_indices(4)
    tau = int(policy.config.delay_seconds/policy.config.time_step_seconds)*policy.config.time_step_seconds
    delayed_source = target[:, sources]-2*np.pi*detuning.min()*1e9*tau
    argument = delayed_source-target[:, receivers]-phase[:, 0, receivers, sources]
    np.testing.assert_allclose(np.exp(1j*argument), 1., atol=3e-6, rtol=0)
    zero = engine.encode_context(target, np.zeros_like(detuning), policy)
    torch.testing.assert_close(delay_phase_offsets(zero, policy.config), torch.zeros((1, 1)), rtol=0, atol=0)


@pytest.mark.parametrize('printed', [1, 598, 999, 1000])
def test_mirrored_sampling_scores_and_endpoint_use_delay_corrected_mean(tmp_path, printed):
    old, new = paired_policies(tmp_path, printed=printed)
    features, masks = features_and_masks(new)
    torch.manual_seed(42)
    a, ab, _ = sample_annealed_candidates(old, features, masks, mirrored=True)
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    b, bb, score = sample_annealed_candidates(new, features, masks, mirrored=True)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(a[:, :, :6], b[:, :, :6], rtol=0, atol=0)
    torch.testing.assert_close(ab, bb, rtol=0, atol=0)
    offset = delay_phase_offsets(features, new.config)[:, :, None]
    torch.testing.assert_close(b[:, :, 6:], a[:, :, 6:]+offset, rtol=0, atol=2e-6)
    mean, std, bm, bs = gaussian_parameters(new, features, masks)
    if printed == 1000:
        torch.testing.assert_close(b, mean, rtol=0, atol=0)
        assert torch.count_nonzero(score) == 0
        return
    expected = (torch.distributions.Normal(mean, std).log_prob(b)*torch.cat((masks, masks), dim=-1)).sum(-1)
    expected += torch.distributions.Normal(bm, bs).log_prob(bb)
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    weights = torch.linspace(-1., 1., score.numel()).reshape_as(score)
    (score*weights).mean().backward()
    assert torch.isfinite(new.edge_output.weight.grad).all()
    assert new.edge_output.weight.grad[1].abs().max() > 0


def test_first_batch_changes_only_phases_and_preserves_all_random_draws(tmp_path):
    batches = []
    for public in (previous.CONFIG, run.CONFIG):
        config = public.to_backend(tmp_path)
        rng = core.set_random_seed(7)
        policy = build_policy(config)
        config = engine.calibrate_budget_references(policy, config)
        policy.config = refinement_config(config, public.schedules, 0)
        prepared = [training_batch.prepare_size(policy, policy.config, rng, size, 5, 0,
                    sample_band_random_topologies) for size in config.training_n_lasers]
        batches.append((prepared, rng.bit_generator.state, torch.get_rng_state().clone(), config.coupling_budget_reference_by_m))
    assert batches[0][1] == batches[1][1]
    assert torch.equal(batches[0][2], batches[1][2])
    assert batches[0][3] == batches[1][3]
    for a, b in zip(batches[0][0], batches[1][0]):
        for key in ('targets', 'detunings', 'kappa', 'counts', 'rho', 'shortfall'):
            np.testing.assert_array_equal(a[key], b[key])
        torch.testing.assert_close(a['gates'], b['gates'], rtol=0, atol=0)
        tau = int(b['config'].delay_seconds/b['config'].time_step_seconds)*b['config'].time_step_seconds
        expected = -2*np.pi*b['detunings'].min(axis=1)*1e9*tau
        receivers, sources = core.directed_link_indices(b['n_lasers'])
        difference = b['phi'][:, :, receivers, sources]-a['phi'][:, :, receivers, sources]-expected[:, None, None]
        np.testing.assert_allclose(np.angle(np.exp(1j*difference)), 0., atol=4e-6, rtol=0)


def test_full_pipeline_validation_checkpoint_and_inference_keep_new_convention(tmp_path, monkeypatch, fake_physics):
    schedules = replace(run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                        learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(run.CONFIG, schedules=schedules,
        optimization=replace(run.CONFIG.optimization, iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    from rl.rl_api.simulation import simulate_indexed_chunk
    calls = []
    class Pool:
        def __init__(self, processes, initializer): assert processes == 14
        def imap_unordered(self, worker, jobs):
            jobs = list(jobs)
            if worker is simulate_indexed_chunk:
                assert len(jobs) == 35
                assert all(j[3][4].shape[:2] == (5, 8) for j in jobs)
                calls.append(jobs)
            assert all(job[3][-1].phase_parameterization == 'target_delay_residual'
                       for job in jobs) if worker is simulate_indexed_chunk else all(job[-1].phase_parameterization == 'target_delay_residual' for job in jobs)
            return reversed([worker(job) for job in jobs])
        def close(self): pass
        def join(self): pass
    class Context:
        def Pool(self, **options): return Pool(**options)
    monkeypatch.setattr(training.mp, 'get_context', lambda method: Context())
    result = Experiment('delay-test', config=public, output_directory=tmp_path/'run').train(
        live_plot=False, verify_reference=run.VERIFY_REFERENCE)
    assert len(calls) == 3
    assert result.history['pcgrad_task_count'] == [21]*3
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert saved['config']['phase_parameterization'] == 'target_delay_residual'
    assert saved['config']['action_sampling'] == 'mirrored'
    assert saved['config']['reward_baseline'] == 'pair_excluded'
    assert saved['config']['exploration_std_scale'] == 0.
    assert all(int(v['step']) == 2 for v in saved['optimizer_state_dict']['state'].values())
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert isinstance(loaded.policy, TargetDelayResidualPhasePolicy)
    assert type(pickle.loads(pickle.dumps(loaded.backend_config))) is type(loaded.backend_config)
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['model']['phase_parameterization'] == 'target_delay_residual'
    options = dict(target_phases_rad=[0., -.7, 1.3], detunings_ghz=[-1.5, -.5, 2.], count=2, seed=23)
    for mode in ('mean', 'sample'):
        first, second = result.model.propose(**options, continuous_actions=mode), loaded.propose(**options, continuous_actions=mode)
        np.testing.assert_array_equal(first.coupling_phases_rad, second.coupling_phases_rad)
        np.testing.assert_array_equal(first.coupling_strengths_per_ns, second.coupling_strengths_per_ns)
        assert np.isfinite(second.evaluate().rms_reward).all()


def test_single_cell_starts_new_run_and_worker_import_does_not(tmp_path, monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment, 'train', lambda self, **options:calls.append((self, options)))
    cells = read_cells('train_mirrored_delay_1000.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None', f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell, '<delay-cell>', 'exec'), {'__name__':'__main__'})
    assert len(calls) == 1
    assert calls[0][0].config.model.phase_parameterization == 'target_delay_residual'
    assert calls[0][1]['verify_reference'] is False
    exec(compile(cell, '<worker-import>', 'exec'), {'__name__':'__mp_main__'})
    assert len(calls) == 1
    assert not (tmp_path/'cell').exists()
