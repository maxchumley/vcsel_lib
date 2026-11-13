"""Sampled roots survive action expansion, validation, save/load and inspection."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment, Topology, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api import evaluation, sampling, training
from rl.rl_api.experiments import train_mirrored_reachable_1000 as previous
from rl.rl_api.experiments import train_mirrored_root_delay_1000 as run
from rl.rl_api.exploration import refinement_config
from rl.rl_api.models import build_policy
from rl.rl_api.phase_reference import decode_samples
from rl.rl_api.training_batch import prepare_size
from rl.rl_api.tests.test_random_reachable_run import reachable_from


def policies(tmp_path, size=5):
    result = []
    for public in (previous.CONFIG, run.CONFIG):
        config = replace(public.to_backend(tmp_path), n_lasers=size)
        torch.manual_seed(7)
        result.append(build_policy(config))
    return result


def assert_phase_translation(original, changed, masks, detunings, roots, config):
    """Independently check physical phases on directed enabled edges."""
    r, s = core.directed_link_indices(config.n_lasers)
    tau = int(config.delay_seconds/config.time_step_seconds)*config.time_step_seconds
    gap = np.take_along_axis(detunings, roots, axis=1)-detunings.min(axis=1, keepdims=True)
    expected = -2*np.pi*1e9*tau*gap
    error = np.angle(np.exp(1j*(changed-original-expected[..., None, None])))
    np.testing.assert_allclose(error[..., r, s][masks.astype(bool)], 0., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(changed[..., r, s][~masks.astype(bool)],
                                  original[..., r, s][~masks.astype(bool)])
    np.testing.assert_array_equal(changed[roots == 0], original[roots == 0])


def test_only_reference_setting_changes_and_initial_weights_match(tmp_path):
    old, new = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert old['model'].pop('phase_parameterization') == 'target_delay_residual'
    assert new['model'].pop('phase_parameterization') == 'target_root_delay_residual'
    assert old == new
    old, new = asdict(previous.CONFIG.to_backend(tmp_path)), asdict(run.CONFIG.to_backend(tmp_path))
    old.pop('phase_parameterization'); new.pop('phase_parameterization')
    assert old == new
    a, b = policies(tmp_path)
    assert sum(p.numel() for p in b.parameters()) == 127304
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    for key in ('START_TRAINING', 'RESTART_EXISTING_RUN', 'LIVE_PLOT', 'VERIFY_REFERENCE', 'OUTPUT_DIRECTORY'):
        assert getattr(run, key) == getattr(previous, key)
    assert run.RUN_NAME != previous.RUN_NAME
    assert run.CONFIG.summary()['simulations_per_iteration'] == 1400
    with pytest.raises(ValueError, match='require random_reachable'):
        replace(run.CONFIG, topology=replace(run.CONFIG.topology, backbone='rooted_star'))


@pytest.mark.parametrize('size', [3, 9])
@pytest.mark.parametrize('sampler', [sampling.sample_random_reachable_topologies,
                                   sampling.sample_band_random_reachable_topologies])
def test_retained_roots_are_actual_draws_and_do_not_change_masks_or_rng(size, sampler, monkeypatch):
    original = sampling.random_reachable_mask
    recorded = []
    def record(*args, **kwargs):
        mask, root = original(*args, **kwargs)
        recorded.append(root)
        return mask, root
    monkeypatch.setattr(sampling, 'random_reachable_mask', record)
    a, b = np.random.default_rng(713), np.random.default_rng(713)
    old = sampler(8, 5, size, a, 'cpu')
    recorded.clear()
    masks, roots = sampler(8, 5, size, b, 'cpu', return_roots=True)
    torch.testing.assert_close(old, masks, rtol=0, atol=0)
    np.testing.assert_array_equal(a.random(10), b.random(10))
    np.testing.assert_array_equal(roots.ravel(), recorded)
    assert np.any(roots != 0)
    for mask, root in zip(masks.numpy().reshape(-1, size*(size-1)), roots.ravel()):
        assert reachable_from(mask, int(root), size) == set(range(size))


def test_expanded_training_actions_keep_their_topology_root_and_scores(tmp_path):
    a, b = policies(tmp_path)
    prepared = []
    for policy, public in zip((a, b), (previous.CONFIG, run.CONFIG)):
        config = refinement_config(policy.config, public.schedules, 249)
        policy.config = config
        torch.manual_seed(137)
        prepared.append(prepare_size(policy, config, np.random.default_rng(581), 5, 5, 249,
                                     sampling.sample_band_random_reachable_topologies))
    old, new = prepared
    for key in ('targets', 'detunings', 'kappa', 'counts'):
        np.testing.assert_array_equal(old[key], new[key])
    for key in ('features', 'gates', 'log_prob'):
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert np.all(new['roots'].reshape(5, 5, 8) == new['roots'].reshape(5, 5, 8)[:, :, :1])
    assert np.any(new['roots'] != 0)
    pairs = sampling.sample_pairs(b.config, np.random.default_rng(581), n_lasers=5, iteration=249)
    np.testing.assert_array_equal(new['roots'][:, ::8], pairs.root_indices)
    assert_phase_translation(old['phi'], new['phi'], new['gates'].numpy(),
                             new['detunings'], new['roots'], b.config)
    # Fixed physical translations must not alter latent Gaussian score gradients.
    score_gradients = []
    weights = torch.linspace(-1., 1., old['log_prob'].numel()).reshape_as(old['log_prob'])
    for policy, item in zip((a, b), prepared):
        grad = torch.autograd.grad((item['log_prob']*weights).sum(),
                                   list(policy.parameters()), allow_unused=True)
        score_gradients.append(grad)
    for x, y in zip(*score_gradients):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('conjugate', [False, True])
def test_zero_residual_cancels_the_root_delay_not_the_minimum_delay(tmp_path, conjugate):
    _, policy = policies(tmp_path, size=4)
    config = replace(policy.config, allow_global_phase_conjugate=conjugate)
    policy.config = config
    target = np.array([[0., -.7, 1.3, 2.1], [0., .2, -1.9, .9]])
    detuning = np.array([[-1.7, -.2, .3, 1.6], [-1.2, -.8, .5, 1.5]])
    roots = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    with torch.no_grad():
        policy.edge_output.weight[1].zero_(); policy.edge_output.bias[1].zero_()
    features = engine.encode_context(target, detuning, policy)
    repeated = features.repeat_interleave(4, dim=0)
    masks = torch.ones((2, 4, 12))
    actions = policy.parameters_for_edges(repeated, topology_masks=masks.reshape(8, 12))[0].reshape(2, 4, 24)
    _, phi, _, _, _ = decode_samples(actions, masks, config, detunings_ghz=detuning, root_indices=roots)
    encoded = np.arctan2(features[..., 1].numpy(), features[..., 0].numpy())
    receivers, sources = core.directed_link_indices(4)
    tau = int(config.delay_seconds/config.time_step_seconds)*config.time_step_seconds
    root_f = np.take_along_axis(detuning, roots, axis=1)
    argument = (encoded[:, None, sources]-encoded[:, None, receivers]
                -2*np.pi*root_f[..., None]*1e9*tau-phi[..., receivers, sources])
    np.testing.assert_allclose(np.angle(np.exp(1j*argument)), 0., atol=3e-6, rtol=0)
    with pytest.raises(ValueError, match='one valid root'):
        decode_samples(actions, masks, config, detunings_ghz=detuning)


def test_validation_retains_sampled_roots_with_identical_contexts_and_strengths(tmp_path, monkeypatch):
    a, b = policies(tmp_path)
    rng = np.random.default_rng(11)
    target, detuning = core.sort_by_detuning(rng.uniform(-3., 3., (8, 5)), rng.uniform(-2., 2., (8, 5)))
    recorded = []
    def simulate(targets, kappa, phi, config, **options):
        recorded.append((kappa, phi))
        z = np.zeros(kappa.shape[:2])
        return dict(phase_reward=z, phase_rms_reward=z, mean_phase_rms_error_deg=z,
                    worst_phase_rms_error_deg=z, phase_rms_success=z)
    monkeypatch.setattr(evaluation, 'simulate_phase_model_candidates', simulate)
    for policy in (a, b):
        policy.config = engine.calibrate_budget_references(policy, policy.config)
        evaluation.evaluate_phase_model_policy(policy, target, detuning, policy.config, seed=127)
    masks, roots = sampling.sample_random_reachable_topologies(1, 8, 5,
        np.random.default_rng(127), 'cpu', return_roots=True)
    np.testing.assert_array_equal(recorded[0][0], recorded[1][0])
    assert_phase_translation(recorded[0][1], recorded[1][1], masks.numpy().reshape(8, 1, 20),
                             detuning, roots.reshape(8, 1), b.config)


def test_pipeline_checkpoint_and_loaded_inspection_share_root_reference(tmp_path, monkeypatch, fake_physics):
    schedule = replace(run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                       learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(run.CONFIG, schedules=schedule,
        optimization=replace(run.CONFIG.optimization, iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    from rl.rl_api.simulation import simulate_indexed_chunk
    batches = []
    class Pool:
        def __init__(self, processes, initializer): assert processes == 14
        def imap_unordered(self, worker, jobs):
            jobs = list(jobs)
            if worker is simulate_indexed_chunk:
                assert len(jobs) == 35
                assert all(job[3][4].shape[:2] == (5, 8) for job in jobs)
                batches.append(jobs)
            return reversed([worker(job) for job in jobs])
        def close(self): pass
        def join(self): pass
    class ProcessContext:
        def Pool(self, **kwargs): return Pool(**kwargs)
    monkeypatch.setattr(training.mp, 'get_context', lambda method: ProcessContext())
    result = Experiment('root-delay-test', config=public, output_directory=tmp_path/'run').train(
        live_plot=False, verify_reference=False)
    assert len(batches) == 3
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert saved['config']['phase_parameterization'] == 'target_root_delay_residual'
    assert saved['sampling_recipe']['phase_delay_reference'] == 'sampled root detuning'
    assert saved['config']['exploration_std_scale'] == 0.
    assert all(int(v['step']) == 2 for v in saved['optimizer_state_dict']['state'].values())
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert pickle.loads(pickle.dumps(loaded.backend_config)).phase_parameterization == 'target_root_delay_residual'
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['model']['phase_parameterization'] == 'target_root_delay_residual'
    context = Context([0., -.4, 1.2, .8], [2., -1., 1., .5])
    first = result.model.propose(context, count=12, seed=17)
    second = loaded.propose(context, count=12, seed=17)
    np.testing.assert_array_equal(first.coupling_phases_rad, second.coupling_phases_rad)
    old_config = replace(loaded.backend_config, phase_parameterization='target_delay_residual')
    old_policy = build_policy(old_config); old_policy.load_state_dict(loaded.policy.state_dict())
    baseline = CouplingModel(old_policy, old_config).propose(context, count=12, seed=17)
    np.testing.assert_array_equal(baseline.coupling_strengths_per_ns, second.coupling_strengths_per_ns)
    masks, roots = sampling.sample_random_reachable_topologies(1, 12, 4,
        np.random.default_rng(17), 'cpu', return_roots=True)
    np.testing.assert_array_equal(second.raw['root_indices'], roots[0])
    assert_phase_translation(baseline.coupling_phases_rad[None], second.coupling_phases_rad[None],
        masks.numpy(), second.context.detunings_ghz[None], roots, replace(old_config, n_lasers=4))
    centered = second.context.detunings_ghz-second.context.detunings_ghz.mean()
    np.testing.assert_array_equal(second.raw['phase_reference_detunings_ghz'], centered[roots[0]])
    assert np.isfinite(second.evaluate().rms_reward).all()
    # Manual topology is in original caller order; its unique root is original
    # laser 0, which becomes sorted laser 3. Do not fall back to laser 0 sorted.
    manual = np.zeros((4, 4), dtype=bool); manual[0, 1:] = True
    design = loaded.propose(context, topology=Topology.from_source_receiver(manual), count=2, seed=17)
    np.testing.assert_array_equal(design.raw['root_indices'], [3, 3])
    order = np.argsort(context.detunings_ghz)
    np.testing.assert_array_equal(design.topologies[0], manual.T[order][:, order])


def test_single_cell_sets_up_separate_run_without_starting_on_worker_import(tmp_path, monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment, 'train', lambda self, **options: calls.append((self, options)))
    cells = read_cells('train_mirrored_root_delay_1000.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None', f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell, '<root-delay-cell>', 'exec'), {'__name__': '__main__'})
    assert len(calls) == 1 and calls[0][0].name == run.RUN_NAME
    assert calls[0][0].config.model.phase_parameterization == 'target_root_delay_residual'
    exec(compile(cell, '<root-delay-worker>', 'exec'), {'__name__': '__mp_main__'})
    assert len(calls) == 1 and not (tmp_path/'cell').exists()
