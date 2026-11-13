"""Random reachability growth, unchanged mirrored recipe, and checkpoint reuse."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api.experiments import train_mirrored_reachable_1000 as run
from rl.rl_api.experiments import train_mirrored_delay_1000 as previous
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import (random_reachable_mask, sample_band_random_reachable_topologies,
                                sample_random_reachable_topologies, sample_pairs)
from rl.rl_api import sampling, training


def reachable_from(mask, root, size):
    """Independent traversal in source-row adjacency convention."""
    receiver, source = core.directed_link_indices(size)
    graph = np.zeros((size, size), dtype=bool)
    graph[source, receiver] = mask
    visited, pending = {root}, [root]
    while pending:
        node = pending.pop()
        for neighbor in np.flatnonzero(graph[node]):
            if int(neighbor) not in visited:
                visited.add(int(neighbor)); pending.append(int(neighbor))
    return visited


def test_configuration_and_initial_weights_change_only_topology(tmp_path):
    old, new = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert old['topology'].pop('backbone') == 'rooted_star'
    assert new['topology'].pop('backbone') == 'random_reachable'
    assert old == new
    old, new = asdict(previous.CONFIG.to_backend(tmp_path)), asdict(run.CONFIG.to_backend(tmp_path))
    assert old.pop('topology_star_backbone') is True
    assert new.pop('topology_star_backbone') is False
    assert new.pop('topology_random_reachable') is True
    assert old == new
    states = []
    for public in (previous.CONFIG, run.CONFIG):
        torch.manual_seed(7)
        policy = build_policy(public.to_backend(tmp_path))
        states.append((policy.state_dict(), torch.get_rng_state().clone()))
    for key in states[0][0]:
        torch.testing.assert_close(states[0][0][key], states[1][0][key], rtol=0, atol=0)
    assert torch.equal(states[0][1], states[1][1])
    for key in ('START_TRAINING', 'RESTART_EXISTING_RUN', 'LIVE_PLOT', 'VERIFY_REFERENCE', 'OUTPUT_DIRECTORY'):
        assert getattr(run, key) == getattr(previous, key)
    assert run.RUN_NAME != previous.RUN_NAME
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['worker_jobs'] == 35 and summary['pcgrad_tasks'] == 21
    assert summary['pcgrad_tasks_are_upper_bound']
    assert not Experiment('reachable-preview', config=run.CONFIG, output_directory=tmp_path/'preview').directory.exists()


def test_growth_keeps_cycles_unreached_edges_and_root_feedback_and_stops_at_reachability():
    receiver, source = core.directed_link_indices(4)
    prefix = [(0, 1), (1, 0), (3, 2), (2, 3), (3, 0)]  # (source, receiver)
    indices = [int(np.flatnonzero((source == s) & (receiver == r))[0]) for s, r in prefix]
    order = np.r_[indices, [i for i in range(12) if i not in indices]]
    class ScriptedRNG:
        def integers(self, high): return 2
        def permutation(self, count): return order.copy()
    mask, root = random_reachable_mask(4, ScriptedRNG())
    assert root == 2 and mask.sum() == 5
    np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(indices))
    assert reachable_from(mask, root, 4) == set(range(4))
    before_last = mask.copy(); before_last[indices[-1]] = False
    assert reachable_from(before_last, root, 4) == {2, 3}
    extra, _ = random_reachable_mask(4, ScriptedRNG(), minimum_links=8)
    assert extra.sum() == 8 and np.all(extra[mask])


@pytest.mark.parametrize('size', [3, 5, 9])
def test_random_roots_reach_all_lasers_without_forced_star(size):
    rng = np.random.default_rng(101)
    roots, masks = [], []
    for _ in range(150):
        mask, root = random_reachable_mask(size, rng)
        roots.append(root); masks.append(mask)
        assert reachable_from(mask, root, size) == set(range(size))
    assert set(roots) == set(range(size))
    masks = np.array(masks)
    _, sources = core.directed_link_indices(size)
    assert np.any(~masks[:, sources == 0].all(axis=1))
    assert np.all(masks.any(axis=0)) and np.all((~masks).any(axis=0))
    assert np.any(masks.sum(axis=1) > size-1)  # Graph growth is not a hidden tree sampler.


@pytest.mark.parametrize('size', [3, 6, 9])
def test_banded_optional_links_and_sampler_reproducibility(size, tmp_path):
    kwargs = dict(n_pairs=8, n_groups=5, n_lasers=size, device='cpu', fixed_star=False)
    first = sample_band_random_reachable_topologies(rng=np.random.default_rng(71), **kwargs)
    second = sample_band_random_reachable_topologies(rng=np.random.default_rng(71), **kwargs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (8, 5, size*(size-1))
    assert torch.all(first[:, -1] == 1)
    assert np.all(engine.root_reach_fractions(first.numpy(), size) == 1.)
    assert any(not torch.equal(first[0, 1:4], pair[1:4]) for pair in first[1:])
    # Some intermediate masks retain feedback into every possible source;
    # no individual directed edge is mandatory across these random graphs.
    assert torch.all(first[:, :4].reshape(-1, size*(size-1)).any(dim=0))
    assert torch.all((first[:, :4] == 0).reshape(-1, size*(size-1)).any(dim=0))
    config = run.CONFIG.to_backend(tmp_path)
    a = sample_pairs(config, np.random.default_rng(12), n_lasers=size)
    b = sample_pairs(config, np.random.default_rng(12), n_lasers=size, topology_sampling_mode='band_random')
    torch.testing.assert_close(a.edge_masks, b.edge_masks, rtol=0, atol=0)
    assert np.all(engine.root_reach_fractions(a.edge_masks.numpy(), size) == 1.)


def test_pipeline_uses_growth_for_training_validation_and_loaded_inspection(tmp_path, monkeypatch, fake_physics):
    schedule = replace(run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                       learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(run.CONFIG, schedules=schedule,
        optimization=replace(run.CONFIG.optimization, iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    sampled = []
    original = sampling.sample_band_random_reachable_topologies
    def record_sampler(*args, **kwargs):
        masks = original(*args, **kwargs)
        sampled.append(masks.clone())
        return masks
    monkeypatch.setattr(sampling, 'sample_band_random_reachable_topologies', record_sampler)
    validation_calls = []
    original_generic = sampling.sample_random_reachable_topologies
    def record_generic(*args, **kwargs):
        validation_calls.append(args[:3])
        return original_generic(*args, **kwargs)
    monkeypatch.setattr(sampling, 'sample_random_reachable_topologies', record_generic)
    batches = []
    from rl.rl_api.simulation import simulate_indexed_chunk
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
    result = Experiment('reachable-test', config=public, output_directory=tmp_path/'run').train(
        live_plot=False, verify_reference=False)
    assert len(batches) == 3 and len(sampled) == 21
    assert len(validation_calls) == 14  # Seven sizes at iterations 1 and 3.
    assert all(1 <= count <= 21 for count in result.history['pcgrad_task_count'])
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert saved['config']['topology_random_reachable'] is True
    assert saved['config']['topology_star_backbone'] is False
    assert saved['config']['phase_parameterization'] == 'target_delay_residual'
    assert saved['config']['exploration_std_scale'] == 0.
    assert saved['sampling_recipe']['backbone'] == 'random_reachable'
    assert saved['sampling_recipe']['ordinary_action_pairs'] == 3
    assert saved['sampling_recipe']['wide_action_pairs'] == 1
    assert all(int(v['step']) == 2 for v in saved['optimizer_state_dict']['state'].values())
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert pickle.loads(pickle.dumps(loaded.backend_config)).topology_random_reachable
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['topology']['backbone'] == 'random_reachable'
    context = Context([0., -.4, 1.2, .8], [1., -2., .5, -.5])
    options = dict(context=context, count=12, seed=17)
    first = result.model.propose(**options)
    second = loaded.propose(**options)
    np.testing.assert_array_equal(first.coupling_strengths_per_ns, second.coupling_strengths_per_ns)
    np.testing.assert_array_equal(first.coupling_phases_rad, second.coupling_phases_rad)
    expected = original_generic(1, 12, 4, np.random.default_rng(17), 'cpu')
    receiver, source = core.directed_link_indices(4)
    np.testing.assert_array_equal(second.topologies[:, receiver, source], expected[0].numpy())
    assert not second.topologies[:, 1:, 0].all()
    assert np.isfinite(second.evaluate().rms_reward).all()
    # Explicit source-row masks keep their original meaning and override generation.
    from rl.rl_api import Topology
    supplied = np.zeros((4, 4), dtype=bool); supplied[2, [0, 1, 3]] = True
    manual = loaded.propose(context, topology=Topology.from_source_receiver(supplied), count=2, seed=17)
    order = np.argsort(context.detunings_ghz, kind='stable')
    np.testing.assert_array_equal(manual.topologies[0], supplied.T[order][:, order])


def test_single_cell_starts_separate_run_and_worker_import_does_not(tmp_path, monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment, 'train', lambda self, **options: calls.append((self, options)))
    cells = read_cells('train_mirrored_reachable_1000.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None', f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell, '<reachable-cell>', 'exec'), {'__name__': '__main__'})
    assert len(calls) == 1 and calls[0][0].config.topology.backbone == 'random_reachable'
    assert calls[0][0].name == run.RUN_NAME and calls[0][1]['verify_reference'] is False
    exec(compile(cell, '<reachable-worker>', 'exec'), {'__name__': '__mp_main__'})
    assert len(calls) == 1 and not (tmp_path/'cell').exists()
