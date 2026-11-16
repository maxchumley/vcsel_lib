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
    assert new.pop('topology_root_index') is None
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


def test_fixed_laser1_growth_keeps_arbitrary_paths_optional_edges_and_reproducibility():
    for size in (3, 6, 9):
        options = dict(n_pairs=24, n_groups=5, n_lasers=size, device='cpu',
                       root_index=0, return_roots=True)
        masks, roots = sample_band_random_reachable_topologies(rng=np.random.default_rng(71), **options)
        replay, replay_roots = sample_band_random_reachable_topologies(rng=np.random.default_rng(71), **options)
        torch.testing.assert_close(masks, replay, rtol=0, atol=0)
        np.testing.assert_array_equal(roots, replay_roots)
        assert np.all(roots == 0) and torch.all(masks[:, -1] == 1)
        receiver, source = core.directed_link_indices(size)
        flat = masks[:, :4].reshape(-1, size*(size-1)).numpy().astype(bool)
        assert any(not row[source == 0].all() for row in flat)
        assert flat[:, receiver == 0].any()  # Feedback into laser 1 remains eligible.
        assert all(reachable_from(row, 0, size) == set(range(size)) for row in flat)


def test_fixed_root_configuration_rejects_an_absent_root():
    from rl.rl_api import TopologyConfig
    with pytest.raises(ValueError, match='nonnegative integer'):
        TopologyConfig(backbone='random_reachable', root_index=-1)
    with pytest.raises(ValueError, match='every trained array size'):
        replace(run.CONFIG, topology=TopologyConfig(backbone='random_reachable', root_index=3))


def test_fixed_bank_recipe_preserves_training_and_splits_validation_budget(tmp_path):
    from rl.rl_api.experiments import train_mirrored_laser1_reachable_1000 as varying
    from rl.rl_api.experiments import train_mirrored_laser1_fixed_topologies_1000 as fixed
    a,b=asdict(varying.CONFIG),asdict(fixed.CONFIG)
    assert a['sampling'].pop('fixed_topology_bank_seed') is None
    assert b['sampling'].pop('fixed_topology_bank_seed') == 60007
    assert a['validation'].pop('pairs_per_size') == 64
    assert b['validation'].pop('pairs_per_size') == 32
    assert a == b
    old_backend,new_backend=asdict(varying.CONFIG.to_backend(tmp_path)),asdict(fixed.CONFIG.to_backend(tmp_path))
    assert old_backend.pop('held_out_target_count') == 64
    assert new_backend.pop('held_out_target_count') == 32
    assert old_backend == new_backend
    assert 2*len(fixed.CONFIG.sizes)*fixed.CONFIG.validation.pairs_per_size == 448
    assert fixed.CONFIG.summary()['validation_cases_per_group'] == 224
    assert fixed.CONFIG.summary()['validation_simulations_per_check'] == 448
    assert fixed.INITIAL_CHECKPOINT == varying.INITIAL_CHECKPOINT
    assert not fixed.CONFIG.phase_correction.enabled
    assert fixed.CONFIG.summary()['distinct_training_topologies_total'] == 35
    assert fixed.CONFIG.summary()['simulations_per_iteration'] == 1400
    with pytest.raises(ValueError,match='nonnegative integer'):
        replace(fixed.CONFIG.sampling,fixed_topology_bank_seed=-1)


def test_fixed_bank_graphs_rng_and_validation_partitions(tmp_path):
    from rl.rl_api.topology_bank import FixedTopologyBank
    from rl.rl_api.experiments import train_mirrored_laser1_fixed_topologies_1000 as fixed
    config=fixed.CONFIG.to_backend(tmp_path)
    bank=FixedTopologyBank(config,60007)
    assert bank.metadata() == FixedTopologyBank(config,60007).metadata()
    for size in (3,5,9):
        a,b=np.random.default_rng(91),np.random.default_rng(91)
        masks=bank(5,5,size,a,'cpu',root_index=0)
        sample_band_random_reachable_topologies(5,5,size,b,'cpu',root_index=0)
        np.testing.assert_array_equal(a.random(10),b.random(10))
        assert all(torch.equal(row,masks[0]) for row in masks)
        again=bank(5,5,size,a,'cpu',root_index=0)
        torch.testing.assert_close(masks,again,rtol=0,atol=0)
        keys={row.tobytes() for row in masks[0].numpy()}
        assert len(keys)==5
        assert len(np.unique(engine.topology_density_band_ids(masks[0].sum(1).numpy(),size)))==3
        assert all(reachable_from(row,0,size)==set(range(size)) for row in masks[0].numpy())
        seen,roots=bank.validation_masks(size,64,seen=True,seed=17)
        unseen,new_roots=bank.validation_masks(size,64,seen=False,seed=17)
        assert np.all(roots==0) and np.all(new_roots==0)
        assert all(row.tobytes() in keys for row in seen.numpy())
        assert all(row.tobytes() not in keys for row in unseen.numpy())
        replay,_=bank.validation_masks(size,64,seen=False,seed=17)
        torch.testing.assert_close(unseen,replay,rtol=0,atol=0)


def test_fixed_bank_training_records_seen_and_unseen_validation(tmp_path,monkeypatch,fake_physics):
    from rl.rl_api.experiments import train_mirrored_laser1_fixed_topologies_1000 as fixed
    from rl.rl_api.checkpoints import save
    from rl.rl_api.root_context import calibrate_budget_references
    from rl.rl_api.topology_bank import FixedTopologyBank
    source_config=replace(previous.CONFIG,sizes=(3,4)).to_backend(tmp_path/'source')
    source_policy=build_policy(source_config)
    source_config=calibrate_budget_references(source_policy,source_config)
    checkpoint=tmp_path/'source.pt'
    save(checkpoint,source_policy,torch.optim.Adam(source_policy.parameters()),source_config,1000,.99)
    public=replace(fixed.CONFIG,sizes=(3,4),
        sampling=replace(fixed.CONFIG.sampling,pairs_per_size=1,pairs_per_job=1),
        optimization=replace(fixed.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(fixed.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=ValidationConfig(pairs_per_size=3,plot_use_tex=False))
    recorded=[]
    original=FixedTopologyBank.__call__
    def capture(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        recorded.append((args[2],result.detach().clone()))
        return result
    monkeypatch.setattr(FixedTopologyBank,'__call__',capture)
    groups=[]
    original_evaluate=training.evaluate_bank
    def capture_evaluation(policy,config,bank,iteration,pool,**options):
        groups.append(options['topologies'])
        return original_evaluate(policy,config,bank,iteration,pool,**options)
    monkeypatch.setattr(training,'evaluate_bank',capture_evaluation)
    result=Experiment('fixed-bank-test',config=public,output_directory=tmp_path/'run').train(
        initial_checkpoint=checkpoint,live_plot=False,verify_reference=False)
    assert len(recorded)==4 and len(groups)==4
    for size in (3,4):
        first=[m for n,m in recorded if n==size]
        torch.testing.assert_close(first[0],first[1],rtol=0,atol=0)
        for unseen,seen in zip(groups[::2],groups[1::2]):
            known={row.tobytes() for row in first[0][0].numpy()}
            assert all(row.tobytes() in known for row in seen[size][0].numpy())
            assert all(row.tobytes() not in known for row in unseen[size][0].numpy())
    for key in ['seen_topology_held_out_reward','seen_topology_held_out_phase_rms_success_fraction']:
        assert len(result.history[key])==2
    saved=torch.load(result.checkpoint('final'),weights_only=False)
    assert saved['sampling_recipe']['mode']=='fixed_graph_bank'
    metadata=saved['sampling_recipe']['fixed_topology_bank']
    assert metadata==json.loads((result.directory/'fixed_topology_bank.json').read_text())
    loaded=CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert loaded.metadata['sampling_recipe']['fixed_topology_bank']==metadata
    assert not any(k.startswith('phase_correction') for k in result.history)
    figure=result.plot(show=False)
    assert len(figure.axes)==3
    figure.canvas.draw()
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_warm_start_keeps_weights_calibration_and_phase_convention(tmp_path):
    from rl.rl_api.checkpoints import initialize_from_checkpoint, save
    from rl.rl_api.models import build_policy
    from rl.rl_api.root_context import calibrate_budget_references
    from rl.rl_api.experiments import train_mirrored_laser1_reachable_1000 as next_run
    source_config = replace(previous.CONFIG, sizes=(3, 4)).to_backend(tmp_path/'source')
    source_policy = build_policy(source_config)
    source_config = calibrate_budget_references(source_policy, source_config)
    source_policy.config = source_config
    with torch.no_grad():
        source_policy.edge_output.bias[1].add_(.3)
    source_config = replace(source_config, exploration_std_scale=0.)
    path = tmp_path/'source.pt'
    save(path, source_policy, torch.optim.Adam(source_policy.parameters()), source_config, 1000, .99)
    target_config = replace(next_run.CONFIG, sizes=(3, 4)).to_backend(tmp_path/'target')
    target_policy = build_policy(target_config)
    config, metadata = initialize_from_checkpoint(target_policy, target_config, path)
    for name, value in source_policy.state_dict().items():
        torch.testing.assert_close(target_policy.state_dict()[name], value, rtol=0, atol=0)
    assert config.coupling_budget_reference_by_m == source_config.coupling_budget_reference_by_m
    assert config.exploration_std_scale == 1. and config.topology_root_index == 0
    assert metadata['source_iteration'] == 1000
    source_model = CouplingModel(source_policy, source_config)
    target_model = CouplingModel(target_policy, config)
    context = Context([0., .4, -.7], [1., -.5, .3])
    topology = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    a = source_model.propose(context, topology=topology)
    b = target_model.propose(context, topology=topology)
    np.testing.assert_array_equal(a.coupling_strengths_per_ns, b.coupling_strengths_per_ns)
    np.testing.assert_array_equal(a.coupling_phases_rad, b.coupling_phases_rad)
    incompatible = replace(target_config, phase_parameterization='target_residual')
    with pytest.raises(ValueError, match='phase_parameterization'):
        initialize_from_checkpoint(build_policy(incompatible), incompatible, path)


def test_fixed_root_warm_start_training_validation_and_loaded_inspection(tmp_path, monkeypatch, fake_physics):
    from rl.rl_api import training_batch
    from rl.rl_api.checkpoints import save
    from rl.rl_api.models import build_policy
    from rl.rl_api.root_context import calibrate_budget_references
    from rl.rl_api.experiments import train_mirrored_laser1_reachable_1000 as next_run
    import hashlib
    source_config = replace(previous.CONFIG, sizes=(3, 4)).to_backend(tmp_path/'source')
    source_policy = build_policy(source_config)
    source_config = calibrate_budget_references(source_policy, source_config)
    path = tmp_path/'source.pt'
    save(path, source_policy, torch.optim.Adam(source_policy.parameters()), source_config, 1000, .99)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    public = replace(next_run.CONFIG, sizes=(3, 4),
        sampling=replace(next_run.CONFIG.sampling, pairs_per_size=1, pairs_per_job=1),
        optimization=replace(next_run.CONFIG.optimization, iterations=2, workers=1),
        schedules=replace(next_run.CONFIG.schedules, ordinary_end=2, wide_end=2),
        validation=ValidationConfig(pairs_per_size=3))
    original_batch = training_batch.run_training_batch
    seen = []
    def check_start(policy, optimizer, config, *args, **kwargs):
        if not seen:
            assert not optimizer.state
            assert config.coupling_budget_reference_by_m == source_config.coupling_budget_reference_by_m
            for name, value in source_policy.state_dict().items():
                torch.testing.assert_close(policy.state_dict()[name], value, rtol=0, atol=0)
        seen.append(config.topology_root_index)
        return original_batch(policy, optimizer, config, *args, **kwargs)
    monkeypatch.setattr(training_batch, 'run_training_batch', check_start)
    validation_options = []
    original_sampler = sampling.sample_random_reachable_topologies
    def record_sampler(*args, **kwargs):
        validation_options.append(kwargs['root_index'])
        return original_sampler(*args, **kwargs)
    monkeypatch.setattr(sampling, 'sample_random_reachable_topologies', record_sampler)
    result = Experiment('fixed-root-warm', config=public, output_directory=tmp_path/'new').train(
        live_plot=False, verify_reference=False, initial_checkpoint=path)
    assert seen == [0, 0] and validation_options == [0]*4
    assert len(fake_physics) > 0
    assert not any(k.startswith('phase_correction') for k in result.history)
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['initialization']['sha256'] == digest
    saved = torch.load(result.checkpoint('final'), weights_only=False)
    assert saved['sampling_recipe']['root_selection'] == 'fixed laser 1 in detuning order'
    assert saved['sampling_recipe']['initialization']['sha256'] == digest
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert loaded.backend_config.topology_root_index == 0
    designs = loaded.propose(Context([0., .3, -.2], [1., -.5, .1]), count=20, seed=3)
    assert validation_options[-1] == 0
    for mask in designs.raw['gates'].astype(bool):
        assert reachable_from(mask, 0, 3) == {0, 1, 2}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


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
