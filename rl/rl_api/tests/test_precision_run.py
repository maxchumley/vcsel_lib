"""Band coverage, precision schedules and checkpoint distributions for the next run."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel, ValidationConfig
from rl.rl_api._compat import core, engine, make_backend_config, exported_settings
from rl.rl_api.checkpoints import save
from rl.rl_api.exploration import refinement_config, exploration_state
from rl.rl_api.experiments import train_precision as run
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import sample_band_random_topologies, sample_pairs
from rl.rl_api import training_batch, training
from rl.rl_api.tests.test_parity import assert_equal


@pytest.mark.parametrize('size', range(3,10))
def test_each_pair_has_endpoints_and_one_random_count_in_every_band(size):
    rng = np.random.default_rng(31)
    masks = sample_band_random_topologies(5, 5, size, rng, 'cpu')
    assert masks.shape == (5, 5, size*(size-1))
    counts = masks.sum(-1).numpy().astype(int)
    np.testing.assert_array_equal(engine.topology_density_band_ids(counts,size),
                                  np.tile([0,0,1,2,2], (5,1)))
    np.testing.assert_array_equal(counts[:,0], np.repeat(size-1,5))
    np.testing.assert_array_equal(counts[:,-1], np.repeat(size*(size-1),5))
    assert np.all(counts[:,1:] > counts[:,:-1])
    assert np.all(engine.root_reach_fractions(masks.numpy(),size) == 1.)
    _, sources = core.directed_link_indices(size)
    assert masks[:,:,sources==0].bool().all()
    assert any(not torch.equal(masks[0,1:4],row[1:4]) for row in masks[1:])
    again = sample_band_random_topologies(5,5,size,rng,'cpu')
    assert not torch.equal(masks[:,1:4],again[:,1:4])


def test_random_counts_reach_previously_omitted_intermediate_counts():
    rng = np.random.default_rng(902)
    counts = set()
    for _ in range(64):
        masks = sample_band_random_topologies(5,5,6,rng,'cpu')
        counts.update(masks[:,1:4].sum(-1).numpy().astype(int).ravel())
    assert counts == set(range(6,30))  # Includes 10; excludes both endpoints.


def test_pair_helper_uses_new_sampler_and_five_detuning_bins(tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    pair = sample_pairs(config,np.random.default_rng(9),n_lasers=6,
                        topology_sampling_mode='band_random')
    assert pair.edge_masks.shape == (5,5,30)
    np.testing.assert_array_equal(np.sort((np.ptp(pair.detunings_ghz,axis=1)/.8).astype(int)),
                                  np.arange(5))


@pytest.mark.parametrize('printed,low,high,budget',[
    (1,-1.5,.75,.35), (999,-1.5,.75,.35), (1500,-3.,-1.5,.35),
    (1750,-3.75,-2.25,.20), (2000,-4.5,-3.,.05), (2500,-4.5,-3.,.05),
    (3000,-4.5,-3.,.05)])
def test_two_stage_sd_and_budget_schedule(printed,low,high,budget,tmp_path):
    base = run.CONFIG.to_backend(tmp_path)
    current = refinement_config(base,run.CONFIG.schedules,printed-1)
    assert core.minimum_log_std_at(printed-1,current) == pytest.approx(low)
    assert core.maximum_log_std_at(printed-1,current) == pytest.approx(high)
    assert current.budget_exploration_end_std == pytest.approx(budget)
    assert base.minimum_log_std == -3. and base.budget_exploration_end_std == .35
    if printed >= 1500:
        assert engine.phase_exploration_at(printed-1,current) == 0.


@pytest.mark.parametrize('printed,rate',[(1,1e-4),(1499,1e-4),(1500,1e-4),
    (1750,math.sqrt(1e-4*1e-5)),(2000,1e-5),(2500,1e-5),(3000,1e-5)])
def test_lr_finishes_decay_at_2000_then_holds(printed,rate,tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))],lr=config.learning_rate)
    training.set_learning_rate(optimizer,config,printed-1,
                              decay_end=run.CONFIG.schedules.learning_rate_decay_end)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(rate)


def test_new_recipe_keeps_model_reward_and_selects_standard_lk(tmp_path):
    from rl.rl_api.experiments import train_backup_sampling as previous
    assert run.CONFIG.model == previous.CONFIG.model
    assert run.CONFIG.reward == previous.CONFIG.reward
    assert {key:value for key,value in run.CONFIG.phys.items() if key != 'phase_model'} == previous.CONFIG.phys
    assert run.CONFIG.phys['phase_model'] == 'standard_lk'
    assert run.CONFIG.optimization.iterations == 3000
    assert run.CONFIG.sampling.reuse_masks_across_pairs is False
    assert run.START_TRAINING is True
    assert run.RUN_NAME != previous.RUN_NAME
    preview = Experiment(run.RUN_NAME,config=run.CONFIG,output_directory=tmp_path/'preview')
    assert not preview.directory.exists()
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['pcgrad_tasks'] == 21
    assert summary['worker_jobs'] == 35
    assert summary['sampled_masks_per_size'] == 25
    torch.manual_seed(7)
    first = build_policy(run.CONFIG.to_backend(tmp_path))
    torch.manual_seed(7)
    second = build_policy(previous.CONFIG.to_backend(tmp_path))
    assert_equal(first.state_dict(),second.state_dict())
    assert sum(p.numel() for p in first.parameters()) == 127304


def test_physical_band_batch_matches_original_when_given_original_sampler(tmp_path,fake_physics):
    settings = exported_settings(run.CONFIG.to_backend(tmp_path))
    settings.pop('phase_model')  # Original workers support the historical default only.
    config = replace(make_backend_config(settings),training_n_lasers=(3,6,9),targets_per_batch=15)
    states = []
    for reference in (True,False):
        rng = core.set_random_seed(7)
        policy = build_policy(config)
        current = engine.calibrate_budget_references(policy,config)
        policy.config = current
        optimizer = torch.optim.Adam(policy.parameters(),lr=current.learning_rate)
        variance = {}
        for iteration in (0,999,1499):
            if reference:
                engine.run_training_batch(policy,optimizer,current,rng,iteration,reward_scale_state=variance)
            else:
                training_batch.run_training_batch(policy,optimizer,current,rng,iteration,
                    reward_scale_state=variance,mask_sampler=engine.sample_stratified_rooted_topologies)
        states.append((policy.state_dict(),optimizer.state_dict(),variance,
                       rng.bit_generator.state,torch.get_rng_state()))
    assert_equal(states[0],states[1])


def test_full_batch_has_35_vectorized_jobs_and_21_five_pair_tasks(tmp_path,monkeypatch,fake_physics):
    from rl.rl_api.simulation import simulate_indexed_chunk
    settings = replace(run.CONFIG,
        optimization=replace(run.CONFIG.optimization,iterations=1),
        validation=ValidationConfig(pairs_per_size=4))
    batches = []
    class Pool:
        def __init__(self,processes,initializer):
            assert processes == 14
        def imap_unordered(self,worker,jobs):
            if worker is simulate_indexed_chunk:
                batches.extend(jobs)
            return reversed([worker(job) for job in jobs])
        def close(self): pass
        def join(self): pass
    class Context:
        def Pool(self, **options):
            return Pool(**options)
    monkeypatch.setattr(training.mp,'get_context',lambda method:Context())
    original = training_batch.pcgrad_update
    def check_tasks(*args,**kwargs):
        result = original(*args,**kwargs)
        assert result['pcgrad_task_context_counts'] == [5]*21
        return result
    monkeypatch.setattr(training_batch,'pcgrad_update',check_tasks)
    result = Experiment('new-test',config=settings,output_directory=tmp_path/'new').train(live_plot=False)
    assert len(batches) == 35
    assert all(job[3][4].shape[:2] == (5,8) for job in batches)
    assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in batches) == 1400
    assert result.history['pcgrad_task_count'] == [21]
    assert result.history['rooted_fraction'] == [1.]
    data = torch.load(result.checkpoint('current'),weights_only=False)
    assert data['config']['phase_model'] == 'standard_lk'
    assert data['sampling_recipe']['sampler'] == 'band_random'
    assert data['sampling_recipe']['precision_schedule']['precision_end'] == 2000
    assert data['exploration_state']['maximum_log_std'] == .75
    history = json.loads((result.directory/'current_history.json').read_text())
    assert history['exploration_state'] == data['exploration_state']


def test_precision_updates_and_stochastic_checkpoint_reload(tmp_path,fake_physics):
    base = replace(run.CONFIG.to_backend(tmp_path),training_n_lasers=(3,),targets_per_batch=5)
    rng = core.set_random_seed(7)
    policy = build_policy(base)
    base = engine.calibrate_budget_references(policy,base)
    policy.config = base
    optimizer = torch.optim.Adam(policy.parameters(),lr=base.learning_rate)
    for iteration in (1499,1749,1999):
        training_batch.run_training_batch(policy,optimizer,base,rng,iteration,
            reward_scale_state={},mask_sampler=sample_band_random_topologies,
            schedules=run.CONFIG.schedules)
        path = tmp_path/f'iteration_{iteration+1}.pt'
        save(path,policy,optimizer,policy.config,iteration+1,.9,
             exploration_state=exploration_state(policy))
        loaded = CouplingModel.load(path,device='cpu')
        assert loaded.backend_config.phase_model == 'standard_lk'
        assert_equal(policy.state_dict(),loaded.policy.state_dict())
        assert loaded.policy.minimum_log_std == policy.minimum_log_std
        assert loaded.policy.maximum_log_std == policy.maximum_log_std
        features = engine.encode_context(np.array([[0.,-.7,1.3]]),np.array([[-.5,0.,.5]]),policy)
        topology = torch.ones((1,6))
        with torch.inference_mode():
            _, first_sd = policy.budget_parameters(features,topology_masks=topology)
            _, loaded_sd = loaded.policy.budget_parameters(features,topology_masks=topology)
        torch.testing.assert_close(first_sd,loaded_sd,rtol=0,atol=0)
    assert policy.minimum_log_std == -4.5
    assert policy.maximum_log_std == -3.
    assert policy.config.budget_exploration_end_std == .05
    assert .05*math.exp(-.25) <= first_sd.exp().item() <= .05*math.exp(.25)


def test_precision_starts_from_one_cell_and_worker_import_does_not_restart(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment,'train',lambda self,**options:calls.append((self,options)))
    scope = {'__name__':'__main__'}
    cells = read_cells('train_precision.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None',
                           f'OUTPUT_DIRECTORY = {str(tmp_path/"cell-test")!r}')
    exec(compile(cell,'<precision-cell>','exec'),scope)
    assert len(calls) == 1
    assert not (tmp_path/'cell-test').exists()
    assert '__file__' not in scope
    assert calls[0][0].config.optimization.iterations == 3000
    assert calls[0][1]['restart'] is True
    worker_scope = {'__name__':'__mp_main__'}
    exec(compile(cell,'<worker-import>','exec'),worker_scope)
    assert len(calls) == 1
    preview_cell = cell.replace('START_TRAINING = True','START_TRAINING = False')
    exec(compile(preview_cell,'<preview-cell>','exec'),{'__name__':'__main__'})
    assert len(calls) == 1
