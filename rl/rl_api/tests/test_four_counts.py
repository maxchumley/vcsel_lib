"""Restore the successful topology sampler without changing physical settings or schedules."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, ValidationConfig, CouplingModel
from rl.rl_api._compat import core, engine
from rl.rl_api.experiments import train_four_counts as run, train_fixed_counts as previous
from rl.rl_api.experiments import train_backup_sampling as working
from rl.rl_api.sampling import sample_pairs
from rl.rl_api import training, training_batch


@pytest.mark.parametrize('size', range(3,10))
def test_sampler_exactly_matches_last_working_masks_contexts_and_rng(size,tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    first, second = np.random.default_rng(45), np.random.default_rng(45)
    actual = sample_pairs(config,first,n_lasers=size)
    local = replace(config,n_lasers=size)
    targets = core.sample_target_phases(5,second,size,0.,config.structured_target_jitter_rad)
    detuning = core.sample_detuning_distributions(5,second,local,iteration=0)
    targets, detuning = core.sort_by_detuning(targets,detuning)
    expected = engine.sample_stratified_rooted_topologies(5,4,size,second,'cpu',fixed_star=True)
    np.testing.assert_array_equal(actual.target_phases_rad,targets)
    np.testing.assert_array_equal(actual.detunings_ghz,detuning)
    torch.testing.assert_close(actual.edge_masks,expected,rtol=0,atol=0)
    np.testing.assert_array_equal(np.sort(actual.edge_masks.sum(-1).numpy(),axis=1),
        np.tile(np.rint(np.linspace(size-1,size*(size-1),4)),(5,1)))
    assert first.random() == second.random()


def test_only_topology_population_changes_from_five_count_run(tmp_path):
    before, after = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert before['sampling'].pop('topologies_per_pair') == 5
    assert after['sampling'].pop('topologies_per_pair') == 4
    assert before['sampling'].pop('topology_sampling_mode') == 'band_fixed'
    assert after['sampling'].pop('topology_sampling_mode') == 'per_target'
    assert before == after
    assert run.CONFIG.sampling == working.CONFIG.sampling
    assert run.CONFIG.schedules == previous.CONFIG.schedules
    assert run.CONFIG.phys['phase_model'] == 'standard_lk'
    assert run.CONFIG.optimization.iterations == 3000
    assert run.START_TRAINING is True
    assert run.RUN_NAME not in {previous.RUN_NAME,working.RUN_NAME}
    assert run.RESTART_EXISTING_RUN == previous.RESTART_EXISTING_RUN
    assert run.LIVE_PLOT == previous.LIVE_PLOT
    assert run.VERIFY_REFERENCE == previous.VERIFY_REFERENCE
    assert run.OUTPUT_DIRECTORY == previous.OUTPUT_DIRECTORY
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1120
    assert summary['worker_jobs'] == 28
    assert summary['pcgrad_tasks'] == 16
    assert summary['sampled_masks_per_size'] == 20
    preview = Experiment(run.RUN_NAME,config=run.CONFIG,output_directory=tmp_path/'preview')
    assert not preview.directory.exists()


def test_working_five_bin_settings_are_preserved_except_phase_and_schedule():
    baseline, proposed = asdict(working.CONFIG), asdict(run.CONFIG)
    baseline['phys']['phase_model'] = 'standard_lk'
    baseline['schedules'] = proposed['schedules']
    assert baseline == proposed
    assert run.CONFIG.reward == working.CONFIG.reward
    assert run.CONFIG.schedules.ordinary_start == 1000
    assert run.CONFIG.schedules.ordinary_end == 1500
    assert run.CONFIG.schedules.wide_start == 1000
    assert run.CONFIG.schedules.wide_end == 1500
    assert run.CONFIG.schedules.precision_start == 1500
    assert run.CONFIG.schedules.precision_end == 2000
    assert run.CONFIG.schedules.learning_rate_decay_start == 1500
    assert run.CONFIG.schedules.learning_rate_decay_end == 2000


def test_full_batch_restores_16_tasks_and_passes_precision_schedule(tmp_path,monkeypatch,fake_physics):
    from rl.rl_api.simulation import simulate_indexed_chunk
    settings = replace(run.CONFIG,
        optimization=replace(run.CONFIG.optimization,iterations=1),
        validation=ValidationConfig(pairs_per_size=4))
    jobs_seen = []

    class Pool:
        def __init__(self,processes,initializer): assert processes == 14
        def imap_unordered(self,worker,jobs):
            jobs = list(jobs)
            if worker is simulate_indexed_chunk: jobs_seen.extend(jobs)
            return reversed([worker(job) for job in jobs])
        def close(self): pass
        def join(self): pass

    class Context:
        def Pool(self,**options): return Pool(**options)

    monkeypatch.setattr(training.mp,'get_context',lambda method:Context())
    original_batch = training_batch.run_training_batch

    def check_options(*args,**options):
        assert options['mask_sampler'] is engine.sample_stratified_rooted_topologies
        assert options['schedules'] == run.CONFIG.schedules
        return original_batch(*args,**options)

    monkeypatch.setattr(training_batch,'run_training_batch',check_options)
    original_combine = core.combine_task_gradients_pcgrad
    groups = []

    def check_weights(gradients,clip,rng,*,task_weights):
        groups.append(gradients.shape[0])
        expected = []
        for size in range(3,10):
            expected.extend([.25,.25,.5] if size in (4,7) else [.5,.5])
        np.testing.assert_allclose(task_weights,np.array(expected)/7)
        assert clip == 1.
        return original_combine(gradients,clip,rng,task_weights=task_weights)

    monkeypatch.setattr(core,'combine_task_gradients_pcgrad',check_weights)
    result = Experiment('four-count-test',config=settings,
                        output_directory=tmp_path/'run').train(live_plot=False)
    assert groups == [16]
    assert len(jobs_seen) == 28
    assert all(job[3][4].shape[:2] == (5,8) for job in jobs_seen)
    assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in jobs_seen) == 1120
    assert result.history['pcgrad_task_count'] == [16]
    saved = torch.load(result.checkpoint('current'),weights_only=False)
    assert saved['sampling_recipe']['sampler'] == 'backup_per_target'
    assert saved['sampling_recipe']['masks_per_pair'] == 4
    assert saved['sampling_recipe']['precision_schedule'] == asdict(run.CONFIG.schedules)
    assert saved['exploration_state']['maximum_log_std'] == .75
    assert saved['config']['phase_model'] == 'standard_lk'
    history = json.loads((result.directory/'current_history.json').read_text())
    assert history['sampling_recipe'] == saved['sampling_recipe']
    assert history['exploration_state'] == saved['exploration_state']


def test_four_count_precision_bounds_and_checkpoint_reload(tmp_path,fake_physics):
    from rl.rl_api.models import build_policy
    from rl.rl_api.checkpoints import save
    from rl.rl_api.exploration import exploration_state
    base = replace(run.CONFIG.to_backend(tmp_path),training_n_lasers=(3,),targets_per_batch=5)
    rng = core.set_random_seed(7)
    policy = build_policy(base)
    base = engine.calibrate_budget_references(policy,base)
    policy.config = base
    optimizer = torch.optim.Adam(policy.parameters(),lr=base.learning_rate)
    metrics = training_batch.run_training_batch(policy,optimizer,base,rng,1999,
        reward_scale_state={},mask_sampler=engine.sample_stratified_rooted_topologies,
        schedules=run.CONFIG.schedules)
    assert metrics['pcgrad_task_count'] == 2
    assert policy.minimum_log_std == -4.5
    assert policy.maximum_log_std == -3.
    assert policy.config.budget_exploration_end_std == .05
    path = tmp_path/'precision.pt'
    save(path,policy,optimizer,policy.config,2000,.9,exploration_state=exploration_state(policy))
    loaded = CouplingModel.load(path,device='cpu')
    assert loaded.policy.minimum_log_std == -4.5
    assert loaded.policy.maximum_log_std == -3.
    assert loaded.backend_config.phase_model == 'standard_lk'


def test_next_run_starts_from_one_cell_and_keeps_worker_guard(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment,'train',lambda self,**options:calls.append((self,options)))
    cells = read_cells('train_four_counts.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None',f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell,'<four-count-cell>','exec'),{'__name__':'__main__'})
    assert len(calls) == 1
    assert calls[0][0].config.sampling.topologies_per_pair == 4
    assert calls[0][1]['restart'] is True
    exec(compile(cell,'<worker-import>','exec'),{'__name__':'__mp_main__'})
    assert len(calls) == 1
    assert not (tmp_path/'cell').exists()
