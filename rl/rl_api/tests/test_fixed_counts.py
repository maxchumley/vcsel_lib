"""Only link-count selection changes in the next precision experiment."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api.experiments import train_fixed_counts as run, train_precision as previous
from rl.rl_api.sampling import sample_fixed_count_topologies, sample_pairs
from rl.rl_api import training, training_batch


COUNTS = {
    3: [2,3,4,5,6], 4: [3,5,8,10,12], 5: [4,8,12,16,20],
    6: [5,11,18,24,30], 7: [6,15,24,33,42],
    8: [7,19,32,44,56], 9: [8,24,40,56,72],
}


@pytest.mark.parametrize('size', range(3,10))
def test_counts_stay_fixed_but_wiring_changes_independently(size):
    rng = np.random.default_rng(31)
    masks = sample_fixed_count_topologies(5,5,size,rng,'cpu')
    again = sample_fixed_count_topologies(5,5,size,rng,'cpu')
    expected = np.tile(COUNTS[size], (5,1))
    for batch in (masks, again):
        counts = batch.sum(-1).numpy().astype(int)
        np.testing.assert_array_equal(counts, expected)
        np.testing.assert_array_equal(engine.topology_density_band_ids(counts,size),
                                      np.tile([0,0,1,2,2], (5,1)))
        assert np.all(engine.root_reach_fractions(batch.numpy(),size) == 1.)
        _, sources = core.directed_link_indices(size)
        assert batch[:,:,sources==0].bool().all()
    assert any(not torch.equal(masks[0,1:4], row[1:4]) for row in masks[1:])
    assert not torch.equal(masks[:,1:4], again[:,1:4])


def test_recipe_changes_only_count_selection_and_output_name(tmp_path):
    before, after = asdict(previous.CONFIG), asdict(run.CONFIG)
    assert before['sampling'].pop('topology_sampling_mode') == 'band_random'
    assert after['sampling'].pop('topology_sampling_mode') == 'band_fixed'
    assert before == after
    assert run.RUN_NAME != previous.RUN_NAME
    assert run.START_TRAINING == previous.START_TRAINING == True
    assert run.RESTART_EXISTING_RUN == previous.RESTART_EXISTING_RUN
    assert run.LIVE_PLOT == previous.LIVE_PLOT
    assert run.VERIFY_REFERENCE == previous.VERIFY_REFERENCE
    assert run.OUTPUT_DIRECTORY == previous.OUTPUT_DIRECTORY
    # The sampler is an API choice; every numerical backend setting matches.
    assert asdict(run.CONFIG.to_backend(tmp_path)) == asdict(previous.CONFIG.to_backend(tmp_path))
    old_summary, new_summary = previous.CONFIG.summary(), run.CONFIG.summary()
    old_summary.pop('topology_sampling')
    assert new_summary.pop('topology_sampling') == 'band_fixed'
    assert old_summary == new_summary
    assert new_summary['simulations_per_iteration'] == 1400
    assert new_summary['pcgrad_tasks'] == 21
    assert new_summary['worker_jobs'] == 35


def test_pair_preview_uses_fixed_counts_and_preserves_five_detuning_bins(tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    pair = sample_pairs(config,np.random.default_rng(7),n_lasers=6,
                        topology_sampling_mode=run.CONFIG.sampling.topology_sampling_mode)
    np.testing.assert_array_equal(pair.edge_masks.sum(-1).numpy(), np.tile(COUNTS[6],(5,1)))
    np.testing.assert_array_equal(np.sort((np.ptp(pair.detunings_ghz,axis=1)/.8).astype(int)),
                                  np.arange(5))


def test_full_iteration_keeps_jobs_tasks_weights_and_checkpoint_metadata(tmp_path,monkeypatch,fake_physics):
    from rl.rl_api.simulation import simulate_indexed_chunk
    settings = replace(run.CONFIG,
        optimization=replace(run.CONFIG.optimization,iterations=1),
        validation=ValidationConfig(pairs_per_size=4))
    jobs_seen = []

    class Pool:
        def __init__(self, processes, initializer):
            assert processes == 14
        def imap_unordered(self, worker, jobs):
            jobs = list(jobs)
            if worker is simulate_indexed_chunk:
                jobs_seen.extend(jobs)
            return reversed([worker(job) for job in jobs])
        def close(self): pass
        def join(self): pass

    class Context:
        def Pool(self, **options): return Pool(**options)

    monkeypatch.setattr(training.mp,'get_context',lambda method:Context())
    original = core.combine_task_gradients_pcgrad
    calls = []

    def check_weights(gradients, clip, rng, *, task_weights):
        calls.append(gradients.shape[0])
        np.testing.assert_allclose(task_weights, np.tile([.4,.2,.4],7)/7)
        assert clip == 1.
        return original(gradients,clip,rng,task_weights=task_weights)

    monkeypatch.setattr(core,'combine_task_gradients_pcgrad',check_weights)
    result = Experiment('fixed-count-test',config=settings,
                        output_directory=tmp_path/'run').train(live_plot=False)
    assert calls == [21]
    assert len(jobs_seen) == 35
    assert all(job[3][4].shape[:2] == (5,8) for job in jobs_seen)
    assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in jobs_seen) == 1400
    assert result.history['pcgrad_task_count'] == [21]
    assert result.model.backend_config.phase_model == 'standard_lk'
    data = torch.load(result.checkpoint('current'),weights_only=False)
    assert data['sampling_recipe']['sampler'] == 'band_fixed'
    assert data['sampling_recipe']['precision_schedule'] == asdict(run.CONFIG.schedules)
    assert data['exploration_state']['maximum_log_std'] == .75
    history = json.loads((result.directory/'current_history.json').read_text())
    assert history['sampling_recipe'] == data['sampling_recipe']
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['sampling']['topology_sampling_mode'] == 'band_fixed'


def test_fixed_batch_keeps_precision_stage_and_saved_bounds(tmp_path,fake_physics):
    from rl.rl_api.models import build_policy
    from rl.rl_api.checkpoints import save
    from rl.rl_api.exploration import exploration_state
    from rl.rl_api import CouplingModel
    base = replace(run.CONFIG.to_backend(tmp_path),training_n_lasers=(3,),targets_per_batch=5)
    rng = core.set_random_seed(7)
    policy = build_policy(base)
    base = engine.calibrate_budget_references(policy,base)
    policy.config = base
    optimizer = torch.optim.Adam(policy.parameters(),lr=base.learning_rate)
    metrics = training_batch.run_training_batch(policy,optimizer,base,rng,1999,
        reward_scale_state={},mask_sampler=sample_fixed_count_topologies,schedules=run.CONFIG.schedules)
    assert metrics['pcgrad_task_count'] == 3
    assert policy.minimum_log_std == -4.5
    assert policy.maximum_log_std == -3.
    assert policy.config.budget_exploration_end_std == .05
    path = tmp_path/'precision.pt'
    save(path,policy,optimizer,policy.config,2000,.9,exploration_state=exploration_state(policy))
    loaded = CouplingModel.load(path,device='cpu')
    assert loaded.policy.minimum_log_std == -4.5
    assert loaded.policy.maximum_log_std == -3.
    assert loaded.backend_config.phase_model == 'standard_lk'


def test_next_run_starts_from_one_cell_with_main_process_guard(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment,'train',lambda self,**options:calls.append((self,options)))
    cells = read_cells('train_fixed_counts.py')
    assert len(cells) == 1
    cell = cells[0].replace('OUTPUT_DIRECTORY = None',f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell,'<fixed-count-cell>','exec'),{'__name__':'__main__'})
    assert len(calls) == 1
    assert calls[0][0].config.sampling.topology_sampling_mode == 'band_fixed'
    assert calls[0][1]['restart'] is True
    exec(compile(cell,'<worker-import>','exec'),{'__name__':'__mp_main__'})
    assert len(calls) == 1
    assert not (tmp_path/'cell').exists()
