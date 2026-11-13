"""Check the new recipe against BACKUP without running physical training."""
from dataclasses import asdict, fields, replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, ValidationConfig
from rl.rl_api._compat import REPOSITORY_DIR, core, engine
from rl.rl_api.experiments import train_backup_sampling, train_current
from rl.rl_api.sampling import sample_pairs
from rl.rl_api.tests.test_parity import assert_equal


@pytest.fixture(scope="module")
def backup():
    """Import the preserved BACKUP sampler fixture, not local backup files."""
    name = "rl.rl_api.tests._backup_reference"
    path = Path(__file__).parent / "reference/bernoulli_sparse.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


@pytest.mark.parametrize("size", range(3, 10))
def test_independent_masks_match_backup_and_cover_all_five_span_bins(size, backup, tmp_path):
    config = train_backup_sampling.CONFIG.to_backend(tmp_path)
    first, second = np.random.default_rng(45), np.random.default_rng(45)
    actual = sample_pairs(config, first, n_lasers=size)
    local = replace(config, n_lasers=size)
    targets = core.sample_target_phases(5, second, size, 0., config.structured_target_jitter_rad)
    detunings = core.sample_detuning_distributions(5, second, local, iteration=0)
    targets, detunings = core.sort_by_detuning(targets, detunings)
    expected = backup.sample_stratified_rooted_topologies(
        5, 4, size, second, torch.device("cpu"), fixed_star=True)
    np.testing.assert_array_equal(actual.target_phases_rad, targets)
    np.testing.assert_array_equal(actual.detunings_ghz, detunings)
    torch.testing.assert_close(actual.edge_masks, expected, rtol=0, atol=0)
    assert first.random() == second.random()
    # A separately randomized mask row for each pair; identical star/dense
    # endpoints are expected, while the intermediate graphs can differ.
    if size >= 4:
        assert any(not torch.equal(actual.edge_masks[0], row) for row in actual.edge_masks[1:])
    counts = actual.edge_masks.sum(-1).numpy()
    expected_counts = np.rint(np.linspace(size-1, size*(size-1), 4))
    np.testing.assert_array_equal(np.sort(counts, axis=1), np.tile(expected_counts, (5, 1)))
    spans = np.ptp(actual.detunings_ghz, axis=1)
    np.testing.assert_array_equal(np.sort((spans/.8).astype(int)), np.arange(5))


def test_new_recipe_keeps_rms_and_current_schedule_without_curricula(tmp_path, monkeypatch):
    public = train_backup_sampling.CONFIG
    config = public.to_backend(tmp_path)
    assert public.schedules == train_current.CONFIG.schedules
    assert public.reward == train_current.CONFIG.reward
    assert config.phase_reward_mode == "wrapped_rms"
    assert config.training_iterations == 3000
    assert config.topology_sampling_mode == "per_target"
    assert config.pcgrad_task_grouping == "array_size_topology_density"
    assert config.detuning_curriculum_iterations == 0
    assert config.topology_curriculum_start_iteration is None
    assert config.topology_curriculum_end_iteration is None
    assert config.actions_per_topology == 8
    assert config.phase_exploration_fraction == .25
    summary = public.summary()
    assert summary['simulations_per_iteration'] == 1120
    assert summary['worker_jobs'] == 28
    assert summary['pcgrad_tasks'] == 16  # Empty physical bands are skipped.
    assert summary['sampled_masks_per_size'] == 20
    # The user may have enabled this run since setup; preview a temporary
    # output without imposing a value on their editable start flag.
    monkeypatch.setattr(train_backup_sampling, 'OUTPUT_DIRECTORY', tmp_path/'preview')
    experiment = Experiment(train_backup_sampling.RUN_NAME, vcsel=train_backup_sampling.vcsel,
                            config=train_backup_sampling.CONFIG,
                            output_directory=train_backup_sampling.OUTPUT_DIRECTORY)
    assert not experiment.directory.exists()


def test_backup_sampler_and_optimizer_match_across_exploration_stages(backup, tmp_path, fake_physics):
    """With cosine selected just for this test, reproduce BACKUP's whole batch.

    The production run uses RMS. This isolates the restored sampler, physical
    density grouping, advantage centering, clipping, PCGrad and Adam against
    BACKUP rather than comparing only names or a second copy of the new code.
    """
    config = replace(train_backup_sampling.CONFIG.to_backend(tmp_path),
                     phase_reward_mode="cosine", n_jobs=1)
    old_fields = {item.name for item in fields(backup.BernoulliSparseConfig)}
    old_config = backup.BernoulliSparseConfig(**{
        key: value for key, value in asdict(config).items() if key in old_fields})
    final_states = []
    for implementation, settings in ((backup, old_config), (engine, config)):
        rng = core.set_random_seed(7)
        policy = implementation.BernoulliSparseGNNPolicy(settings)
        settings = implementation.calibrate_budget_references(policy, settings)
        policy.config = settings
        optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
        scale = {}
        metrics = []
        for iteration in (0, 1000, 1499):
            measured = implementation.run_training_batch(policy, optimizer, settings, rng,
                iteration, reward_scale_state=scale)
            assert measured['pcgrad_task_count'] == 16
            metrics.append(measured)
        final_states.append((policy.state_dict(), optimizer.state_dict(), scale,
                             rng.bit_generator.state, torch.get_rng_state(), metrics))
    previous, current = final_states
    for old, new in zip(previous[:-1], current[:-1]):
        assert_equal(old, new)
    for old, new in zip(previous[-1], current[-1]):
        assert_equal({key: value for key, value in old.items() if key != 'size_metrics'},
                     {key: new[key] for key in old if key != 'size_metrics'})
        for size, measured in old['size_metrics'].items():
            assert_equal(measured, {key: new['size_metrics'][size][key] for key in measured})


def test_full_rms_iteration_dispatches_1120_cases_and_16_tasks(tmp_path, monkeypatch, fake_physics):
    public = replace(train_backup_sampling.CONFIG,
        optimization=replace(train_backup_sampling.CONFIG.optimization, iterations=1, workers=1),
        validation=ValidationConfig(pairs_per_size=4))
    worker = core.simulate_indexed_target_chunk
    jobs = []
    def record(job):
        jobs.append(job)
        return worker(job)
    monkeypatch.setattr(core, "simulate_indexed_target_chunk", record)
    result = Experiment('backup-sampling-test', config=public,
                        output_directory=tmp_path/'run').train(live_plot=False)
    assert len(jobs) == 28
    assert all(job[3][4].shape[:2] == (5, 8) for job in jobs)
    assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in jobs) == 1120
    assert result.history['pcgrad_task_count'] == [16]
    assert result.history['rooted_fraction'] == [1.]
    assert result.history['training_reward'] != result.history['phase_reward']
    assert result.model.backend_config.phase_reward_mode == 'wrapped_rms'
    for kind in ('best', 'current', 'final'):
        loaded, config, _ = engine.load_checkpoint(result.checkpoint(kind))
        assert config.topology_sampling_mode == 'per_target'
        assert config.pcgrad_task_grouping == 'array_size_topology_density'
        assert config.phase_reward_mode == 'wrapped_rms'


def test_new_run_cells_wait_for_explicit_start(tmp_path, monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    monkeypatch.chdir(REPOSITORY_DIR/'rl')
    monkeypatch.setattr(sys, 'argv', ['ipykernel_launcher.py', '-f', 'connection.json'])
    calls = []
    monkeypatch.setattr(Experiment, 'train', lambda self, **kwargs: calls.append(self))
    scope = {'__name__': '__main__'}
    cells = read_cells('train_backup_sampling.py')
    for cell in cells:
        preview_cell = cell.replace('START_TRAINING = True', 'START_TRAINING = False')
        exec(compile(preview_cell, '<backup-sampling-cell>', 'exec'), scope)
        if 'CONFIG' in scope:
            scope['OUTPUT_DIRECTORY'] = tmp_path/'cells'
    assert calls == []
    assert '__file__' not in scope
    assert not (tmp_path/'cells').exists()
    scope['START_TRAINING'] = True
    start = next(cell for cell in cells if 'result = experiment.train(' in cell)
    exec(compile(start, '<train-cell>', 'exec'), scope)
    assert len(calls) == 1
