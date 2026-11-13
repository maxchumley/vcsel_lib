"""Fresh restarts preserve old files and never reuse their optimizer state."""
import torch

from rl.rl_api import Experiment, OptimizationConfig, TrainingConfig, ValidationConfig
from rl.rl_api.tests.test_parity import assert_equal


def test_repeated_restarts_archive_each_attempt_and_initialize_fresh(tmp_path, fake_physics):
    directory = tmp_path/'repeat'
    directory.mkdir()
    (directory/'current.pt').write_bytes(b'previous checkpoint')
    (directory/'notes.txt').write_text('keep these notes')
    settings = TrainingConfig(sizes=(3,),
        optimization=OptimizationConfig(iterations=3000, workers=14),
        validation=ValidationConfig(pairs_per_size=4))
    experiment = Experiment('repeat', config=settings, output_directory=directory)

    first = experiment.train(iterations=1, workers=1, live_plot=False, restart=True)
    archives = list((tmp_path/'restarts').iterdir())
    assert len(archives) == 1
    assert (archives[0]/'current.pt').read_bytes() == b'previous checkpoint'
    assert (archives[0]/'notes.txt').read_text() == 'keep these notes'
    saved_first = torch.load(first.checkpoint('current'), weights_only=False)
    assert saved_first['iteration'] == 1
    assert not (directory/'notes.txt').exists()

    second = experiment.train(iterations=1, workers=1, live_plot=False, restart=True)
    archives = list((tmp_path/'restarts').iterdir())
    assert len(archives) == 2
    saved_second = torch.load(second.checkpoint('current'), weights_only=False)
    assert saved_second['iteration'] == 1
    # Restarting with the same seed reproduces a fresh first update, rather
    # than using the existing model or continuing Adam's step count.
    assert_equal(saved_first['policy_state_dict'], saved_second['policy_state_dict'])
    assert_equal(saved_first['optimizer_state_dict'], saved_second['optimizer_state_dict'])


def test_failed_reference_check_keeps_previous_attempt_in_place(tmp_path, monkeypatch):
    from rl.rl_api import training
    directory = tmp_path/'protected'
    directory.mkdir()
    sentinel = directory/'current.pt'
    sentinel.write_bytes(b'preserve this')
    monkeypatch.setattr(training, 'reference_source_status', lambda: {})
    import pytest
    with pytest.raises(RuntimeError, match='Reference numerical source changed'):
        Experiment('protected', output_directory=directory).train(restart=True, live_plot=False)
    assert sentinel.read_bytes() == b'preserve this'
    assert not (tmp_path/'restarts').exists()
