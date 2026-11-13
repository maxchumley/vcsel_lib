"""Native phase choices reach workers, held-out scoring and saved inspection."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel, TrainingConfig, OptimizationConfig, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api import evaluation, simulation
from rl.rl_api.models import build_policy
from rl.rl_api.experiments import train_current
from rl.rl_api.tests.test_parity import assert_equal


@pytest.mark.parametrize('phase_model', ['chumley_2026', 'standard_lk', 'ma_2019'])
def test_native_flag_reaches_scaled_vcsel_parameters(phase_model, tmp_path):
    native = train_current.VCSEL(dict(train_current.phys, phase_model=phase_model))
    experiment = Experiment('phase-choice', vcsel=native, output_directory=tmp_path)
    config = experiment.backend_config
    assert getattr(config, 'phase_model', 'chumley_2026') == phase_model
    vcsel, _, _ = simulation.prepare_vcsel(np.zeros((1,3,3)), np.zeros((1,3,3)),
        config, detunings_ghz=np.array([-.5,0.,.5]))
    assert vcsel.phys['phase_model'] == vcsel.nd['phase_model'] == phase_model
    native.phys['phase_model'] = 'ma_2019'  # Snapshot is independent of the template.
    assert getattr(experiment.backend_config, 'phase_model', 'chumley_2026') == phase_model


def test_unknown_phase_model_fails_before_training():
    with pytest.raises(ValueError, match='Unknown phase_model'):
        TrainingConfig(phys=dict(train_current.phys, phase_model='typo'))


def test_validation_matches_reference_procedure_and_passes_flag(tmp_path, fake_physics):
    settings = TrainingConfig(phys=dict(train_current.phys, phase_model='standard_lk'))
    config = replace(settings.to_backend(tmp_path), n_jobs=2)
    policy = build_policy(config)
    config = engine.calibrate_budget_references(policy, config)
    policy.config = config
    targets = np.array([[0.,-.7,1.3], [0.,.2,-2.], [0.,1.,-1.], [0.,0.,0.]])
    detunings = np.tile([-.5,0.,.5], (len(targets),1))
    workers = []

    class Pool:
        def imap_unordered(self, worker, jobs):
            jobs = list(jobs)
            workers.append(worker)
            assert all(job[-1].phase_model == 'standard_lk' for job in jobs)
            return reversed([worker(job) for job in jobs])

    reference = engine.evaluate_policy(policy, targets, detunings, config,
        seed=91, simulation_pool=Pool())
    reference_calls = list(fake_physics)
    fake_physics.clear()
    before = torch.get_rng_state().clone()
    actual = evaluation.evaluate_phase_model_policy(policy, targets, detunings, config,
        seed=91, simulation_pool=Pool())
    assert_equal(reference, actual)
    assert_equal(before, torch.get_rng_state())
    assert workers == [core.simulate_target_chunk, simulation.simulate_phase_model_chunk]
    for old, new in zip(reference_calls, fake_physics):
        assert_equal(old[:2], new[:2])
        assert_equal(old[2]['detuning_distribution_ghz'], new[2]['detuning_distribution_ghz'])


def test_paired_training_validation_and_checkpoint_keep_flag(tmp_path, monkeypatch, fake_physics):
    settings = TrainingConfig(sizes=(3,), phys=dict(train_current.phys, phase_model='standard_lk'),
        optimization=OptimizationConfig(iterations=1, workers=1),
        validation=ValidationConfig(pairs_per_size=4))
    seen = []
    original = simulation.simulate

    def record_simulation(kappa, phi, config, **options):
        seen.append(config.phase_model)
        return original(kappa, phi, config, **options)

    monkeypatch.setattr(simulation, 'simulate', record_simulation)
    result = Experiment('phase-test', config=settings,
        output_directory=tmp_path/'run').train(live_plot=False)
    assert len(seen) >= 2 and set(seen) == {'standard_lk'}
    # This recipe has no precision exploration metadata; the flag must still load.
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert loaded.backend_config.phase_model == 'standard_lk'
    designs = loaded.propose(target_phases_rad=[0.,-1.,2.],
        detunings_ghz=[-.5,0.,.5], count=1, seed=3)
    assert designs.config.phase_model == 'standard_lk'
    assert designs.evaluate().config.phase_model == 'standard_lk'
