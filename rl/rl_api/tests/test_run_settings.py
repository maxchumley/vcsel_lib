"""Run-file settings are visible, effective, and independent of the JSON snapshot."""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pytest

from rl.rl_api import Experiment, EvaluationBatch
from rl.rl_api.experiments import train_current, inspect_current
from rl.rl_api.tests.test_workflow import make_model


def test_training_run_matches_original_without_reading_json_settings(tmp_path,monkeypatch):
    from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import config
    read_text=Path.read_text
    def read_without_preset(path,*args,**kwargs):
        if path.name=='current_training.json':
            raise AssertionError('Runtime attempted to read JSON experiment settings')
        return read_text(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_without_preset)
    monkeypatch.setattr(train_current,'OUTPUT_DIRECTORY',tmp_path/'preview')
    actual=asdict(train_current.main([]).backend_config)
    expected=asdict(config)
    differences={key for key in actual if actual[key]!=expected[key]}
    assert differences=={f'{kind}_checkpoint_file' for kind in ('best','current','final')}
    assert not (tmp_path/'preview').exists()


def test_edited_run_config_controls_preview_without_training(tmp_path,monkeypatch):
    config=train_current.CONFIG
    config=replace(config,
        sampling=replace(config.sampling,pairs_per_size=2,topologies_per_pair=3,actions_per_topology=8),
        schedules=replace(config.schedules,wide_phase_actions=1,wide_phase_extra_sd_deg=45.))
    monkeypatch.setattr(train_current,'CONFIG',config)
    monkeypatch.setattr(train_current,'OUTPUT_DIRECTORY',tmp_path/'edited')
    def unexpected_training(*args,**kwargs):
        raise AssertionError('Preview started training')
    monkeypatch.setattr(Experiment,'train',unexpected_training)
    experiment=train_current.main([])
    assert experiment.config.summary()['simulations_per_iteration']==7*2*3*8
    assert experiment.backend_config.phase_exploration_fraction==1/8
    assert experiment.backend_config.phase_exploration_start_std==np.pi/4
    assert not experiment.directory.exists()


def test_inspection_settings_reach_prediction_evaluation_and_plotting(tmp_path,monkeypatch,fake_physics):
    model=make_model(tmp_path)
    monkeypatch.setattr(inspect_current.CouplingModel,'load',lambda *args,**kwargs:model)
    monkeypatch.setattr(inspect_current,'N_LASERS',3)
    monkeypatch.setattr(inspect_current,'MANUAL_TARGET_PHASES_PI',np.array([0.,-.7,1.3])/np.pi)
    monkeypatch.setattr(inspect_current,'MANUAL_DETUNINGS_GHZ',[-1.,0.,1.])
    monkeypatch.setattr(inspect_current,'NUMBER_OF_CANDIDATES',4)
    monkeypatch.setattr(inspect_current,'NOISE_ITERATIONS',2)
    monkeypatch.setattr(inspect_current,'VALIDATION_TIME_NS',450.)
    monkeypatch.setattr(inspect_current,'ENSEMBLE_SELECTED_TOPOLOGY','best_rms')
    plot_calls=[]
    monkeypatch.setattr(EvaluationBatch,'plot',lambda self,**kwargs:plot_calls.append(kwargs))
    result=inspect_current.inspect(show=False)['results']
    np.testing.assert_array_equal(result.designs.context.target_phases_rad,[0.,-.7,1.3])
    assert len(result.designs)==4
    assert result.noise_realizations==2
    assert result.config.simulation_time_seconds==450.*1e-9
    assert len(fake_physics)==1
    assert plot_calls[0]['select']==int(np.argmax(result.rms_reward))
    assert plot_calls[0]['options'].layout=='inspection'
    assert plot_calls[0]['show'] is False


def test_native_vcsel_parameters_are_snapshotted_and_used_for_each_size(tmp_path):
    from rl.rl_api._compat import core
    phys=dict(train_current.phys)
    phys.update(I=phys['I']*.98,dt=phys['dt']*2,Tmax=400.*1e-9)
    vcsel=train_current.VCSEL(phys)
    experiment=Experiment('native',vcsel=vcsel,sizes=(3,5),output_directory=tmp_path)
    config=experiment.backend_config
    assert experiment.vcsel is vcsel
    assert config.pump_current_amp==phys['I']
    assert config.time_step_seconds==phys['dt']
    assert config.simulation_time_seconds==phys['Tmax']
    assert not hasattr(experiment.config,'physics')
    for size in (3,5):
        local=replace(config,n_lasers=size)
        actual=core.make_vcsel_physical_parameters(local,
            detuning_distribution_ghz=np.linspace(-1,1,size))
        for name,value in experiment.config.phys.items():
            assert actual[name]==value
        assert actual['N_lasers']==size
        assert actual['kappa_c_mat'].shape==(size,size)
    saved_current=experiment.config.phys['I']
    vcsel.phys['I']*=2
    assert experiment.config.phys['I']==saved_current


def test_unsupported_native_modes_are_not_silently_ignored(tmp_path):
    phys=dict(train_current.phys,self_feedback=.2)
    with pytest.raises(NotImplementedError,match='self_feedback'):
        Experiment('native',vcsel=train_current.VCSEL(phys),output_directory=tmp_path)
