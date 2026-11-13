"""Public inference/evaluation workflow and preserved inspection conventions."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment, Topology, PlotOptions
from rl.rl_api._compat import core, engine
from rl.rl_api.models import build_policy


def make_model(tmp_path):
    config=Experiment("inspect",output_directory=tmp_path).backend_config
    torch.manual_seed(9)
    policy=build_policy(config)
    config=engine.calibrate_budget_references(policy,config);policy.config=config
    return CouplingModel(policy,config,output_directory=tmp_path)


@pytest.mark.parametrize("actions", ["mean","sample"])
def test_proposals_match_existing_inference_and_matrix_order(tmp_path,actions):
    model=make_model(tmp_path)
    target=np.array([0.,-.4,1.2]);detuning=np.array([1.,-1.,0.])
    # Source 2 (lowest detuning) drives all lasers, in the user's original order.
    source_receiver=np.array([[0,0,0],[1,0,1],[0,0,0]])
    mask=Topology.from_source_receiver(source_receiver)
    designs=model.propose(target_phases_rad=target,detunings_ghz=detuning,
        topology=mask,count=4,continuous_actions=actions,seed=11)
    config=replace(model.backend_config,n_lasers=3,n_jobs=1,jupyter_mode=False,noise_amplitude=0.)
    expected=engine.infer_designs(model.policy,target,detuning,config,count=4,seed=11,
        continuous_mean=actions=="mean",topology=source_receiver)
    np.testing.assert_array_equal(designs.coupling_strengths_per_ns,expected["kappa_per_ns"])
    np.testing.assert_array_equal(designs.coupling_phases_rad,expected["phi_p_rad"])
    np.testing.assert_array_equal(designs.context.detunings_ghz,[-1.,0.,1.])
    assert designs.gate_logits is designs.gate_probabilities is None
    assert designs.topologies.shape==(4,3,3)
    assert np.all(designs.topologies[:,1:,0])
    np.testing.assert_array_equal(mask.as_source_receiver(),source_receiver)


def test_loading_current_format_does_not_change_global_torch_rng(tmp_path):
    model=make_model(tmp_path)
    path=tmp_path/'model.pt'
    engine.save_checkpoint(path,model.policy,torch.optim.Adam(model.policy.parameters()),
        model.backend_config,500,.9)
    before=torch.get_rng_state().clone()
    loaded=CouplingModel.load(path,device='cpu',output_directory=tmp_path/'plots')
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
    assert loaded.metadata['iteration']==500
    for a,b in zip(model.policy.parameters(),loaded.policy.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_evaluation_matches_both_metrics_and_selection_reuses_trajectories(tmp_path,fake_physics):
    model=make_model(tmp_path)
    designs=model.propose(target_phases_rad=[0.,.4,-.8],detunings_ghz=[-.5,0.,.5],count=6,seed=4)
    result=designs.evaluate(noise_realizations=2,seed=123)
    assert len(fake_physics)==1
    target=np.repeat(designs.context.target_phases_rad[None],12,axis=0)
    cos=core.calculate_phase_reward(*result.simulation[:2],target,result.config)
    rms=core.calculate_phase_rms_metrics(*result.simulation[:2],target,result.config)
    np.testing.assert_array_equal(result.cosine_reward,cos[0].reshape(6,2).mean(1))
    np.testing.assert_array_equal(result.rms_reward,rms['phase_rms_reward'].reshape(6,2).mean(1))
    for mode in ['best','best_rms','sparsest','min_budget']:
        chosen=result.select(mode)
        assert 0<=chosen.index<6
        assert 'maximum_mean_phase_error_deg' in chosen.metrics
    assert len(fake_physics)==1


def test_plotting_does_not_simulate_and_uses_only_its_output_directory(tmp_path,fake_physics):
    import matplotlib.pyplot as plt
    model=make_model(tmp_path)
    designs=model.propose(target_phases_rad=[0.,.4,-.8],detunings_ghz=[-.5,0.,.5],count=3,seed=4)
    result=designs.evaluate()
    calls=len(fake_physics)
    figures=result.plot(show=False,options=PlotOptions(dpi=45,n_theta=51,max_time_points=16))
    assert set(figures)=={'design','phase_comparison','dynamics','ensemble','far_field'}
    assert len(fake_physics)==calls
    assert len(list((tmp_path/'figures').glob('*.png')))==5
    for figure in figures.values():plt.close(figure)


def test_far_field_matches_existing_normalization(tmp_path,fake_physics):
    import matplotlib.pyplot as plt
    from rl.rl_api.plotting.far_field import calculate_far_field
    model=make_model(tmp_path)
    result=model.propose(target_phases_rad=[0.,.4,-.8],detunings_ghz=[-.5,0.,.5],count=2,seed=4).evaluate()
    options=PlotOptions(n_theta=51,max_time_points=16)
    before=dict(plt.rcParams)
    data=calculate_far_field(result.simulation,options)
    assert plt.rcParams['text.usetex']==before['text.usetex']
    # Read only the pure reference functions. Importing the example itself also
    # imports its standalone simulator and changes global plotting defaults.
    import ast
    from rl.rl_api._compat import REPOSITORY_DIR
    tree=ast.parse((REPOSITORY_DIR/'examples/visualization/far_field_intensity.py').read_text())
    names={'emitter_positions','angle_grid','evaluate_array_factor','far_field_intensity_map'}
    functions=ast.Module(body=[node for node in tree.body
        if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    namespace={'np':np}
    exec(compile(functions,'reference_far_field','exec'),namespace)
    far_field_intensity_map=namespace['far_field_intensity_map']
    theta,indices,intensity=far_field_intensity_map(result.simulation[1][:,1::3],
        result.simulation[1][:,2::3],lambda0=options.wavelength_m,d=10e-6,
        theta_range_deg=options.theta_range_deg,n_theta=51,max_time_points=16,element_fwhm_deg=5.)
    np.testing.assert_allclose(data['intensity_normalized'],intensity/np.max(intensity[:,-1]),atol=1e-14)


@pytest.mark.parametrize('target,detuning', [([0.],[0.]),([0.,1.],[0.]),([0.,np.nan],[0.,1.])])
def test_context_rejects_ambiguous_or_nonfinite_inputs(target,detuning):
    with pytest.raises(ValueError):Context(target,detuning)
