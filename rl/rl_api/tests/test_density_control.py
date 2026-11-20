"""Connected pair selection, policy-gradient scores, and checkpoint compatibility."""
from dataclasses import asdict, replace
import itertools
import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment
from rl.rl_api._compat import core, engine
from rl.rl_api.checkpoints import initialize_from_checkpoint, save
from rl.rl_api.density_control import (DensityConditionedPolicy, decode_density_actions,
    pair_counts, select_masks)
from rl.rl_api.experiments import train_mirrored_delay_1000 as star
from rl.rl_api.experiments import train_mirrored_symmetric_density_1000 as run
from rl.rl_api.experiments import train_mirrored_symmetric_density_values_1000 as symmetric
from rl.rl_api.experiments import train_mirrored_symmetric_density_values_zeroref_1000 as zero
from rl.rl_api.models import build_policy
from rl.rl_api.root_context import calibrate_budget_references


def connected(matrix):
    seen = {0}
    while True:
        updated = seen | {int(j) for i in seen for j in np.flatnonzero(matrix[i])}
        if updated == seen:
            return len(seen) == len(matrix)
        seen = updated


def test_density_recipe_keeps_the_working_star_training_settings():
    before, after = asdict(star.CONFIG), asdict(run.CONFIG)
    assert before['topology'].pop('mode') == 'provided'
    assert after['topology'].pop('mode') == 'density'
    assert before == after
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['node_input_features'] == 4 and summary['new_parameters'] == 0
    assert run.INITIAL_CHECKPOINT is None
    assert run.RUN_NAME != star.RUN_NAME


def test_selected_masks_have_exact_counts_connectivity_and_learned_trees():
    rng = np.random.default_rng(71)
    for size in range(3, 10):
        fractions = np.array([[0., .2, .5, .8, 1.]])
        strengths = rng.uniform(.1, 3., (1, 5, size, size))
        masks = select_masks(strengths, fractions)
        receiver, source = core.directed_link_indices(size)
        matrices = np.zeros_like(strengths, dtype=bool)
        matrices[..., receiver, source] = masks
        np.testing.assert_array_equal(matrices, matrices.swapaxes(-1,-2))
        np.testing.assert_array_equal(masks.sum(-1), 2*pair_counts(size, fractions))
        assert all(connected(matrix) for matrix in matrices[0])
        assert not np.diagonal(matrices, axis1=-2, axis2=-1).any()
        assert masks[0,-1].all()
        np.testing.assert_array_equal(select_masks(strengths, fractions), masks)
    # Strong path 0-1-2-3 wins over a direct root-0 star.
    strengths = np.ones((1,1,4,4))
    for a,b in [(0,1),(1,2),(2,3)]:
        strengths[0,0,a,b] = 20.
    receiver,source=core.directed_link_indices(4)
    actual=np.zeros((4,4),bool)
    actual[receiver,source]=select_masks(strengths,[[0.]])[0,0]
    assert {tuple(edge) for edge in np.argwhere(np.triu(actual,1))} == {(0,1),(1,2),(2,3)}
    # For a tiny case, verify that the selected tree maximizes the pair score.
    pairs=list(itertools.combinations(range(4),2))
    best=-np.inf
    for edges in itertools.combinations(pairs,3):
        graph=np.zeros((4,4),bool)
        for a,b in edges:
            graph[a,b]=graph[b,a]=True
        if connected(graph):
            best=max(best,sum(strengths[0,0,a,b]+strengths[0,0,b,a] for a,b in edges))
    assert strengths[0,0][actual].sum() == best


def test_density_checkpoint_reuses_every_weight_and_dense_prediction(tmp_path):
    torch.manual_seed(7)
    old_config=star.CONFIG.to_backend(tmp_path/'old')
    old=build_policy(old_config)
    old_config=calibrate_budget_references(old,old_config)
    checkpoint=tmp_path/'star.pt'
    save(checkpoint,old,torch.optim.Adam(old.parameters()),old_config,1000,.99)
    new_config=run.CONFIG.to_backend(tmp_path/'new')
    new=build_policy(new_config)
    new_config,info=initialize_from_checkpoint(new,new_config,checkpoint)
    assert info['new_parameters'] == 0
    assert new.node_encoder[0].in_features == 4
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters())
    for key,value in old.state_dict().items():
        copied=new.state_dict()[key]
        torch.testing.assert_close(value,copied,rtol=0,atol=0)
    for size in range(3,10):
        target=np.linspace(0,2,size)[None]
        detuning=np.linspace(-1,1,size)[None]
        features=engine.encode_context(target,detuning,old)
        dense=torch.ones((1,size*(size-1)))
        for density in (0.,.5,1.):
            retained=float(pair_counts(size,density))/(size*(size-1)/2)
            a=old.parameters_for_edges(features,topology_masks=dense*retained,return_pooled=True)
            view=DensityConditionedPolicy(new,[density])
            b=view.parameters_for_edges(features,topology_masks=dense,return_pooled=True)
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
    model=CouplingModel(new,new_config)
    context=Context([0.,.4,-.7],[1.,-.5,.3])
    designs=model.propose(context,density_fraction=[0.,1.],count=2)
    np.testing.assert_array_equal(designs.active_link_counts,[4,6])
    assert not np.allclose(designs.coupling_strengths_per_ns,
                           designs.coupling_strengths_per_ns.swapaxes(-1,-2))
    with pytest.raises(ValueError,match='density_fraction'):
        model.propose(context,topology=np.ones((3,3))-np.eye(3))
    new_path=tmp_path/'density.pt'
    save(new_path,new,torch.optim.Adam(new.parameters()),new_config,1,.9)
    loaded=CouplingModel.load(new_path,device='cpu')
    replay=loaded.propose(context,density_fraction=[0.,1.],count=2)
    np.testing.assert_array_equal(replay.coupling_strengths_per_ns,designs.coupling_strengths_per_ns)


def test_none_density_ensemble_uses_seeded_requests_and_each_candidates_density(tmp_path):
    config=replace(run.CONFIG.to_backend(tmp_path),exploration_std_scale=0.)
    model=CouplingModel(build_policy(config).eval(),config)
    context=Context([0.,.2,-.7,.8,.1],[-1.,-.4,.1,.5,1.])
    state=torch.get_rng_state().clone()
    designs=model.propose(context,density_fraction=None,count=11,seed=72)
    assert torch.equal(state,torch.get_rng_state())
    fractions=designs.raw['density_fractions']
    assert len(fractions)==11 and fractions.min()==0. and fractions.max()==1.
    np.testing.assert_array_equal(designs.active_link_counts,2*pair_counts(5,fractions))
    assert np.all(designs.topologies==designs.topologies.swapaxes(-1,-2))
    assert all(connected(matrix) for matrix in designs.topologies)
    replay=model.propose(context,count=11,seed=72)
    np.testing.assert_array_equal(replay.raw['density_fractions'],fractions)
    np.testing.assert_array_equal(replay.topologies,designs.topologies)
    different=model.propose(context,count=11,seed=73)
    assert not np.array_equal(different.raw['density_fractions'],fractions)
    for index,fraction in enumerate(fractions):
        single=model.propose(context,density_fraction=float(fraction),count=1)
        np.testing.assert_allclose(designs.coupling_strengths_per_ns[index],
            single.coupling_strengths_per_ns[0],rtol=2e-6,atol=2e-6)
        np.testing.assert_array_equal(designs.topologies[index],single.topologies[0])
    one=model.propose(context,count=1,seed=72)
    assert 0. <= one.raw['density_fractions'][0] <= 1.


def test_pruning_preserves_budget_and_scores_removed_strengths(tmp_path):
    config=run.CONFIG.to_backend(tmp_path)
    policy=build_policy(config)
    from rl.rl_api.exploration import sample_annealed_candidates
    features=engine.encode_context([[0.,.2,.4]], [[-1.,0.,1.]],policy)
    dense=torch.ones((1,8,6))
    fractions=np.zeros((1,8))
    view=DensityConditionedPolicy(policy,fractions)
    torch.manual_seed(81)
    actions,budget,log_prob=sample_annealed_candidates(view,features,dense,mirrored=True)
    masks,decoded=decode_density_actions(actions,budget,fractions,config)
    kappa,_,counts,_,shortfall=decoded
    np.testing.assert_allclose(kappa.sum((-2,-1)),budget.detach().exp().numpy(),rtol=2e-6)
    assert np.all(counts == 4) and not shortfall.any()
    assert not kappa[...,0,0].any()
    # Reconstruct the full Gaussian score: masking removed strengths is wrong
    # because they determine the selected tree. No pathwise action gradient.
    from rl.rl_api.exploration import shared_pair_parameters
    repeated=features.repeat_interleave(8,dim=0)
    mean,log_std,_,pooled=view.parameters_for_edges(repeated,topology_masks=dense.reshape(8,6),return_pooled=True)
    mean=shared_pair_parameters(mean.reshape(1,8,12))
    std=log_std.reshape(1,8,12).exp()
    extra=torch.zeros((1,8,1)); extra[:,:2]=config.phase_exploration_start_std
    std=torch.cat((std[...,:6],torch.sqrt(std[...,6:]**2+extra**2)),dim=-1)
    std=shared_pair_parameters(std)
    bm,bs=view.budget_parameters(repeated,pooled_states=pooled)
    bm=shared_pair_parameters(bm.reshape(1,8))
    bs=shared_pair_parameters(bs.reshape(1,8).exp())
    expected=torch.distributions.Normal(mean,std).log_prob(actions).sum(-1)
    expected+=torch.distributions.Normal(bm,bs).log_prob(budget)
    torch.testing.assert_close(log_prob,expected)
    assert not actions.requires_grad and not masks.requires_grad
    gradients=torch.autograd.grad(log_prob[0,0],policy.edge_output.bias)[0]
    assert torch.isfinite(gradients).all() and gradients[:2].abs().sum()>0


def test_density_training_validation_and_reloading(tmp_path,fake_physics,monkeypatch):
    from rl.rl_api import training
    initial={}
    original=training.build_policy
    def capture(config):
        policy=original(config)
        initial['edge_weights']=policy.edge_output.weight.detach().clone()
        return policy
    monkeypatch.setattr(training,'build_policy',capture)
    config=replace(run.CONFIG,sizes=(3,4),
        sampling=replace(run.CONFIG.sampling,pairs_per_size=1,pairs_per_job=1),
        optimization=replace(run.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(run.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(run.CONFIG.validation,pairs_per_size=5,plot_use_tex=False))
    result=Experiment('density-test',config=config,output_directory=tmp_path/'run').train(
        initial_checkpoint=run.INITIAL_CHECKPOINT,live_plot=False,verify_reference=False)
    assert len(result.history['training_reward']) == 2
    assert 'held_out_tree_success_fraction' in result.history
    assert np.isfinite(result.history['held_out_ata_reward']).all()
    assert result.model.policy.node_encoder[0].in_features == 4
    assert not torch.equal(result.model.policy.edge_output.weight,initial['edge_weights'])
    saved=torch.load(result.checkpoint('final'),weights_only=False)
    assert saved['sampling_recipe']['mode']=='density_controlled'
    assert 'initialization' not in saved['sampling_recipe']
    assert saved['exploration_state']['action_std_scale']==0.
    assert saved['iteration']==2
    model=CouplingModel.load(result.checkpoint('final'),device='cpu')
    proposals=model.propose(Context([0.,.2,.4],[-1.,0.,1.]),density_fraction=0.,count=2)
    assert np.all(proposals.active_link_counts == 4)
    assert connected(proposals.topologies[0])
    figure=result.plot(show=False)
    assert len(figure.axes)==2
    figure.canvas.draw()
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize('fraction',[-.1,1.1,np.nan,np.inf])
def test_invalid_density_requests_are_rejected(fraction):
    with pytest.raises(ValueError,match=r'\[0,1\]'):
        pair_counts(5,fraction)


def test_symmetric_density_recipes_only_change_values_reference_and_plot_names():
    for recipe,phase in ((symmetric,'target_delay_residual'),(zero,'target_residual')):
        before,after=asdict(run.CONFIG),asdict(recipe.CONFIG)
        assert not before['model'].pop('symmetric_coupling_values')
        assert after['model'].pop('symmetric_coupling_values')
        before['model'].pop('phase_parameterization')
        assert after['model'].pop('phase_parameterization')==phase
        assert before['validation'].pop('plot_every')==1
        assert after['validation'].pop('plot_every')==10
        before['validation'].pop('progress_filename')
        assert after['validation'].pop('progress_filename').endswith('progress.png')
        assert before==after
        assert recipe.INITIAL_CHECKPOINT is None
        summary=recipe.CONFIG.summary()
        assert summary['symmetric_masks'] and summary['symmetric_strengths'] and summary['symmetric_phases']
        assert summary['simulations_per_iteration']==1400
        assert recipe.CONFIG.to_backend('.').plot_update_interval==10
    assert len({r.RUN_NAME for r in (run,symmetric,zero)})==3
    assert symmetric.CONFIG.validation.progress_filename!=zero.CONFIG.validation.progress_filename


def test_zero_reference_only_removes_the_explicit_phase_rotation(tmp_path):
    from rl.rl_api.models.policy import delay_phase_offsets
    config=symmetric.CONFIG.to_backend(tmp_path/'reference')
    no_reference=zero.CONFIG.to_backend(tmp_path/'zero')
    policy=build_policy(config)
    other=build_policy(no_reference)
    other.load_state_dict(policy.state_dict())
    features=engine.encode_context([[0.,.4,-.7]], [[-.9,-.1,1.2]],policy)
    masks=torch.ones((1,6))
    a=DensityConditionedPolicy(policy,[.5]).parameters_for_edges(features,topology_masks=masks)
    b=DensityConditionedPolicy(other,[.5]).parameters_for_edges(features,topology_masks=masks)
    torch.testing.assert_close(a[0][...,:6],b[0][...,:6],rtol=0,atol=0)
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    offset=delay_phase_offsets(features,config).expand(-1,6)
    assert offset.abs().min()>0.
    torch.testing.assert_close(a[0][...,6:]-b[0][...,6:],offset)


@pytest.mark.parametrize('recipe',[symmetric,zero],ids=['laser1','zero'])
def test_density_values_symmetric_for_all_sizes_and_sampled_targets(tmp_path,recipe):
    from rl.rl_api.density_control import prepare_density_size
    torch.manual_seed(7)
    config=recipe.CONFIG.to_backend(tmp_path)
    policy=build_policy(config)
    assert sum(p.numel() for p in policy.parameters())==sum(p.numel() for p in build_policy(run.CONFIG.to_backend(tmp_path)).parameters())
    for size in range(3,10):
        item=prepare_density_size(policy,config,np.random.default_rng(71),size,1)
        assert np.any(item['targets']!=0.)
        for matrix in (item['kappa'],item['phi']):
            np.testing.assert_array_equal(matrix,matrix.swapaxes(-1,-2))
        receiver,source=core.directed_link_indices(size)
        graphs=np.zeros_like(item['kappa'],dtype=bool)
        graphs[...,receiver,source]=item['gates'].numpy()
        assert all(connected(g) for g in graphs[0])
        np.testing.assert_array_equal(item['counts'],2*pair_counts(size,item['density_fractions']))
        gradient=torch.autograd.grad(item['log_prob'][0,0],policy.edge_output.bias)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('recipe',[symmetric,zero],ids=['laser1','zero'])
def test_symmetric_density_training_and_inference_roundtrip(tmp_path,fake_physics,recipe):
    config=replace(recipe.CONFIG,sizes=(3,4),
        sampling=replace(recipe.CONFIG.sampling,pairs_per_size=1,pairs_per_job=1),
        optimization=replace(recipe.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(recipe.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(recipe.CONFIG.validation,pairs_per_size=5,plot_use_tex=False,notebook_display=False))
    result=Experiment('symmetric-density',config=config,output_directory=tmp_path/'run').train(
        live_plot=True,verify_reference=False)
    for kappa,phi,_ in fake_physics:
        np.testing.assert_array_equal(kappa,kappa.swapaxes(-1,-2))
        np.testing.assert_array_equal(phi,phi.swapaxes(-1,-2))
    payload=torch.load(result.checkpoint('final'),weights_only=False)
    assert 'initialization' not in payload['sampling_recipe']
    assert payload['sampling_recipe']['mode']=='density_controlled'
    assert 'symmetric_values' in payload['sampling_recipe']
    assert ('zero frequency' in payload['sampling_recipe']['phase_delay_reference'])==(recipe is zero)
    model=CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert model.backend_config.symmetric_coupling_values
    assert model.backend_config.phase_parameterization==recipe.CONFIG.model.phase_parameterization
    context=Context([0.,.2,-.8,.7],[-2.,-.3,.5,2.])
    designs=model.propose(context,density_fraction=None,count=10,seed=71)
    np.testing.assert_array_equal(designs.coupling_strengths_per_ns,designs.coupling_strengths_per_ns.swapaxes(-1,-2))
    np.testing.assert_array_equal(designs.coupling_phases_rad,designs.coupling_phases_rad.swapaxes(-1,-2))
    assert (result.directory/recipe.CONFIG.validation.progress_filename).is_file()


def test_symmetric_density_launcher_submits_two_distinct_recipes():
    import shlex
    import subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    script=root/'scripts/submit_symmetric_density_tests.sh'
    result=subprocess.run(['bash',str(script),'--dry-run','--smoke-test'],check=True,capture_output=True,text=True)
    commands=[shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
    assert len(commands)==2
    for command,name,recipe in zip(commands,('density_sym_ref','density_sym_zero'),(symmetric,zero)):
        assert f'--job-name={name}' in command and '--smoke-test' in command
        assert recipe.__name__.split('.')[-1] in command
        assert '--cpus-per-task=32' in command and '--time=7-00:00:00' in command
        assert command[command.index('--workers')+1]=='31'
        assert command[command.index('--pairs-per-job')+1]=='5'
        assert command[command.index('--actions-per-job')+1]=='8'
    subprocess.run(['bash','-n',str(script)],check=True)
