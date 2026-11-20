"""Matched symmetric experiments, projected score, and three-image cluster flow."""
from dataclasses import asdict, replace
from importlib import import_module, util
from pathlib import Path
import json
import shlex
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel
from rl.rl_api._compat import core
from rl.rl_api.exploration import sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.phase_reference import decode_samples
from rl.rl_api.symmetric_topologies import connected_symmetric_masks, reciprocal_normal_log_prob
from rl.rl_api.experiments import train_four_laser_inphase_all_symmetric_1000 as baseline

ROOT = Path(__file__).resolve().parents[3]
MODULES = (
    'train_four_laser_symmetric_pairscore_1000',
    'train_four_laser_symmetric_meanref_1000',
    'train_four_laser_symmetric_pairscore_meanref_1000',
)


def recipes():
    return [import_module('rl.rl_api.experiments.'+name) for name in MODULES]


def sync_module():
    spec = util.spec_from_file_location('comparison_sync', ROOT/'scripts/sync_great_lakes_plot.py')
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_two_hypotheses_and_progress_names_change(tmp_path):
    original = asdict(baseline.CONFIG)
    original_shapes = None
    filenames = []
    for recipe, score, mode in zip(recipes(), ['shared_pair','directed','shared_pair'],
                                   ['target_delay_residual','target_residual','target_residual']):
        candidate = asdict(recipe.CONFIG)
        assert candidate['sampling'].pop('symmetric_action_score') == score
        assert candidate['model'].pop('phase_parameterization') == mode
        filenames.append(candidate['validation'].pop('progress_filename'))
        expected = asdict(baseline.CONFIG)
        expected['sampling'].pop('symmetric_action_score')
        expected['model'].pop('phase_parameterization')
        expected['validation'].pop('progress_filename')
        assert candidate == expected
        assert recipe.INITIAL_CHECKPOINT is None
        backend = recipe.CONFIG.to_backend(tmp_path/recipe.RUN_NAME)
        assert backend.symmetric_coupling_values and backend.in_phase_targets
        assert recipe.CONFIG.summary()['simulations_per_iteration'] == 1520
        torch.manual_seed(7)
        policy = build_policy(backend)
        state = {key: value.detach().clone() for key,value in policy.state_dict().items()}
        if original_shapes is None:
            original_shapes = state
        else:
            for key in state:
                torch.testing.assert_close(state[key], original_shapes[key], rtol=0, atol=0)
    assert len(set(filenames)) == 3 and asdict(baseline.CONFIG) == original


def test_shared_score_ignores_discarded_directions_and_has_correct_mean_sd_gradients():
    mean = torch.tensor([[[.3,-.8,1.1,.4]]], dtype=torch.float64, requires_grad=True)
    sd = torch.tensor([[[.4,.9,.6,1.2]]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[[.7,-.1,.2,1.8]]], dtype=torch.float64)
    masks = torch.ones((1,1,2), dtype=torch.float64)
    score = reciprocal_normal_log_prob(values, mean, sd, masks, 2)
    shifted = values+torch.tensor([[[4.,-4.,-3.,3.]]], dtype=torch.float64)
    torch.testing.assert_close(score, reciprocal_normal_log_prob(shifted,mean,sd,masks,2),rtol=0,atol=1e-14)
    mean_grad, sd_grad = torch.autograd.grad(score.sum(), (mean,sd))
    a, b = [0,2], [1,3]
    difference = .5*(values[...,a]+values[...,b]-mean[...,a]-mean[...,b])
    variance = .25*(sd[...,a].square()+sd[...,b].square())
    expected_mean = difference/(2*variance)
    torch.testing.assert_close(mean_grad[...,a],expected_mean)
    torch.testing.assert_close(mean_grad[...,b],expected_mean)
    for index in (a,b):
        expected_sd = sd[...,index]/4*(difference.square()/variance.square()-1/variance)
        torch.testing.assert_close(sd_grad[...,index],expected_sd)
    assert reciprocal_normal_log_prob(values,mean,sd,torch.zeros_like(masks),2).item() == 0.
    with pytest.raises(ValueError,match='symmetric topology'):
        reciprocal_normal_log_prob(values,mean,sd,torch.tensor([[[1.,0.]]]),2)


def test_score_change_preserves_actions_rng_physical_matrices_and_budgets(tmp_path):
    config = recipes()[0].CONFIG.to_backend(tmp_path)
    torch.manual_seed(7)
    policy = build_policy(config)
    features = import_module('rl.rl_api._compat').engine.encode_context(
        np.zeros((1,4)),np.array([[-1.5,-.5,.5,1.5]]),policy)
    masks = torch.as_tensor(connected_symmetric_masks()[None],dtype=features.dtype).repeat_interleave(8,dim=1)
    results = []
    for score in ('directed','shared_pair'):
        policy.config = replace(config,symmetric_action_score=score)
        torch.manual_seed(1234)
        actions,budget,log_prob = sample_annealed_candidates(policy,features,masks,mirrored=True)
        rng_state = torch.random.get_rng_state().clone()
        decoded = decode_samples(actions,masks,policy.config,budget_actions=budget)
        results.append((actions,budget,log_prob,rng_state,decoded))
    first, second = results
    for index in (0,1,3):
        torch.testing.assert_close(first[index],second[index],rtol=0,atol=0)
    for a,b in zip(first[4],second[4]):
        np.testing.assert_array_equal(a,b)
    for matrix in second[4][:2]:
        np.testing.assert_array_equal(matrix,matrix.swapaxes(-1,-2))
    assert not torch.equal(first[2],second[2])
    gradient = torch.autograd.grad(second[2].sum(),policy.edge_output.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('variant',range(3))
def test_comparison_training_loading_and_unique_progress_images(tmp_path,fake_physics,variant):
    recipe = recipes()[variant]
    config = replace(recipe.CONFIG,
        sampling=replace(recipe.CONFIG.sampling,pairs_per_size=1),
        optimization=replace(recipe.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(recipe.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(recipe.CONFIG.validation,pairs_per_size=1,plot_use_tex=False,plot_dpi=35))
    directory = tmp_path/recipe.RUN_NAME/'job'
    result = Experiment(recipe.RUN_NAME,config=config,output_directory=directory).train(
        live_plot=True,verify_reference=False)
    assert len(result.history['training_reward']) == 2
    loaded = CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert loaded.backend_config.symmetric_action_score == config.sampling.symmetric_action_score
    assert loaded.backend_config.phase_parameterization == config.model.phase_parameterization
    for kappa,phase,_ in fake_physics:
        np.testing.assert_array_equal(kappa,kappa.swapaxes(-1,-2))
        np.testing.assert_array_equal(phase,phase.swapaxes(-1,-2))
    filename = config.validation.progress_filename
    assert (directory/filename).read_bytes() == (directory.parent/filename).read_bytes()
    assert not (directory/'current_training_progress.png').exists()


def test_three_named_submissions_match_recipes_and_sync_paths(tmp_path):
    launcher = ROOT/'scripts/submit_four_laser_symmetric_tests.sh'
    run = subprocess.run(['bash',str(launcher),'--smoke-test','--dry-run'],capture_output=True,text=True,check=True)
    commands = [shlex.split(line) for line in run.stdout.splitlines() if line.startswith('sbatch ')]
    assert len(commands) == 3
    names = []
    for command, module in zip(commands,MODULES):
        names.append(next(value for value in command if value.startswith('--job-name=')))
        assert '--time=01:00:00' in command and command[-1] == '--smoke-test'
        assert f'--export=ALL,VCSEL_TRAINING_MODULE={module}' in command
    assert len(set(names)) == 3
    sync = sync_module()
    args = sync.options(['--symmetric-tests','--destination',str(tmp_path)])
    paths = sync.remote_plots(args)
    for path, recipe in zip(paths,recipes()):
        assert path.parent.name == recipe.RUN_NAME
        assert path.name == recipe.CONFIG.validation.progress_filename
    calls = []
    def fake_run(command,**kwargs):
        calls.append(command)
        if command[0] == 'rsync':
            filename = Path(command[-2].split(':',1)[1]).name
            (Path(command[-1])/filename).write_bytes(filename.encode())
        return subprocess.CompletedProcess(command,0,stdout='',stderr='')
    with patch.object(sync.subprocess,'run',side_effect=fake_run):
        assert sync.main(['--symmetric-tests','--once','--destination',str(tmp_path)]) == 0
    assert sum('-fN' in command for command in calls) == 1
    assert sum(command[0]=='rsync' for command in calls) == 3
    assert len(list(tmp_path.glob('*.png'))) == 3
    with patch.object(sync.subprocess,'run',side_effect=AssertionError('dry run connects')):
        assert sync.main(['--symmetric-tests','--dry-run']) == 0


def test_auto_sync_default_and_remote_discovery_ignore_job_copies(tmp_path):
    sync=sync_module()
    assert sync.options([]).auto_discover
    assert sync.options(['--all-runs']).auto_discover
    assert not sync.options(['--run','one_run']).auto_discover
    assert not sync.options(['--fixed-context-tests']).auto_discover
    for args in (['--all-runs','--job-id','123'],['--run','']):
        with pytest.raises(SystemExit):
            sync.options(args)
    run=tmp_path/'test_run';run.mkdir()
    (run/'current_training_progress.png').write_bytes(b'plot')
    (run/'final_couplings.png').write_bytes(b'not a training plot')
    job=run/'123';job.mkdir();(job/'current_training_progress.png').write_bytes(b'duplicate')
    (run/'linked_progress.png').symlink_to(run/'current_training_progress.png')
    command=[sys.executable,'-c',sync.DISCOVERY_SCRIPT,str(tmp_path)]
    output=subprocess.run(command,capture_output=True,text=True,check=True).stdout
    assert json.loads(output.partition('=')[2]) == ['test_run/current_training_progress.png']
    new=tmp_path/'new_test';new.mkdir();(new/'new_progress.png').write_bytes(b'new')
    output=subprocess.run(command,capture_output=True,text=True,check=True).stdout
    assert len(json.loads(output.partition('=')[2])) == 2


def test_new_runs_are_synced_without_reauthenticating_or_overwriting(tmp_path,monkeypatch):
    import shutil
    sync=sync_module()
    remote=tmp_path/'remote';remote.mkdir()
    first=remote/'first';first.mkdir();(first/'current_training_progress.png').write_bytes(b'first')
    destination=tmp_path/'viewer'
    calls=[];polls=[]
    original_run=subprocess.run
    def fake_run(command,**kwargs):
        calls.append(command)
        if command[0]=='ssh' and '-c' in command[-1]:
            output=original_run([sys.executable,'-c',sync.DISCOVERY_SCRIPT,str(remote)],
                                capture_output=True,text=True,check=True).stdout
            return subprocess.CompletedProcess(command,0,stdout='login banner\n'+output,stderr='')
        if command[0]=='rsync':
            assert '--files-from=-' in command and '--from0' in command
            assert 'BatchMode=yes' in command[command.index('-e')+1]
            for name in kwargs['input'].split('\0'):
                if name:
                    target=Path(command[-1])/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(remote/name,target)
        return subprocess.CompletedProcess(command,0,stdout='',stderr='')
    def between_polls(seconds):
        polls.append(seconds)
        if len(polls)==1:
            second=remote/'second';second.mkdir()
            (second/'current_training_progress.png').write_bytes(b'second')
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(sync.subprocess,'run',fake_run)
    monkeypatch.setattr(sync.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(sync.time,'sleep',between_polls)
    assert sync.main(['--destination',str(destination)]) == 0
    assert (destination/'first__current_training_progress.png').read_bytes()==b'first'
    assert (destination/'second__current_training_progress.png').read_bytes()==b'second'
    assert sum('-fN' in command for command in calls)==1
    assert sum(command[0]=='rsync' for command in calls)==2
    assert polls==[10.,10.]
    assert not list(destination.glob('*.tmp'))


def test_remote_discovery_rejects_paths_outside_the_live_run_folder():
    sync=sync_module()
    args=sync.options([])
    for bad in ('../private_progress.png','/private_progress.png','one/two/progress.png'):
        def fake_run(command,**kwargs):
            return subprocess.CompletedProcess(command,0,
                stdout='VCSEL_PROGRESS_JSON='+json.dumps([bad])+'\n',stderr='')
        with patch.object(sync.subprocess,'run',side_effect=fake_run):
            with pytest.raises(RuntimeError,match='Invalid remote progress-image path'):
                sync.discover_plots(args,'socket')
