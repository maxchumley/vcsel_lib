"""Fresh fixed-context learning, single-task updates, and cluster launch paths."""
from dataclasses import replace
from importlib import util
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
import torch

from rl.rl_api import Context, CouplingModel, Topology
from rl.rl_api._compat import core
from rl.rl_api.experiments import train_fixed_four_laser_context as run
from rl.rl_api.models import build_policy
from rl.rl_api.training_batch import pcgrad_update
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_spans_share_fresh_weights_and_fix_only_context_and_mask(tmp_path):
    initial = None
    for span in (1, 2, 3, 4):
        config, _ = run.make_config(span, tmp_path/str(span))
        torch.manual_seed(7)
        policy = build_policy(config)
        assert all(p.requires_grad for p in policy.parameters())
        state = {key:value.detach().clone() for key,value in policy.state_dict().items()}
        if initial is None:
            initial = state
        else:
            for key in state:
                torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
        for _ in range(2):
            item = run.fixed_batch(policy, config, span)
            np.testing.assert_array_equal(item['detunings'], np.linspace(-span/2, span/2, 4)[None])
            assert (item['targets'] == 0).all() and np.all(item['counts'] == 10)
            for matrix in (item['kappa'], item['phi']):
                np.testing.assert_array_equal(matrix, matrix.swapaxes(-1, -2))
            receiver, source = core.directed_link_indices(4)
            np.testing.assert_array_equal(item['gates'][0, 0].numpy(), run.TOPOLOGY[receiver, source])
        # Strengths and budgets are explored, not frozen physical inputs.
        assert np.ptp(item['kappa'].sum((-1, -2))) > 0


def test_higher_initial_budget_is_the_only_training_setting_change(tmp_path):
    for span, expected_budget in ((1,4.), (2,6.), (3,9.), (4,12.)):
        high, high_schedule = run.make_config(span, tmp_path/str(span))
        low, low_schedule = run.make_config(span, tmp_path/str(span), initial_budget_per_laser=.5)
        assert replace(high, log_budget_initial_per_laser=.5) == low
        assert high_schedule == low_schedule
        assert high.log_budget_initial_per_laser == expected_budget
        models = []
        for config in (low,high):
            torch.manual_seed(7)
            policy = build_policy(config)
            models.append(run.fixed_batch(policy, config, span, mean=True))
        np.testing.assert_allclose(models[1]['kappa'], models[0]['kappa']*(expected_budget/.5), rtol=2e-6)
        np.testing.assert_array_equal(models[1]['phi'], models[0]['phi'])
        options = run.options(['--span-ghz',str(span),'--initial-budget-per-laser','.5'])
        assert options.initial_budget_per_laser == .5
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='positive and finite'):
            run.make_config(4, tmp_path, initial_budget_per_laser=invalid)


def test_larger_batch_preserves_settings_and_explores_64_mirrored_directions(tmp_path):
    small, small_schedule = run.make_config(4, tmp_path, actions_per_iteration=8)
    large, large_schedule = run.make_config(4, tmp_path)
    assert large.actions_per_topology == large.candidates_per_target == 128
    assert replace(large, actions_per_topology=8, candidates_per_target=8) == small
    assert replace(large_schedule, wide_phase_actions=2) == small_schedule
    assert large.phase_exploration_fraction == small.phase_exploration_fraction == .25
    torch.manual_seed(7)
    policy = build_policy(large)
    item = run.fixed_batch(policy, large, 4)
    actions, budget, score = run.sample_annealed_candidates(policy, item['features'], item['gates'], mirrored=True)
    mean, budget_mean, _ = run.engine.sample_topology_conditioned_candidates(
        policy, item['features'], item['gates'], continuous_mean=True)
    torch.testing.assert_close(actions[:, 1::2], 2*mean[:, ::2]-actions[:, ::2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(budget[:, 1::2], 2*budget_mean[:, ::2]-budget[:, ::2], rtol=1e-5, atol=1e-6)
    assert score.shape == (1,128) and torch.isfinite(score).all()
    assert torch.unique(actions[:, ::2].reshape(64,-1), dim=0).shape[0] == 64
    for invalid in (0, 4, 12, 130, 128., True):
        with pytest.raises(ValueError, match='multiple of eight'):
            run.make_config(4, tmp_path, actions_per_iteration=invalid)


def test_asymmetric_mode_executes_directed_values_without_changing_recipe(tmp_path, monkeypatch):
    symmetric, symmetric_schedule = run.make_config(4, tmp_path)
    asymmetric, asymmetric_schedule = run.make_config(4, tmp_path, asymmetric=True)
    assert replace(asymmetric, symmetric_coupling_values=True, symmetric_action_score='shared_pair') == symmetric
    assert asymmetric_schedule == symmetric_schedule
    models = []
    for config in (symmetric, asymmetric):
        torch.manual_seed(7)
        models.append(build_policy(config))
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
    draws = {}
    original = run.sample_annealed_candidates
    def sample(*args, **kwargs):
        actions, budget, score = original(*args, **kwargs)
        draws.update(actions=actions, budget=budget)
        return actions, budget, score
    monkeypatch.setattr(run, 'sample_annealed_candidates', sample)
    item = run.fixed_batch(models[1], asymmetric, 4)
    expected = run.engine.decode_samples(draws['actions'], item['gates'], asymmetric,
                                         budget_actions=draws['budget'])
    for key, value in zip(('kappa','phi','counts','rho','shortfall'), expected):
        np.testing.assert_array_equal(item[key], value)
    for key in ('kappa','phi'):
        assert not np.allclose(item[key], item[key].swapaxes(-1,-2))
    symmetric_mean = run.fixed_batch(models[0], symmetric, 4, mean=True)
    asymmetric_mean = run.fixed_batch(models[1], asymmetric, 4, mean=True)
    np.testing.assert_array_equal(symmetric_mean['gates'], asymmetric_mean['gates'])
    np.testing.assert_allclose(symmetric_mean['kappa'].sum(), asymmetric_mean['kappa'].sum(), rtol=1e-6)


def test_large_batch_uses_all_workers_and_restores_unordered_results(tmp_path, fake_physics):
    config, _ = run.make_config(4, tmp_path, workers=31)
    torch.manual_seed(7)
    policy = build_policy(config)
    item = run.fixed_batch(policy, config, 4)
    mean = run.fixed_batch(policy, config, 4, mean=True)
    jobs = []
    class ReversePool:
        def imap_unordered(self, function, work):
            jobs.extend(work)
            return map(function, reversed(work))
    measured = run.evaluate_batch(item, mean, config, ReversePool())
    assert len(jobs) == 31
    assert {job[2]-job[1] for job in jobs} == {4,5}
    expected = run.evaluate_batch(item, mean, replace(config, n_jobs=1), None)
    assert measured['phase_rms_reward'].shape == (1,129)
    for key in expected:
        np.testing.assert_allclose(measured[key], expected[key], rtol=0, atol=0)


def test_one_task_uses_clipped_gradient_without_pcgrad_projection(tmp_path):
    config, _ = run.make_config(4, tmp_path, actions_per_iteration=8)
    config = replace(config, per_size_gradient_clip=.5, gradient_clip=10.)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([.2, -.4]))
            self.log_std_kappa = torch.nn.Parameter(torch.tensor(0.))
            self.log_std_phi = torch.nn.Parameter(torch.tensor(0.))
            self.minimum_log_std = -1.5
            self.maximum_log_std = .75
    policy = Policy()
    before = policy.value.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=.2)
    score = policy.value[0]*torch.arange(1., 9.)[None] + policy.value[1]
    item = dict(log_prob=score, features=torch.zeros((1,4,4)),
                n_lasers=4, counts=np.full((1,8), 10))
    result = pcgrad_update(policy, optimizer, config, np.random.default_rng(7), [item], [np.ones((1,8))])
    direction = torch.tensor([4.5, 1.])
    expected = before+.2*.5*direction/torch.linalg.vector_norm(direction)
    torch.testing.assert_close(policy.value, expected)
    assert result['pcgrad_task_count'] == 1
    assert abs(result['gradient_norm']-.5) < 1e-6


@pytest.mark.parametrize('asymmetric', [False, True])
def test_training_updates_strength_phase_budget_and_roundtrips(tmp_path, fake_physics, monkeypatch, asymmetric):
    figures = []
    monkeypatch.setattr(run, 'progress_plot', lambda h,d,s,a,u: figures.append((len(h), s, a, u)))
    directory = tmp_path/'fresh'
    model, history = run.main(['--span-ghz','4','--iterations','3','--workers','1',
                               '--output-directory',str(directory)]+(['--asymmetric'] if asymmetric else []))
    initial = CouplingModel.load(directory/'initial.pt', device='cpu')
    final = CouplingModel.load(directory/'final.pt', device='cpu')
    for key in ['node_encoder.0.weight', 'edge_output.weight', 'budget_head.2.weight']:
        assert not torch.equal(initial.policy.state_dict()[key], final.policy.state_dict()[key])
    for row in (0,1):
        assert not torch.equal(initial.policy.edge_output.weight[row], final.policy.edge_output.weight[row])
    manifest = json.loads((directory/'run.json').read_text())
    assert manifest['initial_checkpoint'] is None and not manifest['teacher_labels']
    assert manifest['initial_budget_per_laser'] == 12. and manifest['budget_remains_learnable']
    assert manifest['actions_per_iteration'] == 128 and manifest['simulations_per_iteration'] == 129
    assert manifest['independent_exploration_directions'] == 64
    assert manifest['ordinary_mirrored_pairs'] == 48 and manifest['wide_mirrored_pairs'] == 16
    assert manifest['symmetric_strengths'] == manifest['symmetric_phases'] == (not asymmetric)
    assert manifest['name'] == run.run_name(4, asymmetric)
    assert manifest['progress_filename'] == run.progress_filename(4, asymmetric)
    assert final.backend_config.symmetric_coupling_values == (not asymmetric)
    assert final.backend_config.symmetric_action_score == ('directed' if asymmetric else 'shared_pair')
    assert not manifest['architecture_change']
    assert [row['optimizer_update'] for row in history] == [True, True, False]
    payload = torch.load(directory/'final.pt', weights_only=False)
    assert max(int(value['step']) for value in payload['optimizer_state_dict']['state'].values()) == 2
    saved = json.loads((directory/'current_design.json').read_text())
    design = final.propose(Context(np.zeros(4), np.linspace(-2.,2.,4)),
        topology=Topology.from_receiver_source(run.TOPOLOGY), count=1)
    np.testing.assert_allclose(design.coupling_strengths_per_ns[0], saved['kappa_per_ns'], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(design.coupling_phases_rad[0], saved['phi_p_rad'], rtol=0, atol=2e-6)
    assert sum(len(kappa) for kappa,_,_ in fake_physics) == 3*129
    assert figures[-1] == (3,4,128,asymmetric)


def test_four_named_jobs_match_viewer_paths_without_checkpoint_inputs(tmp_path):
    launcher = ROOT/'scripts/submit_fixed_four_laser_tests.sh'
    result = subprocess.run(['bash',str(launcher),'--dry-run','--smoke-test'],
                            capture_output=True,text=True,check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
    assert len(commands) == 4
    spec = util.spec_from_file_location('fixed_context_sync',ROOT/'scripts/sync_great_lakes_plot.py')
    sync = util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    paths = sync.remote_plots(sync.options(['--fixed-context-tests','--destination',str(tmp_path)]))
    for span, command, path in zip((1,2,3,4), commands, paths):
        assert f'--job-name=four_dirs64_{span}GHz' in command
        assert command[command.index('--span-ghz')+1] == str(span)
        assert command[command.index('--actions-per-iteration')+1] == '128'
        assert not any(arg.startswith('--time') for arg in command)
        assert '--checkpoint' not in command and '--initial-checkpoint' not in command
        assert command[-1] == '--smoke-test'
        assert path.parent.name == run.run_name(span)
        assert path.name == f'four_fixed_{span}ghz_progress.png'
    subprocess.run(['bash','-n',str(launcher),str(ROOT/'scripts/train_fixed_four_laser_great_lakes.slurm')],check=True)
    slurm = (ROOT/'scripts/train_fixed_four_laser_great_lakes.slurm').read_text()
    assert '#SBATCH --time=2-00:00:00' in slurm
    assert '#SBATCH --cpus-per-task=32' in slurm and '#SBATCH --mem=64G' in slurm
    control = subprocess.run(['bash',str(launcher),'--dry-run','--actions-per-iteration','8'],
                             capture_output=True,text=True,check=True)
    assert '--job-name=four_dirs4_4GHz' in control.stdout
    result = subprocess.run(['bash',str(launcher),'--dry-run','--asymmetric','--smoke-test'],
                            capture_output=True,text=True,check=True)
    asymmetric_commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
    all_paths = sync.remote_plots(sync.options(['--fixed-context-tests','--include-asymmetric']))
    assert all_paths[:4] == paths
    assert len(all_paths) == len({path.name for path in all_paths}) == 8
    for span, command, path in zip((1,2,3,4), asymmetric_commands, all_paths[4:]):
        assert f'--job-name=four_dirs64_{span}GHz_asym' in command
        assert '--asymmetric' in command and command[-1] == '--smoke-test'
        assert path.parent.name == run.run_name(span, asymmetric=True)
        assert path.name == run.progress_filename(span, asymmetric=True)
        assert path.name.endswith('_asym.png')
    with pytest.raises(SystemExit):
        sync.options(['--include-asymmetric'])
