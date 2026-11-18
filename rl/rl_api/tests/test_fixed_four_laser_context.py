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


def test_one_task_uses_clipped_gradient_without_pcgrad_projection(tmp_path):
    config, _ = run.make_config(4, tmp_path)
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


def test_training_updates_strength_phase_budget_and_roundtrips(tmp_path, fake_physics, monkeypatch):
    figures = []
    monkeypatch.setattr(run, 'progress_plot', lambda h,d,s: figures.append((len(h), s)))
    directory = tmp_path/'fresh'
    model, history = run.main(['--span-ghz','4','--iterations','3','--workers','1',
                               '--output-directory',str(directory)])
    initial = CouplingModel.load(directory/'initial.pt', device='cpu')
    final = CouplingModel.load(directory/'final.pt', device='cpu')
    for key in ['node_encoder.0.weight', 'edge_output.weight', 'budget_head.2.weight']:
        assert not torch.equal(initial.policy.state_dict()[key], final.policy.state_dict()[key])
    for row in (0,1):
        assert not torch.equal(initial.policy.edge_output.weight[row], final.policy.edge_output.weight[row])
    manifest = json.loads((directory/'run.json').read_text())
    assert manifest['initial_checkpoint'] is None and not manifest['teacher_labels']
    assert not manifest['architecture_change']
    assert [row['optimizer_update'] for row in history] == [True, True, False]
    payload = torch.load(directory/'final.pt', weights_only=False)
    assert max(int(value['step']) for value in payload['optimizer_state_dict']['state'].values()) == 2
    saved = json.loads((directory/'current_design.json').read_text())
    design = final.propose(Context(np.zeros(4), np.linspace(-2.,2.,4)),
        topology=Topology.from_receiver_source(run.TOPOLOGY), count=1)
    np.testing.assert_allclose(design.coupling_strengths_per_ns[0], saved['kappa_per_ns'], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(design.coupling_phases_rad[0], saved['phi_p_rad'], rtol=0, atol=2e-6)
    assert sum(len(kappa) for kappa,_,_ in fake_physics) == 3*9
    assert figures[-1] == (3,4)


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
        assert f'--job-name=four_fixed_{span}GHz' in command
        assert command[command.index('--span-ghz')+1] == str(span)
        assert '--checkpoint' not in command and '--initial-checkpoint' not in command
        assert command[-1] == '--smoke-test'
        assert path.parent.name == run.run_name(span)
        assert path.name == f'four_fixed_{span}ghz_progress.png'
    subprocess.run(['bash','-n',str(launcher),str(ROOT/'scripts/train_fixed_four_laser_great_lakes.slurm')],check=True)
