"""Reference equivalence, including RNG sequence, Adam state and checkpoints."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from rl.rl_api import (Experiment, TrainingConfig, OptimizationConfig,
                       ValidationConfig, SamplingConfig, ScheduleConfig)
from rl.rl_api._compat import core, engine
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import sample_pairs
from rl.rl_api.training import set_learning_rate


def assert_equal(first, second):
    """Compare nested histories/checkpoint tensors, including NaN placeholders."""
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_equal(a, b)
    elif isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_equal(first, second)


def test_all_reference_settings_match_except_output_paths(tmp_path):
    from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import config
    experiment = Experiment("parity", output_directory=tmp_path / "api")
    api = asdict(experiment.backend_config)
    old = asdict(config)
    different = {key for key in api if api[key] != old[key]}
    assert different == {f"{kind}_checkpoint_file" for kind in ("best", "current", "final")}
    assert not experiment.directory.exists()
    summary = experiment.describe()
    assert summary["simulations_per_iteration"] == 1400
    assert summary["pcgrad_tasks"] == summary["worker_jobs"] == 28
    assert summary["detuning_bin_width_ghz"] == .8
    assert not experiment.directory.exists()


def test_model_initialization_and_parameter_names_are_identical(tmp_path):
    config = Experiment("weights", output_directory=tmp_path / "api").backend_config
    torch.manual_seed(7)
    original = engine.BernoulliSparseGNNPolicy(config)
    torch.manual_seed(7)
    api = build_policy(config)
    assert_equal(original.state_dict(), api.state_dict())
    assert sum(p.numel() for p in api.parameters()) == 127304


def test_source_guard_accepts_reviewed_integrator_but_rejects_unreviewed_edits(tmp_path, monkeypatch):
    from rl.rl_api import _compat
    current = (_compat.REPOSITORY_DIR / 'vcsel_lib.py').read_bytes()
    copy = tmp_path / 'vcsel_lib.py'
    copy.write_bytes(current)
    monkeypatch.setattr(_compat, 'REPOSITORY_DIR', tmp_path)
    assert _compat.reference_source_status()['vcsel_lib.py'] is True
    copy.write_bytes(current + b'\n# unreviewed source edit\n')
    assert _compat.reference_source_status()['vcsel_lib.py'] is False


@pytest.mark.parametrize("size", range(3,10))
def test_pair_sampler_and_next_random_draw_match_reference(size, tmp_path):
    config = Experiment("sampling", output_directory=tmp_path).backend_config
    first = np.random.default_rng(21)
    second = np.random.default_rng(21)
    sampled = sample_pairs(config, first, n_lasers=size)
    local = replace(config, n_lasers=size)
    targets = core.sample_target_phases(5, second, size, 0., config.structured_target_jitter_rad)
    detuning = core.sample_detuning_distributions(5, second, local, iteration=0)
    targets, detuning = core.sort_by_detuning(targets, detuning)
    masks = engine.sample_paired_rooted_topologies(5,4,size,second,torch.device("cpu"),
        min_fraction=0., density_bias=1., fixed_star=True)
    np.testing.assert_array_equal(sampled.target_phases_rad, targets)
    np.testing.assert_array_equal(sampled.detunings_ghz, detuning)
    torch.testing.assert_close(sampled.edge_masks, masks, rtol=0, atol=0)
    assert first.random() == second.random()


@pytest.mark.parametrize("sizes", [(3,), (3,5,9)])
def test_three_full_updates_match_reference_weights_adam_and_histories(tmp_path, monkeypatch, fake_physics, sizes):
    """Physics is mocked, but sampling, rewards, backprop, PCGrad and Adam are real."""
    settings = TrainingConfig(sizes=sizes, optimization=OptimizationConfig(iterations=3,workers=1),
                              validation=ValidationConfig(every=2,pairs_per_size=5))
    experiment = Experiment("api", config=settings, output_directory=tmp_path / "api")
    reference_directory = tmp_path / "reference"
    source_config = replace(experiment.backend_config,
        best_checkpoint_file=str(reference_directory / "best.pt"),
        current_checkpoint_file=str(reference_directory / "current.pt"),
        final_checkpoint_file=str(reference_directory / "final.pt"))
    monkeypatch.setattr(engine, "RESULTS_DIR", reference_directory)
    reference_policy, reference_history = engine.train_policy(source_config, live_plot=False)
    source_numpy_state = np.random.get_state()
    source_torch_state = torch.get_rng_state().clone()
    result = experiment.train(live_plot=False)
    assert_equal({key: reference_history[key] for key in result.history}, result.history)
    assert_equal(reference_policy.state_dict(), result.model.policy.state_dict())
    assert_equal(source_numpy_state, np.random.get_state())
    torch.testing.assert_close(source_torch_state, torch.get_rng_state(), rtol=0, atol=0)
    for kind in ("best", "current", "final"):
        source = torch.load(reference_directory / f"{kind}.pt", weights_only=False)
        api = torch.load(result.checkpoint(kind), weights_only=False)
        assert source["iteration"] == api["iteration"]
        assert source["held_out_reward"] == api["held_out_reward"]
        assert_equal(source["policy_state_dict"], api["policy_state_dict"])
        assert_equal(source["optimizer_state_dict"], api["optimizer_state_dict"])
        assert not any(key.startswith('de_teacher_') for key in api['config'])
        # The original loader reconstructs omitted disabled defaults.
        loaded, loaded_config, _ = engine.load_checkpoint(result.checkpoint(kind))
        assert_equal(result.model.policy.state_dict() if kind=='final'
                     else api['policy_state_dict'], loaded.state_dict())
        assert loaded_config.de_teacher_enabled is False
    saved = json.loads((result.directory / "current_history.json").read_text())
    old_history=json.loads((reference_directory / "current_history.json").read_text())["history"]
    assert_equal(saved["history"], {key:old_history[key] for key in saved['history']})
    assert not any(key.startswith('de_teacher_') for key in saved['config'])
    assert not any(key.startswith('de_teacher_') for key in saved['history'])
    assert (result.directory / "run.json").is_file()
    manifest=json.loads((result.directory/'run.json').read_text())
    assert not any(key.startswith('de_teacher_') for key in manifest['backend_config'])


@pytest.mark.parametrize("printed", [1,999,1000,1250,1499,1500,2000,2001,2500,3000])
def test_schedule_values_use_the_same_reference_boundaries(printed, tmp_path):
    config = Experiment("stages", output_directory=tmp_path).backend_config
    iteration = printed-1
    model = build_policy(config)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    set_learning_rate(optimizer,config,iteration)
    expected = config.learning_rate
    if iteration >= config.learning_rate_decay_start_iteration:
        fraction = min((iteration-config.learning_rate_decay_start_iteration)/
            max(config.training_iterations-config.learning_rate_decay_start_iteration-1,1),1)
        expected *= (config.learning_rate_final/config.learning_rate)**fraction
    assert optimizer.param_groups[0]["lr"] == expected
    assert engine.phase_exploration_at(1499,config) == 0.
    assert core.minimum_log_std_at(1499,config) == -3.
    assert core.maximum_log_std_at(1499,config) == -1.5


def test_existing_run_is_never_overwritten(tmp_path):
    experiment = Experiment("protected",output_directory=tmp_path)
    sentinel=tmp_path/"current.pt";sentinel.write_text("do not change")
    with pytest.raises(FileExistsError):experiment.train(live_plot=False)
    assert sentinel.read_text()=="do not change"


def test_unsupported_bernoulli_mode_is_explicit():
    from rl.rl_api import TopologyConfig
    with pytest.raises(NotImplementedError):Experiment("future",preset="bernoulli_gates")
    with pytest.raises(NotImplementedError):TopologyConfig(mode="bernoulli")


def test_training_owns_one_spawn_pool_and_closes_it(tmp_path,monkeypatch,fake_physics):
    """Exercise all seven sizes and the pool contract without starting workers."""
    from rl.rl_api import training
    events=[]
    class Pool:
        def __init__(self,processes,initializer):
            assert processes==14
            assert initializer is core._initialize_simulation_worker
            events.append('created')
        def imap_unordered(self,function,items):
            return reversed([function(item) for item in items])
        def close(self):events.append('closed')
        def join(self):events.append('joined')
    class Context:
        def Pool(self, **options):return Pool(**options)
    def get_context(method):
        assert method=='spawn'
        return Context()
    monkeypatch.setattr(training.mp,'get_context',get_context)
    settings=TrainingConfig(optimization=OptimizationConfig(iterations=1,workers=14),
                            validation=ValidationConfig(pairs_per_size=4))
    result=Experiment('pool',config=settings,output_directory=tmp_path/'pool').train(live_plot=False)
    assert events==['created','closed','joined']
    assert result.history['pcgrad_task_count']==[28]
    import matplotlib.pyplot as plt
    plt.close(result.plot(show=False))
