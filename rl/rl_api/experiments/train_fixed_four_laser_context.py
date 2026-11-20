"""Train one fixed four-laser case or continue it with narrow exploration.

The existing architecture, Gaussian sampler, decoder, simulator, RMS reward,
pair-excluded baseline and PCGrad update are reused. Only the context and
topology sampling are replaced. All model parameters remain trainable.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .. import CouplingModel
from .._compat import PACKAGE_DIR, core, engine, exported_settings
from ..advantages import pair_excluded_advantages
from ..checkpoints import save, write_json
from ..exploration import exploration_state, refinement_config, sample_annealed_candidates
from ..models import build_policy
from ..phase_reference import decode_samples
from ..plotting.training import save_training_progress
from ..root_context import calibrate_budget_references
from ..simulation import initialize_worker, simulate_indexed_locking_chunk
from ..training import set_learning_rate
from ..training_batch import pcgrad_update
from .four_laser_symmetric_comparison import comparison_config

# Receiver rows, source columns. Five reciprocal pairs; omit lasers 2 <-> 3.
TOPOLOGY = np.ones((4, 4), dtype=bool)
np.fill_diagonal(TOPOLOGY, False)
TOPOLOGY[1, 2] = TOPOLOGY[2, 1] = False
INITIAL_BUDGET_PER_LASER = {1: 4., 2: 6., 3: 9., 4: 12.}  # ns^-1; remains learnable.
DEFAULT_ACTIONS_PER_ITERATION = 128


def run_name(span, asymmetric=False, precision=False):
    if precision:
        return f'four_fixed_{span}ghz_precision'
    return f'four_fixed_fresh_{span}ghz' + ('_asym' if asymmetric else '')


def progress_filename(span, asymmetric=False, precision=False):
    suffix = '_precision' if precision else '_asym' if asymmetric else ''
    return f'four_fixed_{span}ghz_progress{suffix}.png'


def make_config(span, directory, *, iterations=1000, workers=1, seed=7,
                initial_budget_per_laser=None,
                actions_per_iteration=DEFAULT_ACTIONS_PER_ITERATION, asymmetric=False):
    if span not in (1, 2, 3, 4):
        raise ValueError('Choose a fixed span of 1, 2, 3 or 4 GHz')
    if iterations < 2 or workers < 1:
        raise ValueError('Need at least two iterations and one worker')
    if type(actions_per_iteration) is not int or actions_per_iteration < 8 or actions_per_iteration % 8:
        raise ValueError('Actions per iteration must be a positive multiple of eight')
    budget = INITIAL_BUDGET_PER_LASER[span] if initial_budget_per_laser is None else initial_budget_per_laser
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Initial budget per laser must be positive and finite')
    public = comparison_config(shared_pair_score=True, mean_reference=False,
        progress_filename=progress_filename(span, asymmetric))
    schedules = replace(public.schedules, ordinary_end=iterations, wide_end=iterations,
        wide_phase_actions=actions_per_iteration//4,
        learning_rate_decay_start=max(1, int(.9*iterations)), learning_rate_decay_end=iterations)
    public = replace(public, seed=seed, schedules=schedules,
        sampling=replace(public.sampling, actions_per_topology=actions_per_iteration),
        model=replace(public.model, initial_budget_per_laser=float(budget)),
        optimization=replace(public.optimization, iterations=iterations, workers=workers))
    # Retain the existing checkpoint schema. The explicit sampler
    # below supplies one context/mask instead of invoking the exhaustive bank.
    config = replace(public.to_backend(directory), targets_per_batch=1,
        pcgrad_topology_groups=1, candidates_per_target=actions_per_iteration,
        actions_per_topology=actions_per_iteration,
        targets_per_worker_task=1, candidates_per_worker_task=1,
        detuning_span_ghz=float(span), jupyter_mode=False)
    if asymmetric:
        # Every directed action is now executed, so score it independently
        # instead of scoring the reciprocal averages that symmetry discards.
        config = replace(config, symmetric_coupling_values=False, symmetric_action_score='directed')
    return config, schedules


def fixed_batch(policy, config, span, *, mean=False):
    targets = np.zeros((1, 4))
    detunings = np.linspace(-span/2, span/2, 4)[None]
    features = engine.encode_context(targets, detunings, policy)
    receiver, source = core.directed_link_indices(4)
    mask = torch.as_tensor(TOPOLOGY[receiver, source][None, None], dtype=features.dtype)
    gates = mask.repeat(1, 1 if mean else config.actions_per_topology, 1)
    if mean:
        with torch.no_grad():
            actions, budget, score = engine.sample_topology_conditioned_candidates(
                policy, features, gates, continuous_mean=True)
    else:
        actions, budget, score = sample_annealed_candidates(policy, features, gates, mirrored=True)
    kappa, phi, counts, rho, shortfall = decode_samples(actions, gates, config, budget_actions=budget)
    if config.symmetric_coupling_values:
        np.testing.assert_array_equal(kappa, kappa.swapaxes(-1, -2))
        np.testing.assert_array_equal(phi, phi.swapaxes(-1, -2))
    return dict(n_lasers=4, config=config, targets=targets, detunings=detunings,
        features=features, gates=gates, log_prob=score, kappa=kappa, phi=phi,
        counts=counts, rho=rho, shortfall=shortfall, reach=np.ones(counts.shape),
        roots=None, results=[])


def evaluate_batch(item, mean, config, pool):
    """Vectorize sampled and mean designs together; never average their rewards."""
    kappa = np.concatenate((item['kappa'], mean['kappa']), axis=1)
    phi = np.concatenate((item['phi'], mean['phi']), axis=1)
    total = kappa.shape[1]
    boundaries = np.linspace(0, total, min(config.n_jobs, total)+1, dtype=int)
    jobs = [(0, start, stop,
             (0, 0, 1, item['targets'], kappa[:,start:stop], phi[:,start:stop],
              item['detunings'], config)) for start, stop in zip(boundaries[:-1], boundaries[1:])]
    results = (pool.imap_unordered(simulate_indexed_locking_chunk, jobs) if pool is not None
               else map(simulate_indexed_locking_chunk, jobs))
    combined = {}
    coverage = np.zeros(total, int)
    for result in results:
        start, stop = result['candidate_start_index'], result['candidate_stop_index']
        coverage[start:stop] += 1
        for key, value in result.items():
            if not isinstance(value, np.ndarray):
                continue
            if key not in combined:
                combined[key] = np.empty((1, total, *value.shape[2:]), dtype=value.dtype)
            combined[key][:,start:stop] = value
    if not np.all(coverage == 1):
        raise RuntimeError('Every sampled and mean design must return exactly once')
    return combined


def mean_record(item, measured):
    row = dict(mean_reward=float(measured['phase_rms_reward'][0, -1]),
        mean_error_deg=float(measured['worst_phase_rms_error_deg'][0, -1]),
        within_3deg=bool(measured['phase_rms_success'][0, -1]),
        locked=bool(measured['locking_finite'][0, -1]
                    and measured['locking_phase_jitter_deg'][0, -1] < 1.
                    and measured['locking_frequency_spread_mhz'][0, -1] < 1.),
        phase_jitter_deg=float(measured['locking_phase_jitter_deg'][0, -1]),
        frequency_spread_mhz=float(measured['locking_frequency_spread_mhz'][0, -1]),
        budget_per_laser=float(item['kappa'].sum()/4))
    design = dict(detunings_ghz=item['detunings'][0].tolist(), target_phases_rad=[0.]*4,
        kappa_per_ns=item['kappa'][0, 0].tolist(), phi_p_rad=item['phi'][0, 0].tolist(), metrics=row.copy())
    return row, design


def progress_plot(history, directory, span, actions_per_iteration, asymmetric=False, precision=False):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['cmr10'],
                         'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        x = [row['iteration'] for row in history]
        axes[0].plot(x, [row['sample_reward'] for row in history], alpha=.5, label='sampled actions')
        axes[0].plot(x, [row['mean_reward'] for row in history], label='mean design')
        axes[0].set(xlabel='Iteration', ylabel='RMS reward', ylim=(0, 1.02))
        axes[0].legend()
        axes[1].plot(x, [row['mean_error_deg'] for row in history], label='mean design')
        axes[1].axhline(3, ls='--', color='tab:orange', label=r'$3^\circ$ target')
        axes[1].set(xlabel='Iteration', ylabel='Worst-laser RMS error (degrees)',
                    yscale='symlog', ylim=(0, 180))
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=.2)
        figure.suptitle(('Precision' if precision else 'Fresh')+f' four-laser model: fixed {span} GHz span, '
                       f'{actions_per_iteration//2} mirrored directions'
                       + (' (asymmetric)' if asymmetric else ''))
        save_training_progress(figure, directory, 200, filename=progress_filename(span, asymmetric, precision))
        plt.close(figure)


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--span-ghz', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--actions-per-iteration', type=int, default=DEFAULT_ACTIONS_PER_ITERATION,
        help='Sampled designs, excluding the central prediction; a multiple of eight (default: 128)')
    parser.add_argument('--asymmetric', action='store_true',
        help='Allow independent strengths and phases in each direction; suffix runs and figures with _asym')
    parser.add_argument('--precision-checkpoint', type=Path,
        help='Continue a symmetric fixed 4 GHz checkpoint with fresh Adam and narrow exploration')
    parser.add_argument('--initial-budget-per-laser', type=float,
        help='Initial learnable budget/M in ns^-1; defaults for spans 1/2/3/4 are 4/6/9/12')
    parser.add_argument('--workers', type=int,
        default=max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))-1))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--plot-every', type=int, default=10)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Two iterations using the real 500 ns solver')
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.iterations = 2
    if args.iterations < 2 or min(args.workers, args.plot_every) < 1 or args.seed < 0:
        parser.error('Need at least two iterations, positive workers/plot interval, and nonnegative seed')
    if args.actions_per_iteration < 8 or args.actions_per_iteration % 8:
        parser.error('--actions-per-iteration must be a positive multiple of eight')
    if args.initial_budget_per_laser is not None and (
            not np.isfinite(args.initial_budget_per_laser) or args.initial_budget_per_laser <= 0):
        parser.error('--initial-budget-per-laser must be positive and finite')
    if args.precision_checkpoint and (args.span_ghz != 4 or args.asymmetric
                                     or args.initial_budget_per_laser is not None):
        parser.error('Precision continuation uses symmetric 4 GHz values and preserves the checkpoint budget calibration')
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and args.workers + int(args.workers > 1) > int(allocated):
        parser.error('Simulation workers plus the trainer exceed allocated CPUs')
    return args


def main(argv=None):
    args = options(argv)
    torch.set_num_threads(1)
    span = args.span_ghz
    precision = args.precision_checkpoint is not None
    directory = args.output_directory or PACKAGE_DIR/'runs/great_lakes'/run_name(span, args.asymmetric, precision)/os.environ.get('SLURM_JOB_ID', 'local')
    rng = core.set_random_seed(args.seed)
    initialization = None
    if precision:
        from .fixed_four_laser_precision import load_precision_policy, iteration_config
        policy, config, initialization = load_precision_policy(args.precision_checkpoint, directory,
            iterations=args.iterations, workers=args.workers, actions=args.actions_per_iteration,
            topology=TOPOLOGY, seed=args.seed)
        schedules = None
        decay_end = args.iterations-1
    else:
        config, schedules = make_config(span, directory, iterations=args.iterations,
            workers=args.workers, seed=args.seed, initial_budget_per_laser=args.initial_budget_per_laser,
            actions_per_iteration=args.actions_per_iteration, asymmetric=args.asymmetric)
        policy = build_policy(config)
        config = calibrate_budget_references(policy, config)
        decay_end = schedules.learning_rate_decay_end
    policy.config = config
    fingerprint = hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes()
        for value in policy.state_dict().values())).hexdigest()
    recipe = dict(mode='fixed_context_precision_joint' if precision else 'fixed_context_fresh_joint',
        name=run_name(span, args.asymmetric, precision),
        experiment_revision='precision_continuation' if precision else 'exploration_batch_size',
        initialization='checkpoint weights; fresh optimizer' if precision else 'random weights; fresh optimizer and budget calibration',
        initial_checkpoint=initialization,
        initial_budget_per_laser=config.log_budget_initial_per_laser,
        budget_remains_learnable=True,
        initial_state_sha256=fingerprint, seed=args.seed, iterations=args.iterations,
        detunings_ghz=np.linspace(-span/2, span/2, 4).tolist(), span_ghz=span,
        target_phases_rad=[0.]*4, topology_receiver_source=TOPOLOGY.astype(int).tolist(), reciprocal_pairs=5,
        trainable='all original network parameters; strengths, phases and total budget',
        symmetric_strengths=not args.asymmetric, symmetric_phases=not args.asymmetric,
        teacher_labels=False, architecture_change=False,
        actions_per_iteration=args.actions_per_iteration,
        independent_exploration_directions=args.actions_per_iteration//2,
        ordinary_mirrored_pairs=args.actions_per_iteration//2 if precision else 3*args.actions_per_iteration//8,
        wide_mirrored_pairs=0 if precision else args.actions_per_iteration//8,
        simulations_per_iteration=args.actions_per_iteration+1,
        action_sampling='mirrored; all pairs narrow' if precision else 'mirrored; 75 percent ordinary pairs and 25 percent wide pairs',
        schedules='exploration scale .03 to .003 and LR 1e-5 to 1e-6 over updates; final mean evaluation' if precision else 'original full-run anneal to mean; learning-rate decay in final 10 percent',
        measurements='sampled actions and mean measured together before each update; checkpoints match measured means',
        phase_reference='laser 1 (lowest detuning), original target-delay residual',
        score='directed Gaussian' if args.asymmetric else 'reciprocal-pair Gaussian marginal',
        progress_filename=progress_filename(span, args.asymmetric, precision),
        metric_scope='same fixed context; no held-out or generalization claim',
        output_directory=str(directory), backend_config=exported_settings(config))
    import json
    print(json.dumps({k:v for k,v in recipe.items() if k != 'backend_config'}, indent=2), flush=True)
    if args.preview:
        return recipe
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'run.json', recipe)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    reward_scale = {}
    history = []
    best = -np.inf
    context = mp.get_context('spawn').Pool(args.workers, initializer=initialize_worker) if args.workers > 1 else nullcontext(None)
    with context as pool:
        for iteration in range(args.iterations):
            started = perf_counter()
            set_learning_rate(optimizer, config, iteration, decay_end=decay_end)
            current = iteration_config(config, iteration) if precision else refinement_config(config, schedules, iteration)
            policy.config = current
            policy.minimum_log_std = core.minimum_log_std_at(iteration, current)
            policy.maximum_log_std = core.maximum_log_std_at(iteration, current)
            policy.train()
            item = fixed_batch(policy, current, span)
            mean = fixed_batch(policy, current, span, mean=True)
            measured = evaluate_batch(item, mean, current, pool)
            reward = measured['phase_rms_reward'][:, :current.actions_per_topology]
            row, design = mean_record(mean, measured)
            row.update(iteration=iteration+1, sample_reward=float(reward.mean()),
                optimizer_update=current.exploration_std_scale != 0.,
                completed_optimizer_updates=iteration,
                exploration_scale=current.exploration_std_scale,
                learning_rate=float(optimizer.param_groups[0]['lr']),
                iteration_seconds=perf_counter()-started)
            history.append(row)
            state = exploration_state(policy)
            if iteration == 0:
                save(directory/'initial.pt', policy, optimizer, current, 0, row['mean_reward'],
                     sampling_recipe=recipe, exploration_state=state)
            save(directory/'current.pt', policy, optimizer, current, iteration+1,
                 row['mean_reward'], sampling_recipe=recipe, exploration_state=state)
            write_json(directory/'current_design.json', design)
            if row['mean_reward'] > best:
                best = row['mean_reward']
                save(directory/'best.pt', policy, optimizer, current, iteration+1,
                     best, sampling_recipe=recipe, exploration_state=state)
                write_json(directory/'best_design.json', design)
            if iteration == 0 or (iteration+1) % args.plot_every == 0 or iteration+1 == args.iterations:
                plot_options = dict(precision=True) if precision else {}
                progress_plot(history, directory, span, args.actions_per_iteration, args.asymmetric, **plot_options)
            advantages = pair_excluded_advantages([reward], current, reward_scale)
            diagnostics = pcgrad_update(policy, optimizer, current, rng, [item], advantages)
            row['gradient_norm'] = diagnostics['gradient_norm']
            policy.eval()
            row['iteration_seconds'] = perf_counter()-started
            write_json(directory/'current_history.json', dict(recipe=recipe, history=history))
            print(f"{iteration+1}/{args.iterations} RMS={row['mean_reward']:.5f} "
                  f"error={row['mean_error_deg']:.3f}deg within-3deg={row['within_3deg']} "
                  f"locked={row['locked']} budget/M={row['budget_per_laser']:.4f}ns^-1 "
                  f"time={row['iteration_seconds']:.2f}s", flush=True)
        save(directory/'final.pt', policy, optimizer, current, args.iterations,
             history[-1]['mean_reward'], sampling_recipe=recipe, exploration_state=state)
    print('Completed. Final iteration used exact means; optimizer update skipped.', flush=True)
    return CouplingModel(policy, current, metadata={'iteration':args.iterations}, output_directory=directory), history


if __name__ == '__main__':
    main()
