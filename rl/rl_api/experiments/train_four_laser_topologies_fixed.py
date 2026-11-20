"""Learn all 38 symmetric four-laser topologies at one fixed detuning vector."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
import torch

from .. import CouplingModel
from .._compat import PACKAGE_DIR, core, exported_settings
from ..advantages import pair_excluded_advantages
from ..checkpoints import save, write_json
from ..exploration import exploration_state
from ..plotting.training import save_training_progress
from ..simulation import initialize_worker
from ..symmetric_topologies import connected_symmetric_masks, topology_metadata
from ..training_batch import pcgrad_update
from .train_fixed_four_laser_context import fixed_batch, evaluate_batch


def run_name(span):
    return f'four_all38_fixed_{span}ghz'


def progress_filename(span):
    return f'four_all38_fixed_{span}ghz_progress.png'


def load_start(path, directory, span, workers, actions, seed):
    """Preserve learned outputs and physical calibration; create no optimizer."""
    path = Path(path).expanduser().resolve()
    model = CouplingModel.load(path, device='cpu', output_directory=directory)
    source = model.backend_config
    recipe = model.metadata.get('sampling_recipe') or {}
    if (recipe.get('mode') not in {'fixed_context_fresh_joint','fixed_context_precision_joint','fixed_detunings_all_topologies'}
            or recipe.get('span_ghz') != span
            or not np.array_equal(recipe.get('target_phases_rad'), np.zeros(4))
            or not np.allclose(recipe.get('detunings_ghz', []), np.linspace(-span/2,span/2,4))
            or source.n_lasers != 4 or not getattr(source,'symmetric_coupling_values',False)
            or source.symmetric_action_score != 'shared_pair' or source.objective_mode != 'phase_only'
            or source.phase_parameterization != 'target_delay_residual'):
        raise ValueError('Use a symmetric in-phase four-laser checkpoint with the matching fixed detunings')
    config = replace(source, n_jobs=workers, random_seed=seed, targets_per_batch=1,
        pcgrad_topology_groups=38, candidates_per_target=38*actions, actions_per_topology=actions,
        targets_per_worker_task=1, candidates_per_worker_task=32,
        exploration_std_scale=1., phase_exploration_fraction=.25, phase_exploration_start_std=np.pi,
        learning_rate=1e-4, learning_rate_final=1e-5,
        enable_exploration_annealing=False, enable_minimum_log_std_annealing=False,
        minimum_log_std=model.policy.minimum_log_std, maximum_log_std=model.policy.maximum_log_std,
        jupyter_mode=False, best_checkpoint_file=str(directory/'best.pt'),
        current_checkpoint_file=str(directory/'current.pt'), final_checkpoint_file=str(directory/'final.pt'))
    model.policy.config = config
    initialization = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_iteration=model.metadata['iteration'], source_reward=model.metadata['held_out_reward'],
        copied='policy weights, physical settings and budget calibration',
        reset='optimizer, exploration schedule, reward scaling, sampler seed and history')
    return model.policy, config, initialization


def iteration_settings(config, iteration, search_updates, precision_updates):
    if iteration < search_updates:
        progress = iteration/max(search_updates-1,1)
        rate_progress = np.clip((progress-.9)/.1,0.,1.)
        return (replace(config, exploration_std_scale=1.-.97*progress,
                        phase_exploration_fraction=.25, phase_exploration_start_std=np.pi),
                1e-4*.1**rate_progress, 'search')
    if iteration < search_updates+precision_updates:
        progress = (iteration-search_updates)/max(precision_updates-1,1)
        return (replace(config, exploration_std_scale=.03*.1**progress,
                        phase_exploration_fraction=0., phase_exploration_start_std=0.),
                1e-5*.1**progress, 'precision')
    return replace(config, exploration_std_scale=0.,phase_exploration_fraction=0.,phase_exploration_start_std=0.),1e-6,'final'


def mean_records(mean, measured, pair_counts):
    """One deterministic measurement per graph, kept out of policy gradients."""
    count = len(pair_counts)
    rewards = measured['phase_rms_reward'][0,-count:]
    errors = measured['worst_phase_rms_error_deg'][0,-count:]
    success = measured['phase_rms_success'][0,-count:]
    finite = measured['locking_finite'][0,-count:]
    jitter = measured['locking_phase_jitter_deg'][0,-count:]
    spread = measured['locking_frequency_spread_mhz'][0,-count:]
    locked = finite & (jitter < 1.) & (spread < 1.)
    row = dict(mean_reward=float(rewards.mean()),median_error_deg=float(np.median(errors)),
        worst_error_deg=float(errors.max()),mean_error_deg=float(errors.mean()),
        phase_success_count=int(success.sum()),locked_success_count=int((success & locked).sum()),
        locked_count=int(locked.sum()),budget_per_laser=float(mean['kappa'].sum((-2,-1)).mean()/4))
    designs = []
    for index,pairs in enumerate(pair_counts):
        designs.append(dict(topology_id=index,pair_count=int(pairs),rms_reward=float(rewards[index]),
            worst_phase_rms_deg=float(errors[index]),within_3deg=bool(success[index]),locked=bool(locked[index]),
            phase_jitter_deg=float(jitter[index]),frequency_spread_mhz=float(spread[index]),
            budget_per_ns=float(mean['kappa'][0,index].sum()),
            kappa_per_ns=mean['kappa'][0,index].tolist(),phi_p_rad=mean['phi'][0,index].tolist()))
    row['by_pair_count'] = {str(pairs):dict(count=int(np.sum(pair_counts==pairs)),
        mean_reward=float(rewards[pair_counts==pairs].mean()),
        worst_error_deg=float(errors[pair_counts==pairs].max()),
        locked_success_count=int((success & locked)[pair_counts==pairs].sum())) for pairs in (3,4,5,6)}
    return row,designs


def progress_plot(history, directory, span, search_updates):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
                         'axes.formatter.use_mathtext':True}):
        figure,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        x = [row['iteration'] for row in history]
        axes[0].plot(x,[row['sample_reward'] for row in history],alpha=.5,label='sampled actions')
        axes[0].plot(x,[row['mean_reward'] for row in history],label='mean designs')
        axes[0].set(xlabel='Iteration',ylabel='RMS reward',ylim=(0,1.02))
        for key,label in (('median_error_deg','median topology'),('worst_error_deg','worst topology')):
            errors = np.nan_to_num([row[key] for row in history],nan=180.,posinf=180.)
            axes[1].plot(x,errors,label=label)
        axes[1].axhline(3.,ls='--',color='gray',label=r'$3^\circ$ target')
        axes[1].set(xlabel='Iteration',ylabel='Worst-laser RMS error (degrees)',yscale='symlog',ylim=(0,180))
        for axis in axes:
            if len(history)>search_updates:
                axis.axvline(search_updates+.5,color='gray',ls=':',alpha=.6)
            axis.grid(alpha=.2);axis.legend()
        figure.suptitle(f'All 38 symmetric topologies: fixed {span} GHz, '+history[-1]['stage'])
        save_training_progress(figure,directory,200,filename=progress_filename(span))
        plt.close(figure)


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--span-ghz',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--search-updates',type=int,default=1000)
    parser.add_argument('--precision-updates',type=int,default=1000)
    parser.add_argument('--actions-per-topology',type=int,default=128)
    parser.add_argument('--cases-per-job',type=int,default=32)
    parser.add_argument('--workers',type=int,default=max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','2'))-1))
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output-directory',type=Path)
    parser.add_argument('--plot-every',type=int,default=10)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args(argv)
    if args.smoke_test:args.search_updates=args.precision_updates=1
    if min(args.search_updates,args.precision_updates,args.workers,args.cases_per_job,args.plot_every)<1 or args.seed<0:
        parser.error('Update counts, workers, simulation chunks and plot interval must be positive; seed must be nonnegative')
    if args.actions_per_topology<8 or args.actions_per_topology%8:
        parser.error('--actions-per-topology must be a positive multiple of eight')
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and args.workers+int(args.workers>1)>int(allocated):
        parser.error('Simulation workers plus the trainer exceed allocated CPUs')
    return args


def main(argv=None):
    args = options(argv);torch.set_num_threads(1)
    directory = args.output_directory or PACKAGE_DIR/'runs/great_lakes'/run_name(args.span_ghz)/os.environ.get('SLURM_JOB_ID','local')
    rng = core.set_random_seed(args.seed)
    policy,config,initialization = load_start(args.initial_checkpoint,directory,args.span_ghz,
                                             args.workers,args.actions_per_topology,args.seed)
    total = args.search_updates+args.precision_updates+1
    config = replace(config,training_iterations=total)
    bank = connected_symmetric_masks();pair_counts = bank.sum(1)//2
    recipe = dict(mode='fixed_detunings_all_topologies',name=run_name(args.span_ghz),
        initial_checkpoint=initialization,span_ghz=args.span_ghz,
        detunings_ghz=np.linspace(-args.span_ghz/2,args.span_ghz/2,4).tolist(),target_phases_rad=[0.]*4,
        topology_bank=topology_metadata(),symmetric_strengths=True,symmetric_phases=True,
        actions_per_topology=args.actions_per_topology,independent_directions_per_topology=args.actions_per_topology//2,
        sampled_designs_per_iteration=38*args.actions_per_topology,mean_designs_per_iteration=38,
        simulations_per_iteration=38*(args.actions_per_topology+1),
        search_updates=args.search_updates,precision_updates=args.precision_updates,iterations=total,
        stages='broad search; restore best full-bank checkpoint and fresh Adam; narrow precision; final mean evaluation',
        objective='existing phase-only wrapped RMS, reciprocal-pair score and pair-excluded baseline',
        pcgrad='existing three topology-density groups',trainable='all original network parameters, strengths, phases and budget',
        teacher_labels=False,architecture_change=False,seed=args.seed,cases_per_job=args.cases_per_job,
        progress_filename=progress_filename(args.span_ghz),metric_scope='all 38 trained topologies at one fixed detuning vector',
        output_directory=str(directory),backend_config=exported_settings(config))
    import json
    print(json.dumps({k:v for k,v in recipe.items() if k not in {'backend_config','topology_bank'}},indent=2),flush=True)
    if args.preview:return recipe
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {directory}')
    directory.mkdir(parents=True,exist_ok=True);write_json(directory/'run.json',recipe)
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
    history=[];reward_scale={};best=-np.inf;best_worst=np.inf
    context = mp.get_context('spawn').Pool(args.workers,initializer=initialize_worker) if args.workers>1 else nullcontext(None)
    with context as pool:
        for iteration in range(total):
            started=perf_counter()
            if iteration==args.search_updates:
                # Measure the last search update before selecting the precision
                # starting point. This costs only 38 extra mean simulations.
                boundary_config,_,_=iteration_settings(config,iteration-1,args.search_updates,args.precision_updates)
                policy.config=boundary_config
                boundary_mean=fixed_batch(policy,boundary_config,args.span_ghz,mean=True,topology_masks=bank)
                empty={**boundary_mean,'kappa':boundary_mean['kappa'][:,:0],'phi':boundary_mean['phi'][:,:0]}
                boundary=evaluate_batch(empty,boundary_mean,boundary_config,pool,max_cases_per_job=args.cases_per_job)
                boundary_row,boundary_designs=mean_records(boundary_mean,boundary,pair_counts)
                recipe['search_end_measurement']=boundary_row
                write_json(directory/'search_end_designs.json',dict(completed_search_updates=args.search_updates,designs=boundary_designs))
                if boundary_row['mean_reward']>best:
                    best=boundary_row['mean_reward']
                    save(directory/'best.pt',policy,optimizer,boundary_config,args.search_updates,best,
                         sampling_recipe=recipe,exploration_state=exploration_state(policy))
                    write_json(directory/'best_designs.json',dict(iteration=args.search_updates,designs=boundary_designs))
                if boundary_row['worst_error_deg']<best_worst:
                    best_worst=boundary_row['worst_error_deg']
                    save(directory/'best_worst.pt',policy,optimizer,boundary_config,args.search_updates,boundary_row['mean_reward'],
                         sampling_recipe={**recipe,'checkpoint_selection':'lowest worst-topology phase RMS'},
                         exploration_state=exploration_state(policy))
                    write_json(directory/'best_worst_designs.json',dict(iteration=args.search_updates,designs=boundary_designs))
                start=directory/'best_search.pt';copyfile(directory/'best.pt',start)
                saved=torch.load(start,map_location='cpu',weights_only=False)
                policy.load_state_dict(saved['policy_state_dict'])
                optimizer=torch.optim.Adam(policy.parameters(),lr=1e-5);reward_scale={}
                recipe['precision_initialization']=dict(source_iteration=saved['iteration'],
                    source_reward=saved['held_out_reward'],sha256=hashlib.sha256(start.read_bytes()).hexdigest())
                write_json(directory/'run.json',recipe)
            current,rate,stage=iteration_settings(config,iteration,args.search_updates,args.precision_updates)
            policy.config=current;policy.minimum_log_std=current.minimum_log_std;policy.maximum_log_std=current.maximum_log_std
            for group in optimizer.param_groups:group['lr']=rate
            policy.train()
            item=fixed_batch(policy,current,args.span_ghz,topology_masks=bank)
            mean=fixed_batch(policy,current,args.span_ghz,mean=True,topology_masks=bank)
            measured=evaluate_batch(item,mean,current,pool,max_cases_per_job=args.cases_per_job)
            row,designs=mean_records(mean,measured,pair_counts)
            reward=measured['phase_rms_reward'][:,:38*args.actions_per_topology]
            row.update(iteration=iteration+1,stage=stage,sample_reward=float(reward.mean()),learning_rate=float(rate),
                exploration_scale=current.exploration_std_scale,optimizer_update=stage!='final')
            state=exploration_state(policy)
            if iteration==0:
                save(directory/'initial.pt',policy,optimizer,current,0,row['mean_reward'],sampling_recipe=recipe,exploration_state=state)
            save(directory/'current.pt',policy,optimizer,current,iteration+1,row['mean_reward'],sampling_recipe=recipe,exploration_state=state)
            write_json(directory/'current_designs.json',dict(iteration=iteration+1,designs=designs))
            if row['mean_reward']>best:
                best=row['mean_reward']
                save(directory/'best.pt',policy,optimizer,current,iteration+1,best,sampling_recipe=recipe,exploration_state=state)
                write_json(directory/'best_designs.json',dict(iteration=iteration+1,designs=designs))
            if row['worst_error_deg']<best_worst:
                best_worst=row['worst_error_deg']
                save(directory/'best_worst.pt',policy,optimizer,current,iteration+1,row['mean_reward'],
                     sampling_recipe={**recipe,'checkpoint_selection':'lowest worst-topology phase RMS'},exploration_state=state)
                write_json(directory/'best_worst_designs.json',dict(iteration=iteration+1,designs=designs))
            advantages=pair_excluded_advantages([reward],current,reward_scale)
            diagnostics=pcgrad_update(policy,optimizer,current,rng,[item],advantages)
            row.update(gradient_norm=diagnostics['gradient_norm'],pcgrad_tasks=diagnostics['pcgrad_task_count'],
                       iteration_seconds=perf_counter()-started)
            history.append(row);policy.eval()
            write_json(directory/'current_history.json',dict(recipe=recipe,history=history))
            if iteration==0 or (iteration+1)%args.plot_every==0 or iteration+1==total:
                progress_plot(history,directory,args.span_ghz,args.search_updates)
            print(f"{iteration+1}/{total} {stage} RMS={row['mean_reward']:.5f} worst={row['worst_error_deg']:.3f}deg "
                  f"3deg+locked={row['locked_success_count']}/38 locked={row['locked_count']}/38 "
                  f"time={row['iteration_seconds']:.1f}s",flush=True)
        save(directory/'final.pt',policy,optimizer,current,total,history[-1]['mean_reward'],sampling_recipe=recipe,exploration_state=state)
    print('Completed. Final iteration used exact means; optimizer update skipped.',flush=True)
    return CouplingModel(policy,current,metadata={'iteration':total},output_directory=directory),history


if __name__=='__main__':main()
