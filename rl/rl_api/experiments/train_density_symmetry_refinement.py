"""Warm-start the working density policy and progressively enforce reciprocity."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from .. import CouplingModel
from .._compat import PACKAGE_DIR, core, exported_settings, make_backend_config
from ..checkpoints import save, write_json
from ..exploration import exploration_state
from ..simulation import initialize_worker
from ..training import make_validation_bank, evaluate_bank
from ..training_batch import run_training_batch

RUN_NAME = 'density_symmetry_refinement_from_working_laser1'
PROGRESS_FILENAME = 'density_symmetry_refinement_progress.png'
SOURCE_SHA256 = '83ad841f91a2022cd869b8d9cb21a82a57ce11a60f99f85e3ef5b7766008e43b'
DEFAULT_CHECKPOINT = PACKAGE_DIR/'runs/initialization'/f'density_asym_laser1_{SOURCE_SHA256[:12]}.pt'


def quality(validation):
    return dict(reward=float(np.mean([v['training_reward'] for v in validation.values()])),
                success=float(np.mean([v['phase_rms_success_fraction'] for v in validation.values()])))


def retained(measured, baseline, reward_drop, success_drop):
    """Use a fixed original baseline, preventing cumulative accuracy erosion."""
    return (np.isfinite(measured['reward']) and np.isfinite(measured['success'])
        and measured['reward'] >= baseline['reward']-reward_drop
        and measured['success'] >= baseline['success']-success_drop)


def level_config(config, fraction):
    """Keep adaptation possible when a level stalls; taper by symmetry, not time."""
    return replace(config, coupling_symmetry_fraction=fraction,
        exploration_std_scale=.03*.1**fraction)


def load_start(path, directory, args):
    path=Path(path).expanduser().resolve()
    model=CouplingModel.load(path,device='cpu')
    c=model.backend_config
    if (not getattr(c,'density_control',False) or getattr(c,'symmetric_coupling_values',False)
            or getattr(c,'reciprocal_pair_actions',False)
            or c.phase_parameterization!='target_delay_residual'
            or hasattr(c,'coupling_symmetry_fraction')):
        raise ValueError('Start from the working asymmetric-value laser-1 density checkpoint')
    settings=exported_settings(c)
    settings['coupling_symmetry_fraction']=0.
    config=make_backend_config(settings)
    config=replace(config,training_iterations=args.updates,n_jobs=args.workers,
        held_out_target_count=args.validation_pairs or config.held_out_target_count,
        targets_per_worker_task=5,candidates_per_worker_task=8,
        validation_interval=args.every,plot_update_interval=args.plot_every,
        enable_exploration_annealing=False,enable_minimum_log_std_annealing=False,
        minimum_log_std=model.policy.minimum_log_std,maximum_log_std=model.policy.maximum_log_std,
        phase_exploration_fraction=0.,phase_exploration_start_std=0.,
        learning_rate=1e-5,learning_rate_final=1e-6,jupyter_mode=False,
        best_checkpoint_file=str(directory/'best_accepted.pt'),current_checkpoint_file=str(directory/'current.pt'),
        final_checkpoint_file=str(directory/'final.pt'))
    if args.smoke_test:
        config=replace(config,training_n_lasers=(3,4),targets_per_batch=2,
            targets_per_worker_task=1,held_out_target_count=5)
    model.policy.config=config
    return model.policy,config,dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_iteration=model.metadata['iteration'],source_reward=model.metadata['held_out_reward'],
        copied='all model weights and physical budget calibration',reset='Adam, sampler, reward scale and refinement history')


def plot_progress(history,directory):
    from ..plotting.training import save_training_progress
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
                         'axes.formatter.use_mathtext':True}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        x=[r['iteration'] for r in history]
        axes[0].plot(x,[r['sample_reward'] for r in history],alpha=.5,label='sampled actions')
        axes[0].plot(x,[r['reward'] for r in history],label='validation means')
        axes[0].set(xlabel='Refinement update',ylabel='RMS reward',ylim=(0,1.02))
        axes[1].plot(x,[r['success'] for r in history],label=r'validation within $3^\circ$')
        axes[1].plot(x,[r['fraction'] for r in history],label='enforced symmetry')
        axes[1].set(xlabel='Refinement update',ylabel='Fraction',ylim=(0,1.02))
        for ax in axes:ax.grid(alpha=.2);ax.legend()
        save_training_progress(fig,directory,200,filename=PROGRESS_FILENAME)
        plt.close(fig)


def options(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--output-directory',type=Path)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--workers',type=int,default=14)
    p.add_argument('--every',type=int,default=50,help='Validate and consider tightening every N updates')
    p.add_argument('--increment',type=float,default=.05)
    p.add_argument('--reward-drop',type=float,default=.01)
    p.add_argument('--success-drop',type=float,default=.05,help='Allowed absolute drop in success fraction')
    p.add_argument('--validation-pairs',type=int)
    p.add_argument('--plot-every',type=int,default=10)
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--preview',action='store_true')
    args=p.parse_args(argv)
    if (args.updates<1 or args.workers<1 or args.every<1 or args.plot_every<1
            or not 0.<args.increment<=1. or not 0.<=args.reward_drop<=1.
            or not 0.<=args.success_drop<=1. or (args.validation_pairs is not None and args.validation_pairs<1)):
        p.error('Use positive counts, increment in (0,1], and tolerances in [0,1]')
    allocated=os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and args.workers+int(args.workers>1)>int(allocated):p.error('Workers plus trainer exceed allocated CPUs')
    if args.smoke_test:args.updates,args.every,args.increment=2,1,.5
    return args


def main(argv=None):
    args=options(argv);torch.set_num_threads(1)
    directory=args.output_directory or PACKAGE_DIR/'runs/great_lakes'/RUN_NAME/os.environ.get('SLURM_JOB_ID','local')
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f'Working-model checkpoint missing: {args.initial_checkpoint}')
    # No directories or training side effects in preview mode.
    policy,config,initialization=load_start(args.initial_checkpoint,directory,args)
    recipe=dict(mode='density_symmetry_refinement',initialization=initialization,
        updates=args.updates,symmetry_increment=args.increment,validation_every=args.every,
        maximum_reward_drop=args.reward_drop,maximum_success_drop=args.success_drop,plot_every=args.plot_every,
        starting_fraction=0.,goal_fraction=1.,phase_reference='laser 1, unchanged',
        parameterization='blend latent strengths toward reciprocal averages; move phases together along shortest circular arcs',
        objective='unchanged wrapped RMS with mirrored REINFORCE, pair-excluded baseline and PCGrad',
        initial_exploration_scale=.03,final_exploration_scale=.003,wide_phase_actions=False,
        advancement='train at current fraction; tighten only when validation retains original accuracy',
        checkpoint_selection='highest fraction passing original validation floors; reward breaks ties',
        progress_filename=PROGRESS_FILENAME,weights_added=0,teacher_labels=False,
        training_simulations_per_update=config.targets_per_batch*config.candidates_per_target,
        validation_cases=len(config.training_n_lasers)*config.held_out_target_count)
    print(recipe,flush=True)
    if args.preview:return recipe
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Use a new directory: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    rng=core.set_random_seed(config.random_seed);optimizer=torch.optim.Adam(policy.parameters(),lr=1e-5)
    bank=make_validation_bank(config);history=[];reward_scale={}
    context=mp.get_context('spawn').Pool(args.workers,initializer=initialize_worker) if args.workers>1 else nullcontext(None)
    with context as pool:
        policy.config=level_config(config,0.)
        baseline=quality(evaluate_bank(policy,policy.config,bank,0,pool))
        recipe.update(baseline=baseline,backend_config=exported_settings(config))
        write_json(directory/'run.json',recipe)
        save(directory/'initial.pt',policy,optimizer,policy.config,0,baseline['reward'],
             sampling_recipe=recipe,exploration_state=exploration_state(policy))
        save(directory/'best_accepted.pt',policy,optimizer,policy.config,0,baseline['reward'],
             sampling_recipe=recipe,exploration_state=exploration_state(policy))
        best=(0.,baseline['reward'])
        write_json(directory/'baseline.json',baseline)
        fraction=min(args.increment,1.);validation=baseline.copy();validation_fraction=0.
        for iteration in range(1,args.updates+1):
            current=level_config(config,fraction);policy.config=current
            policy.minimum_log_std=current.minimum_log_std;policy.maximum_log_std=current.maximum_log_std
            rate=1e-5*.1**fraction
            for group in optimizer.param_groups:group['lr']=rate
            metrics=run_training_batch(policy,optimizer,current,rng,iteration-1,
                                      simulation_pool=pool,reward_scale_state=reward_scale)
            validate=iteration%args.every==0 or iteration==args.updates
            accepted=False
            if validate:
                validation=quality(evaluate_bank(policy,current,bank,iteration-1,pool));validation_fraction=fraction
                accepted=retained(validation,baseline,args.reward_drop,args.success_drop)
                if accepted and (fraction,validation['reward'])>best:
                    best=(fraction,validation['reward'])
                    save(directory/'best_accepted.pt',policy,optimizer,current,iteration,validation['reward'],
                        sampling_recipe=recipe,exploration_state=exploration_state(policy))
            row=dict(iteration=iteration,fraction=fraction,validation_fraction=validation_fraction,
                sample_reward=metrics['training_reward'],**validation,validated=validate,
                accuracy_retained=accepted if validate else None,learning_rate=rate,
                exploration_scale=current.exploration_std_scale)
            history.append(row)
            save(directory/'current.pt',policy,optimizer,current,iteration,validation['reward'],
                sampling_recipe=recipe,exploration_state=exploration_state(policy))
            write_json(directory/'current_history.json',dict(recipe=recipe,history=history))
            if iteration==1 or iteration%args.plot_every==0 or iteration==args.updates:plot_progress(history,directory)
            print(f"{iteration}/{args.updates} symmetry={fraction:.2f} RMS={metrics['training_reward']:.4f} "
                  f"validation={validation['reward']:.4f} 3deg={100*validation['success']:.1f}%"+
                  ((' retained' if accepted else ' HOLD') if validate else ''),flush=True)
            # No untrained last-minute symmetry increase at the update limit.
            if validate and accepted and iteration<args.updates:fraction=min(round(fraction+args.increment,12),1.)
        final=replace(current,exploration_std_scale=0.);policy.config=final
        final_quality=quality(evaluate_bank(policy,final,bank,args.updates,pool))
        save(directory/'final.pt',policy,optimizer,final,args.updates,final_quality['reward'],
             sampling_recipe=recipe,exploration_state=exploration_state(policy))
        fully=replace(final,coupling_symmetry_fraction=1.);policy.config=fully
        full_quality=quality(evaluate_bank(policy,fully,bank,args.updates,pool)) if fraction!=1. else final_quality
        save(directory/'fully_symmetric.pt',policy,optimizer,fully,args.updates,full_quality['reward'],
             sampling_recipe={**recipe,'forced_endpoint_evaluation':fraction!=1.},exploration_state=exploration_state(policy))
        result=dict(reached_fraction=fraction,best_accepted_fraction=best[0],final_validation=final_quality,
            fully_symmetric_validation=full_quality,
            fully_symmetric_accuracy_retained=retained(full_quality,baseline,args.reward_drop,args.success_drop))
        write_json(directory/'result.json',result);print(result,flush=True)
    return result


if __name__=='__main__':main()
