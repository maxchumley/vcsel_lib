"""The active phase-only training batch, with an explicit mask sampler.

No module globals are replaced. The running legacy experiment is untouched.
Policy, simulation workers, rewards, advantage scaling and PCGrad use the
existing numerical routines. This path omits inactive historical features.
"""
from dataclasses import replace
import numpy as np
import torch

from ._compat import core, engine
from .sampling import sample_shared_rooted_topologies
from .exploration import refinement_config, sample_annealed_candidates
from .advantages import pair_excluded_advantages
from .phase_reference import decode_samples, uses_root_reference
from .root_context import condition_on_roots


def prepare_size(policy, config, rng, size, pair_count, iteration, mask_sampler):
    """Sample contexts/masks, then predict the configured noisy actions."""
    local = replace(config, n_lasers=size)
    targets = core.sample_target_phases(pair_count, rng, size,
        config.structured_target_fraction, config.structured_target_jitter_rad)
    detunings = core.sample_detuning_distributions(pair_count, rng, local, iteration=iteration)
    targets, detunings = core.sort_by_detuning(targets, detunings)
    features = engine.encode_context(targets, detunings, policy)
    root_options = dict(return_roots=True) if uses_root_reference(config) else {}
    masks = mask_sampler(pair_count, config.pcgrad_topology_groups, size, rng, features.device,
        min_fraction=engine.topology_min_fraction_at(iteration, local),
        density_bias=config.training_topology_density_bias, fixed_star=config.topology_star_backbone,
        **root_options)
    masks, roots = masks if root_options else (masks, None)
    gates = masks.repeat_interleave(config.actions_per_topology, dim=1)
    action_roots = None if roots is None else np.repeat(roots, config.actions_per_topology, axis=1)
    conditioned = condition_on_roots(policy, action_roots)
    if hasattr(config, 'exploration_std_scale'):
        actions, budget_actions, log_prob = sample_annealed_candidates(conditioned, features, gates,
            mirrored=getattr(config, 'action_sampling', 'independent') == 'mirrored')
    else:
        actions, budget_actions, log_prob = engine.sample_topology_conditioned_candidates(
            conditioned, features, gates, iteration=iteration, rng=rng)
    reach = engine.root_reach_fractions(gates.detach().cpu().numpy(), size)
    if not np.all(reach == 1.):
        raise RuntimeError("Training masks must be rooted")
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, gates, local, budget_actions=budget_actions,
        detunings_ghz=detunings, root_indices=action_roots)
    return dict(n_lasers=size, config=local, targets=targets, detunings=detunings,
        features=features, gates=gates, log_prob=log_prob, kappa=kappa, phi=phi,
        counts=counts, rho=rho, shortfall=shortfall, reach=reach, roots=action_roots, results=[])


def simulation_jobs(prepared, config):
    """Vectorize pairs x actions per job using the explicit run-file chunk sizes."""
    work = []
    chunk_size = min(config.candidates_per_worker_task, config.candidates_per_target)
    for group, item in enumerate(prepared):
        pair_jobs = max(1, (len(item['targets'])+config.targets_per_worker_task-1)
                       // config.targets_per_worker_task)
        for start in range(0, config.candidates_per_target, chunk_size):
            stop = min(start+chunk_size, config.candidates_per_target)
            chunks = core.split_simulation_batch(item['targets'], item['kappa'][:,start:stop],
                item['phi'][:,start:stop], item['config'],
                detuning_distributions_ghz=item['detunings'], n_jobs=pair_jobs)
            work.extend((group,start,stop,chunk) for chunk in chunks)
    work.sort(key=lambda job: prepared[job[0]]['n_lasers'], reverse=True)
    return work


def collect_rewards(item, config):
    """Restore candidate order and verify every physical case returned once."""
    shape = (len(item['targets']), config.candidates_per_target)
    phase = np.empty(shape)
    rms = {key:np.empty(shape) for key in core.PHASE_RMS_METRIC_KEYS}
    coverage = np.zeros(shape, dtype=int)
    for result in item['results']:
        rows = slice(int(result['start_index']), int(result['stop_index']))
        columns = slice(int(result['candidate_start_index']), int(result['candidate_stop_index']))
        phase[rows,columns] = result['phase_reward']
        for key in rms:
            rms[key][rows,columns] = result[key]
        coverage[rows,columns] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Simulation did not return every candidate exactly once")
    budget = engine.relative_coupling_budget(item['kappa'],item['config'])
    metrics = dict(phase_reward=float(phase.mean()), training_reward=float(rms['phase_rms_reward'].mean()),
        mean_phase_rms_error_deg=float(rms['mean_phase_rms_error_deg'].mean()),
        worst_phase_rms_error_deg=float(rms['worst_phase_rms_error_deg'].mean()),
        phase_rms_success_fraction=float(rms['phase_rms_success'].mean()),
        retained_link_fraction=float(item['rho'].mean()), normalized_coupling_budget=float(budget.mean()),
        mean_redistribution_shortfall_per_ns=float(item['shortfall'].mean()),
        mean_active_links=float(item['counts'].mean()), rooted_fraction=float(np.mean(item['reach']==1.)))
    return rms['phase_rms_reward'], metrics


def pcgrad_update(policy, optimizer, config, rng, prepared, advantages):
    """Group gradients by physical band or mask rank; combine with PCGrad."""
    losses, weights, labels, context_counts = [], [], [], []
    for item, advantage_np in zip(prepared,advantages):
        advantage = torch.as_tensor(advantage_np,
            dtype=item['features'].dtype,device=item['features'].device)
        candidate_losses = -(item['log_prob']*advantage)
        # Match the reference's boolean selection and mean reduction order.
        if config.pcgrad_task_grouping == "array_size_topology_density":
            band_ids = engine.topology_density_band_ids(item['counts'], item['n_lasers'])
            band_names = engine.DENSITY_BANDS
        else:
            band_ids = np.broadcast_to(np.repeat(np.arange(config.pcgrad_topology_groups),
                config.actions_per_topology)[None,:], item['counts'].shape)
            band_names = tuple(f"density_{i+1}of{config.pcgrad_topology_groups}"
                               for i in range(config.pcgrad_topology_groups))
        for band, name in enumerate(band_names):
            selected = band_ids == band
            if not np.any(selected):
                continue
            selected_tensor = torch.as_tensor(selected,device=candidate_losses.device)
            losses.append(candidate_losses[selected_tensor].mean()+candidate_losses.new_zeros(()))
            weights.append(float(selected.mean())/len(prepared))
            labels.append(f"M{item['n_lasers']}:{name}")
            context_counts.append(int(np.count_nonzero(selected.any(axis=1))))
    optimizer.zero_grad(set_to_none=True)
    if getattr(config, 'exploration_std_scale', 1.) == 0.:
        # Adam's momentum can move weights even with zero gradients. This is
        # a deterministic final check, so leave weights AND optimizer untouched.
        return dict(gradient_norm=0., pcgrad_task_count=len(losses),
            pcgrad_task_labels=labels, pcgrad_task_context_counts=context_counts,
            negative_gradient_pair_fraction=float('nan'), mean_gradient_cosine=float('nan'),
            mean_action_only=True)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    gradients = []
    for index,loss in enumerate(losses):
        derivatives = torch.autograd.grad(loss,parameters,allow_unused=True,
                                         retain_graph=index<len(losses)-1)
        gradients.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None
            else g.detach().reshape(-1) for p,g in zip(parameters,derivatives)]))
    combined, diagnostics = core.combine_task_gradients_pcgrad(
        torch.stack(gradients),config.per_size_gradient_clip,rng,task_weights=weights)
    cursor = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = combined[cursor:cursor+count].reshape_as(parameter).clone()
        cursor += count
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),config.gradient_clip)
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(policy.minimum_log_std,policy.maximum_log_std)
        policy.log_std_phi.clamp_(policy.minimum_log_std,policy.maximum_log_std)
    return dict(gradient_norm=float(norm.detach().cpu()), pcgrad_task_count=len(losses),
        pcgrad_task_labels=labels,pcgrad_task_context_counts=context_counts,**diagnostics)


def run_training_batch(policy, optimizer, config, rng, iteration, *,
                       simulation_pool=None, reward_scale_state=None,
                       mask_sampler=sample_shared_rooted_topologies, schedules=None):
    """One supplied-topology batch; explicit samplers also support parity tests."""
    if (not config.topology_input_mode or config.objective_mode!='phase_only'
            or config.phase_reward_mode!='wrapped_rms'
            or config.gradient_combination_mode!='pcgrad'
            or config.pcgrad_task_grouping not in {'array_size_topology_strata', 'array_size_topology_density'}
            or config.topology_sampling_mode not in {'paired_contexts', 'per_target'}):
        raise ValueError("This batch implements supplied-topology RMS training with PCGrad")
    policy.train()
    config = refinement_config(config, schedules, iteration)
    policy.config = config
    policy.minimum_log_std = core.minimum_log_std_at(iteration,config)
    policy.maximum_log_std = core.maximum_log_std_at(iteration,config)
    pair_count = config.targets_per_batch//len(config.training_n_lasers)
    prepared = [prepare_size(policy,config,rng,size,pair_count,iteration,mask_sampler)
                for size in config.training_n_lasers]
    work = simulation_jobs(prepared,config)
    worker = core.simulate_indexed_target_chunk
    if hasattr(config, "phase_model"):
        from .simulation import simulate_indexed_chunk
        worker = simulate_indexed_chunk
    results = (list(simulation_pool.imap_unordered(worker,work)) if simulation_pool is not None
               else [worker(job) for job in work])
    for result in results:
        prepared[int(result['size_batch_index'])]['results'].append(result)
    rewards, size_metrics, budgets = [], {}, []
    for item in prepared:
        reward, metrics = collect_rewards(item,config)
        rewards.append(reward)
        size_metrics[item['n_lasers']] = metrics
        budgets.append((item['kappa'].sum(axis=(-2,-1))/item['n_lasers']).ravel())
    advantage_function = (pair_excluded_advantages
        if getattr(config, 'reward_baseline', 'topology_mean') == 'pair_excluded'
        else engine._topology_advantages)
    advantages = advantage_function(rewards,config,reward_scale_state)
    diagnostics = pcgrad_update(policy,optimizer,config,rng,prepared,advantages)
    average = {key:float(np.mean([metrics[key] for metrics in size_metrics.values()]))
               for key in next(iter(size_metrics.values()))}
    percentiles = np.percentile(np.concatenate(budgets),[10,50,90])
    return dict(**average, sampled_budget_p10_per_laser=float(percentiles[0]),
        sampled_budget_median_per_laser=float(percentiles[1]),
        sampled_budget_p90_per_laser=float(percentiles[2]),
        size_metrics=size_metrics,**diagnostics)
