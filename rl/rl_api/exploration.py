"""Optional final precision stage; all user settings remain in the run file."""
from dataclasses import replace
import math

import torch


def refinement_config(config, schedules, iteration):
    """Return this iteration's bounds and nominal budget SD without mutation.

    The ordinary SD schedule has finished before this stage starts. The
    existing numerical helpers then use these progressively narrower targets.
    Budget SD is nominal: the existing learned multiplier exp(.25*tanh(w))
    still applies, so a final .05 gives actual SD between .039 and .064.
    """
    if schedules is not None and schedules.anneal_to_mean:
        progress = (iteration + 1 - schedules.ordinary_start) / (schedules.ordinary_end - schedules.ordinary_start)
        scale = 1. - min(max(progress, 0.), 1.)
        return replace(config, exploration_std_scale=scale,
            minimum_log_std=config.initial_minimum_log_std,
            enable_minimum_log_std_annealing=False, enable_exploration_annealing=False)
    if schedules is None or schedules.precision_start is None:
        return config
    printed = iteration+1
    if printed <= schedules.precision_start:
        return config
    progress = min((printed-schedules.precision_start)/
                   (schedules.precision_end-schedules.precision_start), 1.)
    blend = .5*(1.-math.cos(math.pi*progress))
    return replace(config,
        minimum_log_std=(1-blend)*config.minimum_log_std+blend*schedules.precision_minimum_log_std,
        final_maximum_log_std=(1-blend)*config.final_maximum_log_std+blend*schedules.precision_maximum_log_std,
        budget_exploration_end_std=(1-blend)*config.budget_exploration_end_std+blend*schedules.precision_budget_std)


def exploration_state(policy):
    """Save actual clipping bounds so stochastic inspection reloads them exactly."""
    state = dict(minimum_log_std=float(policy.minimum_log_std),
                maximum_log_std=float(policy.maximum_log_std),
                nominal_budget_std=float(policy.config.budget_exploration_end_std))
    if hasattr(policy.config, 'exploration_std_scale'):
        scale = float(policy.config.exploration_std_scale)
        state.update(action_std_scale=scale,
            effective_nominal_budget_std=scale*state['nominal_budget_std'],
            extra_wide_phase_sd_deg=math.degrees(policy.config.phase_exploration_start_std)*scale)
    return state


def sample_annealed_candidates(policy, features, masks, *, mirrored=False):
    """Sample the opt-in full-run taper, including extra-wide phase actions.

    The policy scales ordinary edge/budget SDs. Extra-wide phase SD gets the
    same multiplier here. Gaussian log probabilities use those actual widths.
    At zero, return exact means with zero score and consume no sampling RNG.
    Mirrored training reflects every second full action/budget sample around
    its mean. The first two slots form the wide pair in the eight-action run.
    Reflection is detached: learning still uses Gaussian REINFORCE scores.
    """
    batch, candidates, edges = masks.shape
    repeated = features[:, None].expand(-1, candidates, -1, -1).reshape(
        batch*candidates, features.shape[1], features.shape[2])
    mean, log_std, _, pooled = policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(batch*candidates, edges), return_pooled=True)
    mean = mean.reshape(batch, candidates, 2*edges)
    budget_mean, budget_log_std = policy.budget_parameters(repeated, pooled_states=pooled)
    budget_mean = budget_mean.reshape(batch, candidates)
    scale = policy.config.exploration_std_scale
    if scale == 0.:
        return mean.clone(), budget_mean.clone(), features.new_zeros((batch, candidates))
    std = log_std.reshape_as(mean).exp()
    slots = torch.arange(candidates, device=features.device) % policy.config.actions_per_topology
    wide_count = round(policy.config.actions_per_topology*policy.config.phase_exploration_fraction)
    extra = (slots < wide_count).to(std.dtype)[None, :, None]
    extra = extra * policy.config.phase_exploration_start_std * scale
    std = torch.cat((std[..., :edges], torch.sqrt(std[..., edges:].square()+extra.square())), dim=-1)
    if mirrored:
        per_topology = policy.config.actions_per_topology
        if per_topology < 4 or per_topology % 2 or candidates % per_topology or wide_count % 2:
            raise ValueError("Mirrored sampling needs complete even action groups and wide pairs")
        if not torch.equal(masks[:, ::2], masks[:, 1::2]):
            raise ValueError("Mirrored partners must use the same supplied topology")
        # Identical inputs can differ by an ULP across batched BLAS rows.
        # Define one Gaussian per pair, with both scores using its parameters.
        mean = shared_pair_parameters(mean)
        std = shared_pair_parameters(std)
    normal = torch.distributions.Normal(mean, std)
    actions = normal.sample()
    if mirrored:
        actions = mirror_pairs(actions, mean)
    log_prob = (normal.log_prob(actions)*torch.cat((masks, masks), dim=-1)).sum(dim=-1)
    budget_std = budget_log_std.reshape(batch, candidates).exp()
    if mirrored:
        budget_mean = shared_pair_parameters(budget_mean)
        budget_std = shared_pair_parameters(budget_std)
    budget_normal = torch.distributions.Normal(budget_mean, budget_std)
    budget_actions = budget_normal.sample()
    if mirrored:
        budget_actions = mirror_pairs(budget_actions, budget_mean)
    return actions, budget_actions, log_prob+budget_normal.log_prob(budget_actions)


def shared_pair_parameters(values):
    """Use the positive member's prediction for both members of each pair.

    The model receives identical inputs for both slots, but multithreaded
    batched arithmetic need not return bit-identical floats. Sharing the
    prediction explicitly defines the intended common Gaussian and keeps
    both log-probability scores and their gradients consistent.
    """
    return values[:, ::2].repeat_interleave(2, dim=1)


def mirror_pairs(samples, mean):
    """Keep positive draws exactly; reflect the odd slots without a gradient path.

    Drawing all slots before reflecting preserves the existing Torch RNG
    consumption. New/old runs at seed 7 therefore also share the positive
    samples at equal weights, without changing the context/topology RNG.
    """
    with torch.no_grad():
        result = samples.clone()
        result[:, 1::2] = 2*mean[:, 1::2] - samples[:, ::2]
    return result
