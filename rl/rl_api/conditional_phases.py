"""Sequential density policy: select strengths first, condition phases on them."""
import torch

from .density_control import DensityConditionedPolicy, decode_density_actions
from .exploration import sample_annealed_candidates
from .phase_reference import decode_samples


def sample_conditional_density(policy, features, config, fractions, *, mirrored=False,
                               continuous_mean=False):
    batch = len(features)
    candidates = fractions.shape[1]
    size = features.shape[1]
    edges = size*(size-1)
    dense = features.new_ones((batch,candidates,edges))
    actions, budget, score, base_mean, phase_std = sample_annealed_candidates(
        DensityConditionedPolicy(policy, fractions), features, dense,
        mirrored=mirrored, continuous_mean=continuous_mean, return_phase_parameters=True)
    masks, physical = decode_density_actions(actions, budget, fractions, config)
    repeated = features[:,None].expand(-1,candidates,-1,-1).reshape(batch*candidates,size,4)
    residual = policy.phase_residual_for_coupling(
        repeated, physical[0].reshape(batch*candidates,size,size)).reshape(batch,candidates,edges)
    # Each mirrored strength draw has its OWN conditional phase mean. Preserve
    # the original +/- phase noises around those means; do not share residuals.
    phases = actions[...,edges:] + residual.detach()
    if not continuous_mean and config.exploration_std_scale != 0.:
        base_phases, sampled_phases, correction = actions[...,edges:], phases, residual
        if (getattr(config, 'reciprocal_pair_actions', False)
                and not getattr(config,'asymmetric_coupling_phases',False)):
            from .reciprocal_pairs import pair_layout
            canonical = pair_layout(size,features.device)[0]
            base_phases, sampled_phases, correction = (
                value[...,canonical] for value in (base_phases,sampled_phases,correction))
        base_log = torch.distributions.Normal(base_mean,phase_std).log_prob(base_phases)
        conditional_log = torch.distributions.Normal(base_mean+correction,phase_std).log_prob(sampled_phases)
        score = score + (conditional_log-base_log).sum(-1)
    actions = torch.cat((actions[...,:edges],phases),dim=-1)
    # Keep the exact mask/strength draw used as input to the phase decoder.
    decoded = decode_samples(actions, masks, config, budget_actions=budget)
    return actions, budget, score, masks, decoded
