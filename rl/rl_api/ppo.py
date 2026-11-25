"""Critic-free PPO on one fresh, simulated density batch; no historical replay."""
from copy import deepcopy
import numpy as np
import torch

from .density_control import DensityConditionedPolicy
from .mixed_pair_phases import mixed_pair_distributions

HISTORY_KEYS = ('ppo_updates', 'ppo_kl', 'ppo_clip_fraction', 'ppo_early_stop')


def action_distribution(policy, item):
    """Joint latent Gaussian, conditional on the SAME sampled physical strengths.

    Strength selection/redistribution is deterministic given the saved strength
    and budget actions. It must not be redone from the new policy's mean.
    All dense latent coordinates are scored, including subsequently pruned ones.
    """
    features = item['features']
    batch, candidates = item['actions'].shape[:2]
    size = item['n_lasers']
    edges, pairs = size*(size-1), size*(size-1)//2
    dense = features.new_ones((batch, candidates, edges))
    normal, budget, columns = mixed_pair_distributions(
        DensityConditionedPolicy(policy, item['density_fractions']), features, dense)
    repeated = features[:, None].expand(-1, candidates, -1, -1).reshape(-1, size, 4)
    residual = policy.phase_residual_for_coupling(
        repeated, item['kappa'].reshape(-1, size, size)).reshape(batch, candidates, edges)
    mean = torch.cat((normal.loc[..., :pairs], normal.loc[..., pairs:]+residual), -1)
    # Independent canonical strengths, directed conditional phases and budget.
    return torch.distributions.Normal(torch.cat((mean, budget.loc[..., None]), -1),
        torch.cat((normal.scale, budget.scale[..., None]), -1)), columns


def action_log_prob(distribution, columns, item):
    values = torch.cat((item['actions'][..., columns], item['budget_actions'][..., None]), -1)
    return distribution.log_prob(values).sum(-1)


def clipped_loss(log_prob, old_log_prob, advantages, clip_ratio):
    ratio = (log_prob-old_log_prob).exp()
    return -torch.minimum(ratio*advantages, ratio.clamp(1.-clip_ratio, 1.+clip_ratio)*advantages)


def ppo_update(policy, optimizer, config, rng, prepared, advantages, settings):
    """Up to four PCGrad updates, reusing measured rewards and frozen baselines.

    KL is summed over the complete design and averaged separately within each M.
    Conditional phase KL uses the original sampled strengths (a Monte Carlo
    estimate of the conditional expectation). Stop if ANY size exceeds its limit.
    """
    from .training_batch import pcgrad_update
    if config.exploration_std_scale == 0.:
        diagnostics = pcgrad_update(policy, optimizer, config, rng, prepared, advantages)
        return dict(diagnostics, ppo_updates=0, ppo_kl=0., ppo_clip_fraction=0., ppo_early_stop=False)
    old, old_scores = [], []
    with torch.no_grad():
        for item in prepared:
            distribution, columns = action_distribution(policy, item)
            old.append(distribution)
            score = action_log_prob(distribution, columns, item)
            # Verify the fresh-batch sampler and rescoring use the same law.
            torch.testing.assert_close(score, item['log_prob'], rtol=2e-5, atol=2e-4)
            old_scores.append(score)
    advantages = [torch.as_tensor(a, dtype=item['features'].dtype,
        device=item['features'].device).detach() for a, item in zip(advantages, prepared)]
    updates, stopped, max_kl, fraction = 0, False, 0., 0.
    diagnostics = {}
    update_rng = rng
    for _ in range(settings.updates):
        candidate_losses = []
        for item, reference, score, advantage in zip(prepared, old, old_scores, advantages):
            distribution, columns = action_distribution(policy, item)
            current = action_log_prob(distribution, columns, item)
            loss = clipped_loss(current, score, advantage, settings.clip_ratio)
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Nonfinite PPO objective; no update applied')
            candidate_losses.append(loss)
        diagnostics = pcgrad_update(policy, optimizer, config, update_rng, prepared, advantages,
                                    candidate_losses=candidate_losses)
        updates += 1
        if updates == 1:
            # Extra optimizer passes must not shift future context/density draws.
            update_rng = deepcopy(rng)
        # Check AFTER each update, including the final one; no extra simulations.
        kls, clipped = [], []
        with torch.no_grad():
            for item, reference, score in zip(prepared, old, old_scores):
                distribution, columns = action_distribution(policy, item)
                kl = torch.distributions.kl_divergence(reference, distribution).sum(-1).mean()
                kls.append(float(kl))
                ratio = (action_log_prob(distribution, columns, item)-score).exp()
                clipped.append(float(((ratio-1.).abs() > settings.clip_ratio).float().mean()))
        max_kl, fraction = max(kls), float(np.mean(clipped))
        if not np.isfinite(max_kl) or max_kl > settings.target_kl:
            stopped = True
            break
    return dict(diagnostics, ppo_updates=updates, ppo_kl=max_kl,
                ppo_clip_fraction=fraction, ppo_early_stop=stopped)
