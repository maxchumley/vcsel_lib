"""Mean-design teaching after RL, updating only the existing phase-output row.

No additional network head or optimizer state is introduced. Both verification
rollouts use the same post-RL decoded mean strengths and budget. The dedicated
gradient step bypasses PCGrad and Adam momentum, and is kept only if it reduces
the fixed teacher loss. Ordinary RL remains responsible for all network weights.
"""
import numpy as np
import torch

from ._compat import engine
from .phase_reference import decode_samples
from .root_context import condition_on_roots
from .phase_correction import (collect_measurements, prepare_teachers, verify_teachers,
                               teacher_phase_inputs)
from .simulation import simulate_indexed_locking_chunk


HISTORY_KEYS = ('phase_correction_mean_simulations', 'phase_correction_loss_before',
                'phase_correction_update_applied', 'phase_correction_strength_change_max')


def mean_designs(policy, prepared):
    """Decode mean actions; never average stochastic physical strength matrices."""
    means = []
    with torch.no_grad():
        for item in prepared:
            masks = item['gates'][:, ::item['config'].actions_per_topology]
            roots = (None if item['roots'] is None else
                     item['roots'][:, ::item['config'].actions_per_topology])
            action, budget, _ = engine.sample_topology_conditioned_candidates(
                condition_on_roots(policy, roots), item['features'], masks, continuous_mean=True)
            kappa, phi, counts, _, _ = decode_samples(action, masks, item['config'],
                budget_actions=budget, detunings_ghz=item['detunings'], root_indices=roots)
            means.append(dict(config=item['config'], n_lasers=item['n_lasers'],
                targets=item['targets'], detunings=item['detunings'], features=item['features'],
                gates=masks, roots=roots, kappa=kappa, phi=phi, counts=counts, results=[]))
    return means


def mean_simulation_jobs(means):
    jobs = []
    for group, item in enumerate(means):
        for column in range(item['counts'].shape[1]):
            for start in range(0, len(item['targets']), item['config'].targets_per_worker_task):
                stop = min(start+item['config'].targets_per_worker_task, len(item['targets']))
                work = (len(jobs), start, stop, item['targets'][start:stop],
                    item['kappa'][start:stop, column:column+1],
                    item['phi'][start:stop, column:column+1],
                    item['detunings'][start:stop], item['config'])
                jobs.append((group, column, column+1, work))
    jobs.sort(key=lambda job: means[job[0]]['n_lasers'], reverse=True)
    return jobs


def phase_output_step(policy, means, *, learning_rate, weight, gradient_clip):
    """Freeze all shared features and all output rows except phase mean (row 1)."""
    examples = []
    # Cache frozen features. Only the phase-output row enters the gradient graph.
    with torch.no_grad():
        for item in means:
            captured = []
            hook = policy.edge_output.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach()))
            try:
                teachers, phases, target, masks = teacher_phase_inputs(policy, item)
            finally:
                hook.remove()
            if teachers:
                examples.append((captured[0].reshape(len(teachers), masks.shape[1], -1),
                                 phases.detach(), target, masks))
    if not examples:
        return dict(phase_correction_loss_before=0., phase_correction_loss=0.,
                    phase_correction_update_applied=0)
    phase_weight = policy.edge_output.weight[1]
    phase_bias = policy.edge_output.bias[1]
    old_weight, old_bias = phase_weight.detach().clone(), phase_bias.detach().clone()

    def loss():
        values = []
        for latent, phases, target, masks in examples:
            # Exactly the old mean when weights are unchanged; analytic target/
            # delay offsets are already included in the cached phase means.
            prediction = phases + latent @ (phase_weight-old_weight) + (phase_bias-old_bias)
            values.append(((1-torch.cos(prediction-target))*masks).sum(1)/masks.sum(1))
        return torch.cat(values).mean()

    before = loss()
    gw, gb = torch.autograd.grad(weight*before, (phase_weight, phase_bias))
    norm = torch.sqrt(gw.square().sum()+gb.square())
    scale = min(1., gradient_clip/max(float(norm), 1e-12))
    with torch.no_grad():
        phase_weight.add_(gw, alpha=-learning_rate*scale)
        phase_bias.add_(gb, alpha=-learning_rate*scale)
        after = loss()
        accepted = bool(torch.isfinite(after) and after < before.detach())
        if not accepted:
            phase_weight.copy_(old_weight)
            phase_bias.copy_(old_bias)
            after = before.detach()
    return dict(phase_correction_loss_before=float(before.detach()), phase_correction_loss=float(after),
                phase_correction_update_applied=int(accepted))


def run_mean_phase_correction(policy, prepared, settings, *, learning_rate, simulation_pool=None):
    """Classify post-RL means, verify phase targets, then take a frozen-feature step."""
    if getattr(policy.config, 'exploration_std_scale', 1.) == 0.:
        return dict(phase_correction_locked_fraction=float('nan'), phase_correction_attempted=0,
            phase_correction_accepted=0, phase_correction_mean_reward_gain=0.,
            phase_correction_mean_simulations=0, phase_correction_loss_before=0.,
            phase_correction_loss=0., phase_correction_update_applied=0,
            phase_correction_strength_change_max=0.)

    def evaluate(jobs):
        return (list(simulation_pool.imap_unordered(simulate_indexed_locking_chunk, jobs))
            if simulation_pool is not None and jobs else [simulate_indexed_locking_chunk(job) for job in jobs])

    means = mean_designs(policy, prepared)
    for result in evaluate(mean_simulation_jobs(means)):
        means[int(result['size_batch_index'])]['results'].append(result)
    for item in means:
        collect_measurements(item)
    work, metrics = prepare_teachers(means, settings, actions_per_topology=1)
    verify_teachers(means, evaluate(work), settings, metrics, require_phase_success=True)
    metrics['phase_correction_mean_simulations'] = sum(item['counts'].size for item in means)
    metrics.update(phase_output_step(policy, means, learning_rate=learning_rate,
        weight=settings.weight, gradient_clip=policy.config.gradient_clip))
    # Verify the actual decoded strengths, including the predicted budget,
    # after the restricted update, on exactly the same forward batches.
    after = mean_designs(policy, prepared)
    change = max(float(np.abs(a['kappa']-b['kappa']).max()) for a,b in zip(means, after))
    if change != 0.:
        raise RuntimeError('Phase-output update changed mean coupling strengths')
    metrics['phase_correction_strength_change_max'] = change
    return metrics
