"""Steady-phase feedback as a verified, detached teaching target.

Only training opts in. Classifying existing trajectories needs no new rollout.
At most one locked action per context/topology gets a fresh correction rollout;
its strengths, mask and detunings are unchanged. Validation and inference never
apply the correction, so held-out improvements must come from learned weights.
"""
import numpy as np
import torch

from ._compat import core
from .phase_reference import apply_root_reference
from .root_context import condition_on_roots


HISTORY_KEYS = ('phase_correction_locked_fraction', 'phase_correction_attempted',
                'phase_correction_accepted', 'phase_correction_loss',
                'phase_correction_mean_reward_gain')


def wrap(value):
    return np.angle(np.exp(1j*value))


def measure_locking(times, states, config):
    """Finite-tail frequency agreement and phase jitter about its own mean."""
    ramp_end = (config.coupling_ramp_start_delays
                + config.coupling_ramp_rise_delays/.8)*config.delay_seconds
    start = max(int((1-config.reward_tail_fraction)*states.shape[-1]),
                int(np.searchsorted(times, ramp_end)))
    tail = states[..., start:]
    if tail.shape[-1] < 2:
        raise ValueError('Locking measurements need at least two settled samples')
    t = times[start:]-times[start:].mean()
    with np.errstate(invalid='ignore', over='ignore'):
        relative = tail[:, 2::3]-tail[:, 2:3]
        achieved = np.angle(np.mean(np.exp(1j*relative), axis=-1))
        jitter = np.sqrt(np.mean(wrap(relative-achieved[..., None])**2, axis=-1))
        slopes = (relative*t).sum(axis=-1)/(t*t).sum()
    finite = np.isfinite(states).all(axis=(1, 2))
    return dict(locking_finite=finite,
        locking_achieved_phases_rad=achieved,
        locking_phase_jitter_deg=np.where(finite, np.rad2deg(jitter[:, 1:].max(axis=1)), np.inf),
        locking_frequency_spread_mhz=np.where(finite, np.ptp(slopes, axis=1)/(2*np.pi*1e6), np.inf))


def locked(measurements, settings):
    return (measurements['locking_finite']
        & (measurements['locking_frequency_spread_mhz'] < settings.frequency_tolerance_mhz)
        & (measurements['locking_phase_jitter_deg'] < settings.phase_jitter_tolerance_deg))


def collect_measurements(item):
    """Restore unordered worker chunks to the original context/action slots."""
    shape = item['counts'].shape
    measurements = {}
    for result in item['results']:
        rows = slice(int(result['start_index']), int(result['stop_index']))
        columns = slice(int(result['candidate_start_index']), int(result['candidate_stop_index']))
        for key, value in result.items():
            if key.startswith('locking_') or key in ('phase_rms_reward', 'rms_orientation'):
                if key not in measurements:
                    measurements[key] = np.empty(shape+value.shape[2:], dtype=value.dtype)
                measurements[key][rows, columns] = value
    item['locking'] = measurements


def prepare_teachers(prepared, settings, *, verify=True, actions_per_topology=None):
    """Select the best locked action in each supplied-mask block, not across masks."""
    work, total, eligible = [], 0, 0
    for group, item in enumerate(prepared):
        item['phase_teachers'] = []
        measurements = item['locking']
        eligible_mask = locked(measurements, settings)
        total += eligible_mask.size
        eligible += int(eligible_mask.sum())
        if not verify:
            continue
        per_topology = (item['config'].actions_per_topology
                        if actions_per_topology is None else actions_per_topology)
        masks = item['gates'].detach().cpu().numpy().astype(bool)
        receivers, sources = core.directed_link_indices(item['n_lasers'])
        for row in range(len(item['targets'])):
            for start in range(0, eligible_mask.shape[1], per_topology):
                candidates = np.flatnonzero(eligible_mask[row, start:start+per_topology])+start
                if not len(candidates):
                    continue
                column = int(candidates[np.argmax(measurements['phase_rms_reward'][row, candidates])])
                beta = (measurements['rms_orientation'][row, column]*item['targets'][row]
                        - measurements['locking_achieved_phases_rad'][row, column])
                mask = np.zeros((item['n_lasers'], item['n_lasers']), dtype=bool)
                mask[receivers, sources] = masks[row, column]
                phase = np.where(mask, wrap(item['phi'][row, column]
                                          + beta[None, :]-beta[:, None]), 0.)
                item['phase_teachers'].append(dict(row=row, column=column, phase=phase,
                    baseline_reward=float(measurements['phase_rms_reward'][row, column]), accepted=False))
        teachers = item['phase_teachers']
        for start in range(0, len(teachers), item['config'].targets_per_worker_task):
            stop = min(start+item['config'].targets_per_worker_task, len(teachers))
            chunk = teachers[start:stop]
            rows = np.array([teacher['row'] for teacher in chunk])
            columns = np.array([teacher['column'] for teacher in chunk])
            job = (len(work), start, stop, item['targets'][rows],
                   item['kappa'][rows, columns][:, None],
                   np.stack([teacher['phase'] for teacher in chunk])[:, None],
                   item['detunings'][rows], item['config'])
            work.append((group, 0, 1, job))
    return work, dict(phase_correction_locked_fraction=eligible/total,
        phase_correction_attempted=sum(len(item['phase_teachers']) for item in prepared),
        phase_correction_accepted=0, phase_correction_mean_reward_gain=0.)


def verify_teachers(prepared, results, settings, diagnostics, *, require_phase_success=False):
    """Accept only finite, still-locked corrections with a higher RMS reward."""
    gains = []
    coverage = [np.zeros(len(item['phase_teachers']), dtype=int) for item in prepared]
    for result in results:
        group = int(result['size_batch_index'])
        passes = locked(result, settings)[:, 0]
        if require_phase_success:
            passes = passes & result['phase_rms_success'][:, 0]
        for local, index in enumerate(range(int(result['start_index']), int(result['stop_index']))):
            coverage[group][index] += 1
            teacher = prepared[group]['phase_teachers'][index]
            gain = float(result['phase_rms_reward'][local, 0])-teacher['baseline_reward']
            teacher['accepted'] = bool(passes[local] and gain > 0.)
            if teacher['accepted']:
                gains.append(gain)
    if any(np.any(counts != 1) for counts in coverage):
        raise RuntimeError('Correction verification did not return every teacher exactly once')
    diagnostics.update(phase_correction_accepted=len(gains),
        phase_correction_mean_reward_gain=float(np.mean(gains)) if gains else 0.)


def teacher_phase_inputs(policy, item):
    """Return the existing phase means and fixed physical labels in policy coordinates."""
    teachers = [teacher for teacher in item.get('phase_teachers', ()) if teacher['accepted']]
    if not teachers:
        return [], None, None, None
    rows = np.array([teacher['row'] for teacher in teachers])
    columns = np.array([teacher['column'] for teacher in teachers])
    masks = item['gates'][rows, columns]
    roots = None if item['roots'] is None else item['roots'][rows, columns]
    mean, *_ = condition_on_roots(policy, roots).parameters_for_edges(
        item['features'][rows], topology_masks=masks)
    size = item['n_lasers']
    receivers, sources = core.directed_link_indices(size)
    # Physical labels include the root-reference translation. Move the labels
    # back to the policy's Gaussian coordinates using the existing decoder rule.
    rotation = apply_root_reference(np.zeros((len(teachers), 1, size, size)),
        masks[:, None], item['detunings'][rows],
        None if roots is None else roots[:, None], item['config'])[:, 0]
    physical = np.stack([teacher['phase'] for teacher in teachers])
    target = torch.as_tensor((physical-rotation)[:, receivers, sources],
                             dtype=mean.dtype, device=mean.device).detach()
    return teachers, mean[:, len(receivers):], target, masks


def teacher_losses(policy, item):
    """Per-case circular losses on active phase means, with fixed physical labels."""
    teachers, phases, target, masks = teacher_phase_inputs(policy, item)
    if not teachers:
        return [], None
    losses = ((1-torch.cos(phases-target))*masks).sum(dim=1)/masks.sum(dim=1)
    return teachers, losses
