"""Reward baselines for supplied-topology REINFORCE learning.

The RMS reward is calculated by the existing simulation worker. A baseline
only centers that reward for a policy-gradient update; it adds no loss term.
"""
import numpy as np

from ._compat import engine


def exclude_pair_baseline(centered, actions_per_topology):
    """Subtract the other pairs' mean, separately for each context and topology.

    Input can already be centered/scaled: a shared offset cancels in this
    subtraction. Keeping the complete evaluated pair out is important because
    mirrored partners are correlated. With eight actions the baseline uses
    the other six; no additional simulator call is needed.
    """
    if actions_per_topology < 4 or actions_per_topology % 2:
        raise ValueError("Pair-excluded baselines need an even action count >=4")
    centered = np.asarray(centered)
    groups = centered.reshape(len(centered), -1, actions_per_topology)
    pairs = groups.reshape(*groups.shape[:-1], actions_per_topology//2, 2)
    other_mean = (pairs.sum(axis=(-2, -1), keepdims=True)
                  - pairs.sum(axis=-1, keepdims=True)) / (actions_per_topology-2)
    return (pairs-other_mean).reshape(centered.shape)


def pair_excluded_advantages(rewards, config, reward_scale_state=None):
    """Use the existing global reward-SD EMA, updated exactly once per batch."""
    if config.topology_advantage_scale != "global_running":
        raise ValueError("Pair-excluded learning uses the shared global_running reward scale")
    centered = engine._topology_advantages(rewards, config, reward_scale_state)
    return [exclude_pair_baseline(value, config.actions_per_topology) for value in centered]
