"""Matched fresh density experiment: PPO, symmetric strengths, directed phases."""
from dataclasses import replace
from . import train_density_actor_critic_replay_asymphase_1000 as source
from .. import Experiment, PPOConfig

RUN_NAME = 'density_conditioned_ppo_symstrength_asymphase_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    actor_critic=replace(source.CONFIG.actor_critic, enabled=False, use_replay=False),
    ppo=PPOConfig(enabled=True, clip_ratio=.1, updates=4, target_kl=.01),
    validation=replace(source.CONFIG.validation, progress_filename='density_ppo_progress.png'))

if __name__ == '__main__':
    experiment = Experiment(RUN_NAME, vcsel=vcsel, config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True, verify_reference=VERIFY_REFERENCE, restart=False)
