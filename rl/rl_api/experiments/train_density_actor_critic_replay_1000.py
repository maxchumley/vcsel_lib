"""Matched fresh conditional density actor; critic reuses up to 100k designs."""
from dataclasses import replace

from . import train_density_actor_critic_1000 as source
from .. import Experiment

RUN_NAME = 'symmetric_density_conditioned_actor_critic_replay_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    actor_critic=replace(source.CONFIG.actor_critic, use_replay=True),
    validation=replace(source.CONFIG.validation,
        progress_filename='density_ac_replay_progress.png'))


if __name__ == '__main__':
    experiment = Experiment(RUN_NAME, vcsel=vcsel, config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True, verify_reference=VERIFY_REFERENCE,
                     restart=False, initial_checkpoint=INITIAL_CHECKPOINT)
