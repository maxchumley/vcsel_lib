"""Fresh conditional density actor with a current-batch-only reward critic."""
from dataclasses import replace

from . import train_density_conditioned_fresh_1000 as source
from .. import ActorCriticConfig, Experiment

RUN_NAME = 'symmetric_density_conditioned_actor_critic_current_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    actor_critic=ActorCriticConfig(enabled=True, use_replay=False),
    validation=replace(source.CONFIG.validation,
        progress_filename='density_ac_current_progress.png'))


if __name__ == '__main__':
    experiment = Experiment(RUN_NAME, vcsel=vcsel, config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True, verify_reference=VERIFY_REFERENCE,
                     restart=False, initial_checkpoint=INITIAL_CHECKPOINT)
