"""Matched PPO experiment adding source-minus-receiver target phase offsets."""
from dataclasses import replace
from . import train_density_ppo_1000 as source
from .. import Experiment

RUN_NAME = 'density_conditioned_ppo_targetoffset_symstrength_asymphase_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    model=replace(source.CONFIG.model, add_target_phase_offset=True),
    validation=replace(source.CONFIG.validation, progress_filename='density_ppo_target_offset_progress.png'))

if __name__ == '__main__':
    experiment = Experiment(RUN_NAME, vcsel=vcsel, config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True, verify_reference=VERIFY_REFERENCE, restart=False)
