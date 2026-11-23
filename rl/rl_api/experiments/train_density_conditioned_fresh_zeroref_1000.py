"""Matched fresh conditional-phase density training with zero frequency reference."""
from dataclasses import replace

from . import train_density_conditioned_fresh_1000 as source
from .. import Experiment

RUN_NAME = 'symmetric_density_pairs_conditioned_phase_zero_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = source.VERIFY_REFERENCE
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    model=replace(source.CONFIG.model,phase_parameterization='target_residual'),
    validation=replace(source.CONFIG.validation,
        progress_filename='density_conditioned_sym_zero_fresh_progress.png'))


if __name__ == '__main__':
    experiment = Experiment(RUN_NAME,vcsel=vcsel,config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True,verify_reference=VERIFY_REFERENCE,
                     restart=False,initial_checkpoint=INITIAL_CHECKPOINT)
