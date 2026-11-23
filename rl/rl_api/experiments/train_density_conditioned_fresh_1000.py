"""Fresh full-density training with reciprocal values and conditional phases.

Match the existing symmetric-pair density recipe, adding only the phase
conditioning head and unique output names. No checkpoint or symmetry ramp.
"""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

# The source recipe prints a preview on import. Show only this run's settings.
with redirect_stdout(StringIO()):
    from . import train_mirrored_symmetric_density_pairs_1000 as source

from .. import Experiment

RUN_NAME = 'symmetric_density_pairs_conditioned_phase_fresh_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
vcsel = source.vcsel
CONFIG = replace(source.CONFIG,
    model=replace(source.CONFIG.model,condition_phases_on_coupling=True),
    validation=replace(source.CONFIG.validation,
        progress_filename='density_conditioned_sym_fresh_progress.png'))


if __name__ == '__main__':
    experiment = Experiment(RUN_NAME,vcsel=vcsel,config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True,verify_reference=VERIFY_REFERENCE,
                     restart=False,initial_checkpoint=INITIAL_CHECKPOINT)
