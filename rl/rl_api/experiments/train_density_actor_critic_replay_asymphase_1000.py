"""Fresh replay actor–critic: symmetric strengths/masks and directed phases."""
from dataclasses import replace
from . import train_density_actor_critic_replay_1000 as source
from .. import Experiment

RUN_NAME='density_conditioned_ac_replay_symstrength_asymphase_fresh_1k'
INITIAL_CHECKPOINT=None
VERIFY_REFERENCE=False
vcsel=source.vcsel
CONFIG=replace(source.CONFIG,
    model=replace(source.CONFIG.model,asymmetric_coupling_phases=True),
    validation=replace(source.CONFIG.validation,progress_filename='density_ac_replay_asymphase_progress.png'))

if __name__=='__main__':
    experiment=Experiment(RUN_NAME,vcsel=vcsel,config=CONFIG)
    experiment.describe()
    experiment.train(live_plot=True,verify_reference=False,restart=False)
