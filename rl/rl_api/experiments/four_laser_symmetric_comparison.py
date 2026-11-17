"""Build matched comparisons from the original four-laser symmetric recipe."""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

# The original editable runner prints a preview on import. Keep that unrelated
# preview out of comparison logs; train_batch prints the actual chosen recipe.
with redirect_stdout(StringIO()):
    from . import train_four_laser_inphase_all_symmetric_1000 as baseline

vcsel = baseline.vcsel


def comparison_config(*, shared_pair_score, mean_reference, progress_filename):
    config = baseline.CONFIG
    return replace(config,
        model=replace(config.model, phase_parameterization=(
            'target_residual' if mean_reference else 'target_delay_residual')),
        sampling=replace(config.sampling, symmetric_action_score=(
            'shared_pair' if shared_pair_score else 'directed')),
        validation=replace(config.validation, progress_filename=progress_filename))


def run_comparison(name, config):
    from ..api import Experiment
    experiment = Experiment(name, config=config, vcsel=vcsel)
    experiment.describe()
    return experiment.train(live_plot=True, verify_reference=False, restart=False,
                            initial_checkpoint=None)
