"""Test B: original directed score, zero mean-detuning phase baseline."""
from .four_laser_symmetric_comparison import comparison_config, run_comparison, vcsel

RUN_NAME = 'four_sym_B_directed_meanref_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
CONFIG = comparison_config(shared_pair_score=False, mean_reference=True,
                           progress_filename='four_B_meanref_progress.png')

if __name__ == '__main__':
    result = run_comparison(RUN_NAME, CONFIG)
