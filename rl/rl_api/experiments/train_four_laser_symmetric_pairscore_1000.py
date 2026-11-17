"""Test A: shared reciprocal-pair score, original laser-1 phase baseline."""
from .four_laser_symmetric_comparison import comparison_config, run_comparison, vcsel

RUN_NAME = 'four_sym_A_pairscore_laser1_1k'
INITIAL_CHECKPOINT = None
VERIFY_REFERENCE = False
CONFIG = comparison_config(shared_pair_score=True, mean_reference=False,
                           progress_filename='four_A_pairscore_progress.png')

if __name__ == '__main__':
    result = run_comparison(RUN_NAME, CONFIG)
