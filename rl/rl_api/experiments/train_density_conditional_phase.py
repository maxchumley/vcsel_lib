"""Matched symmetry-bound refinement with phases conditioned on actual strengths."""
from .train_density_symmetry_refinement import main as refine


def main(argv=None):
    return refine(argv,projection_method='physical_bounds',components='both',condition_phases=True,
        name='density_conditional_phase_bounds_both_from_working_laser1',
        progress_filename='density_conditional_phase_progress.png')


if __name__=='__main__':main()
