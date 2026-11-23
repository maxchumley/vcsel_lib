"""Matched warm refinements with explicit physical reciprocity limits."""
import argparse
from .train_density_symmetry_refinement import main as refine


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--components',choices=('both','strengths','phases'),default='both')
    args,remaining=parser.parse_known_args(argv)
    label={'both':'both','strengths':'kappa','phases':'phase'}[args.components]
    return refine(remaining,projection_method='physical_bounds',components=args.components,
        name=f'density_physical_bounds_{label}_from_working_laser1',
        progress_filename=f'density_bound_{label}_progress.png')


if __name__=='__main__':main()
