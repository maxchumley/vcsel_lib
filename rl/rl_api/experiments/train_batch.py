"""Run an existing training recipe from a terminal or a Slurm batch job."""
import argparse
from dataclasses import replace
from importlib import import_module
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="train_mirrored_root_encoder_1000",
                        help="Training module under rl.rl_api.experiments")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, help="Override simulation workers only")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Two-iteration launch check with compressed action annealing")
    parser.add_argument("--preview", action="store_true", help="Describe settings without training")
    args = parser.parse_args(argv)

    # Importing the recipe does not execute its guarded training block.
    recipe = import_module(f"rl.rl_api.experiments.{args.experiment}")
    config = recipe.CONFIG
    optimization = config.optimization
    schedules = config.schedules
    if args.workers is not None:
        optimization = replace(optimization, workers=args.workers)
    if args.smoke_test:
        optimization = replace(optimization, iterations=2)
        if schedules.anneal_to_mean:
            schedules = replace(schedules, ordinary_start=1, ordinary_end=2,
                                wide_start=1, wide_end=2)
    config = replace(config,
                     optimization=optimization, schedules=schedules,
                     validation=replace(config.validation, notebook_display=False, plot_use_tex=False))
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    workers = config.optimization.workers
    required_cpus = workers + int(workers > 1)
    if allocated and required_cpus > int(allocated):
        parser.error(f"{config.optimization.workers} workers need at least "
                     f"{required_cpus} CPUs including the trainer; "
                     f"Slurm allocated {allocated}. Request more CPUs or use --workers.")

    job_id = os.environ.get("SLURM_JOB_ID")
    directory = args.output_directory
    if directory is None:
        from rl.rl_api._compat import PACKAGE_DIR
        directory = PACKAGE_DIR / "runs" / "great_lakes" / recipe.RUN_NAME / (job_id or "local")

    from rl.rl_api import Experiment
    import torch
    experiment = Experiment(recipe.RUN_NAME, config=config, vcsel=recipe.vcsel,
                            output_directory=directory)
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nSlurm job: {job_id or 'none'}", flush=True)
    if args.smoke_test:
        print("Smoke test only: two iterations; not a completed training run.", flush=True)
    experiment.describe()
    if args.preview:
        return experiment

    torch.set_num_threads(1)
    # Keep saving progress PNGs with the noninteractive Matplotlib backend.
    # Always start fresh in a new directory; never archive an existing run here.
    return experiment.train(live_plot=True,
                            verify_reference=getattr(recipe, "VERIFY_REFERENCE", True),
                            restart=False)


if __name__ == "__main__":
    main()
