#!/bin/bash
#SBATCH --job-name=vcsel4_sym
#SBATCH --account=arrays0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/four-sym-%j.out
#SBATCH --error=logs/four-sym-%j.err

# Submit from the repo root after mkdir -p logs. Reuse the tested environment
# activation and explicit Conda Python in the existing cluster wrapper.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VCSEL_SIM_WORKERS=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))
# The comparison launcher selects another matched four-laser recipe through
# this environment variable; ordinary submissions keep the original recipe.
VCSEL_TRAINING_MODULE="${VCSEL_TRAINING_MODULE:-train_four_laser_inphase_all_symmetric_1000}"
exec bash scripts/train_rl_great_lakes.slurm \
    "$VCSEL_TRAINING_MODULE" --workers "$VCSEL_SIM_WORKERS" "$@"
