#!/bin/bash
#SBATCH --job-name=vcsel_rl
#SBATCH --account=arrays0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/rl-%j.out
#SBATCH --error=logs/rl-%j.err

# Submit from the repository root, after running: mkdir -p logs
# Optional first argument: another module name, e.g. train_mirrored_1000.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load python3.11-anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VCSEL_CONDA_ENV:-vcsel}"

# One trainer launches the existing multiprocessing pool on this one node.
# Keep each process single-threaded rather than multiplying BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/vcsel-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

python --version
git rev-parse --short HEAD
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python -u -m rl.rl_api.experiments.train_batch "$@"
