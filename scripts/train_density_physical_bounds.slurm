#!/bin/bash
#SBATCH --job-name=density_bounds
#SBATCH --account=arrays0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/rl-%j.out
#SBATCH --error=logs/rl-%j.err

# Submit from the repository root, after running: mkdir -p logs
# Additional physical-bound test; existing refinement jobs are untouched.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load python3.11-anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VCSEL_CONDA_ENV:-vcsel}"
VCSEL_PYTHON="${CONDA_PREFIX}/bin/python"

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

"$VCSEL_PYTHON" --version
git rev-parse --short HEAD
# Use the activated environment explicitly, and export its settings to each step.
# Print the actual compute-node interpreter before trying to import the RL API.
srun --export=ALL --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$VCSEL_PYTHON" -u -c 'import sys; print("Batch Python:", sys.executable, flush=True); import torch; print("Batch PyTorch:", torch.__file__, flush=True)'
srun --export=ALL --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$VCSEL_PYTHON" -u -m rl.rl_api.experiments.train_density_physical_bounds \
    --workers "$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))" "$@"
