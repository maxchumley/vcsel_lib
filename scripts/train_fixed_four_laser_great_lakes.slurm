#!/bin/bash
#SBATCH --job-name=four_fixed
#SBATCH --account=arrays0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from the repository root; the launcher supplies a fixed detuning span.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load python3.11-anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VCSEL_CONDA_ENV:-vcsel}"
VCSEL_PYTHON="${CONDA_PREFIX}/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/vcsel-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
VCSEL_FIXED_WORKERS=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))
"$VCSEL_PYTHON" --version
git rev-parse --short HEAD
srun --export=ALL --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$VCSEL_PYTHON" -u -m rl.rl_api.experiments.train_fixed_four_laser_context \
    --workers "$VCSEL_FIXED_WORKERS" "$@"
