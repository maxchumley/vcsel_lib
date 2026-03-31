#!/bin/bash
#SBATCH --job-name=inj_noise_grid
#SBATCH --output=logs/inj_noise_grid_%j.out
#SBATCH --error=logs/inj_noise_grid_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Adjust environment setup for your cluster
source ~/.bashrc
conda activate vcsel

# Match these to detuning_vals_ghz and kappa_vals sizes in the Python script.
DETUNING_COUNT=2
KAPPA_COUNT=2

OUTDIR="examples/injection_tests/injection_noise_robustness_hpc"
mkdir -p "$OUTDIR"

# Run up to one task per allocated CPU core in parallel.
MAX_PARALLEL="${SLURM_CPUS_PER_TASK:-1}"
active=0

for ((di=0; di<DETUNING_COUNT; di++)); do
  for ((ki=0; ki<KAPPA_COUNT; ki++)); do
    srun --exclusive -N1 -n1 -c1 \
      python examples/working_examples/injection_noise_robustness.py \
        --detuning-idx "$di" \
        --kappa-idx "$ki" \
        --output-dir "$OUTDIR" \
        --save-name "result_d${di}_k${ki}.npz" \
        --no-plot &

    active=$((active + 1))
    if (( active >= MAX_PARALLEL )); then
      wait -n
      active=$((active - 1))
    fi
  done
done

wait
echo "All (detuning, kappa) tasks complete."
