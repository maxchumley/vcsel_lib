#!/bin/bash
# Run on Great Lakes. Submit three independent jobs, each with 32 CPUs/64G/48h.
# Each job starts fresh; Slurm can run them concurrently when nodes are available.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true ;;
        --smoke-test) smoke_test=true ;;
        -h|--help)
            echo "Usage: bash scripts/submit_four_laser_symmetric_tests.sh [--smoke-test] [--dry-run]"
            echo "Full runs: three named jobs, 32 CPUs/64G/48h each. Smoke tests: two iterations, 1h."
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo "Run this launcher on Great Lakes (sbatch is required)." >&2; exit 1; }
    mkdir -p logs
fi

for test in \
    'four_A_pairs|train_four_laser_symmetric_pairscore_1000' \
    'four_B_meanref|train_four_laser_symmetric_meanref_1000' \
    'four_C_both|train_four_laser_symmetric_pairscore_meanref_1000'; do
    IFS='|' read -r job_name training_module <<< "$test"
    command=(sbatch --parsable --job-name="$job_name"
        --output='logs/%x-%j.out' --error='logs/%x-%j.err'
        --export="ALL,VCSEL_TRAINING_MODULE=$training_module")
    training_args=()
    if "$smoke_test"; then
        command+=(--time=01:00:00)
        training_args+=(--smoke-test)
    fi
    command+=(scripts/train_four_laser_symmetric_great_lakes.slurm)
    if "$smoke_test"; then command+=("${training_args[@]}"); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        job_id=$("${command[@]}")
        echo "Submitted $job_name: Slurm job $job_id"
    fi
done

echo 'Monitor names: squeue -u "$USER" -o "%.18i %.24j %.9T %.10M %.20R"'
echo 'On your Mac, sync the three images: python3 scripts/sync_great_lakes_plot.py --symmetric-tests'
