#!/bin/bash
# Submit matched fresh density-input models with symmetric physical values.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_symmetric_density_tests.sh [--dry-run] [--smoke-test]'
            echo 'Two fresh 1000-iteration density runs: symmetric strengths/phases, laser-1 or zero frequency reference.'
            echo 'Each job requests 32 CPUs, 64 GB and seven days; 31 simulation workers.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
recipes=(train_mirrored_symmetric_density_values_1000 train_mirrored_symmetric_density_values_zeroref_1000)
names=(density_sym_ref density_sym_zero)
for index in 0 1; do
    command=(sbatch --parsable --job-name="${names[$index]}" --cpus-per-task=32 --time=7-00:00:00
        scripts/train_rl_great_lakes.slurm "${recipes[$index]}"
        --workers 31 --pairs-per-job 5 --actions-per-job 8)
    if "$smoke_test"; then command+=(--smoke-test); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        job=$("${command[@]}"); echo "Submitted ${names[$index]}: $job"
    fi
done
echo 'Monitor: squeue -u "$USER" -o "%.18i %.24j %.9T %.10M %.20R"'
echo 'Mac plot sync: python3 scripts/sync_great_lakes_plot.py (automatic discovery; no restart needed)'
