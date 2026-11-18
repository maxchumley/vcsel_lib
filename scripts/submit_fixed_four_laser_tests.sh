#!/bin/bash
# Four independent fresh models; 32 CPUs, 64G and 48h per job.
# 128 sampled designs = 64 mirrored directions, plus one central prediction.
# Initial learnable budgets/M for spans 1/2/3/4 GHz are 4/6/9/12 ns^-1.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iterations=1000
actions=128
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        --iterations)
            if (($# < 2)); then echo '--iterations needs a value' >&2; exit 2; fi
            iterations="$2"; shift 2 ;;
        --actions-per-iteration)
            if (($# < 2)); then echo '--actions-per-iteration needs a value' >&2; exit 2; fi
            actions="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/submit_fixed_four_laser_tests.sh [--smoke-test] [--dry-run] [--iterations N] [--actions-per-iteration N]"
            echo 'Submits fresh 1, 2, 3 and 4 GHz fixed-context models learning strengths, phases and budget.'
            echo 'Defaults: 128 sampled designs (64 mirrored directions), 32 CPUs, 64G and 48 hours per job.'
            echo 'Initial budgets/M: 4, 6, 9 and 12 ns^-1. Smoke tests retain the 48-hour request.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] && (( iterations >= 2 )) || { echo '--iterations must be at least two' >&2; exit 2; }
[[ "$actions" =~ ^[1-9][0-9]*$ ]] && (( actions >= 8 && actions % 8 == 0 )) || { echo '--actions-per-iteration must be a positive multiple of eight' >&2; exit 2; }
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
for span in 1 2 3 4; do
    name="four_dirs$(( actions / 2 ))_${span}GHz"
    command=(sbatch --parsable --job-name="$name")
    command+=(scripts/train_fixed_four_laser_great_lakes.slurm --span-ghz "$span" --iterations "$iterations"
        --actions-per-iteration "$actions")
    if "$smoke_test"; then command+=(--smoke-test); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        job=$("${command[@]}")
        echo "Submitted $name: $job"
    fi
done
echo 'Monitor: squeue -u "$USER" -o "%.18i %.24j %.9T %.10M %.20R"'
echo 'Mac plot sync: python3 scripts/sync_great_lakes_plot.py --fixed-context-tests'
