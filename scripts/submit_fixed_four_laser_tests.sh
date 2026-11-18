#!/bin/bash
# Four independent fresh models; 4 CPUs, 16G and 8h per job.
# Initial learnable budgets/M for spans 1/2/3/4 GHz are 4/6/9/12 ns^-1.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iterations=1000
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        --iterations)
            if (($# < 2)); then echo '--iterations needs a value' >&2; exit 2; fi
            iterations="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/submit_fixed_four_laser_tests.sh [--smoke-test] [--dry-run] [--iterations N]"
            echo 'Submits fresh 1, 2, 3 and 4 GHz fixed-context models learning strengths, phases and budget.'
            echo 'Initial budgets/M: 4, 6, 9 and 12 ns^-1. Full jobs request 8 hours.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] && (( iterations >= 2 )) || { echo '--iterations must be at least two' >&2; exit 2; }
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
for span in 1 2 3 4; do
    name="four_fixed_${span}GHz"
    command=(sbatch --parsable --job-name="$name")
    if "$smoke_test"; then command+=(--time=00:20:00); fi
    command+=(scripts/train_fixed_four_laser_great_lakes.slurm --span-ghz "$span" --iterations "$iterations")
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
