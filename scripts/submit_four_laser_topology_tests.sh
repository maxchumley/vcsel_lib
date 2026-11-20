#!/bin/bash
# Four warm-started all-38-topology runs at fixed 1/2/3/4 GHz detunings.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_root=rl/rl_api/runs/great_lakes
search=1000
precision=1000
actions=128
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --checkpoint-root|--search-updates|--precision-updates|--actions-per-topology)
            if (($# < 2)); then echo "$1 needs a value" >&2; exit 2; fi
            case "$1" in
                --checkpoint-root) checkpoint_root="$2" ;;
                --search-updates) search="$2" ;;
                --precision-updates) precision="$2" ;;
                --actions-per-topology) actions="$2" ;;
            esac
            shift 2 ;;
        --smoke-test) smoke_test=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_four_laser_topology_tests.sh [--smoke-test] [--dry-run] [--checkpoint-root PATH] [--search-updates N] [--precision-updates N] [--actions-per-topology N]'
            echo 'Defaults: all 38 topologies, 128 samples each, 1000 search + 1000 precision updates and final mean check.'
            echo 'Each job requests 32 CPUs, 64 GB and four days. Starts from the matching fixed-case best checkpoint.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
for count in "$search" "$precision"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Update counts must be positive integers' >&2; exit 2; }
done
[[ "$actions" =~ ^[1-9][0-9]*$ ]] && (( actions >= 8 && actions % 8 == 0 )) || { echo 'Actions must be a positive multiple of eight' >&2; exit 2; }
jobs=(66366722 66366723 66366724 66366725)
# Check every input before submitting any job.
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    for span in 1 2 3 4; do
        checkpoint="$checkpoint_root/four_fixed_fresh_${span}ghz/${jobs[$((span-1))]}/best.pt"
        [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
    done
    mkdir -p logs
fi
for span in 1 2 3 4; do
    name="four_all38_${span}GHz"
    checkpoint="$checkpoint_root/four_fixed_fresh_${span}ghz/${jobs[$((span-1))]}/best.pt"
    command=(sbatch --parsable --job-name="$name" scripts/train_four_laser_topologies_fixed_great_lakes.slurm
        --span-ghz "$span" --initial-checkpoint "$checkpoint" --search-updates "$search"
        --precision-updates "$precision" --actions-per-topology "$actions")
    if "$smoke_test"; then command+=(--smoke-test); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        job=$("${command[@]}");echo "Submitted $name: $job"
    fi
done
echo 'Monitor: squeue -u "$USER" -o "%.18i %.24j %.9T %.10M %.20R"'
echo 'Mac plot sync: python3 scripts/sync_great_lakes_plot.py (automatic discovery; no restart needed)'
