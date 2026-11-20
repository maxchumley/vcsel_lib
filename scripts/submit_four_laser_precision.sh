#!/bin/bash
# One 4 GHz symmetric precision continuation; same 32 CPUs, 64G and 48h allocation.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=rl/rl_api/runs/great_lakes/four_fixed_fresh_4ghz/66366725/best.pt
iterations=1000
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --checkpoint)
            if (($# < 2)); then echo '--checkpoint needs a path' >&2; exit 2; fi
            checkpoint="$2"; shift 2 ;;
        --iterations)
            if (($# < 2)); then echo '--iterations needs a value' >&2; exit 2; fi
            iterations="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_four_laser_precision.sh [--checkpoint PATH] [--iterations N] [--smoke-test] [--dry-run]'
            echo 'Continues best symmetric 4 GHz checkpoint from job 66366725, with fresh Adam and 128 narrow mirrored samples.'
            echo 'Requests 32 CPUs, 64 GB and 48 hours; writes four_fixed_4ghz_progress_precision.png.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] && (( iterations >= 2 )) || { echo '--iterations must be at least two' >&2; exit 2; }
command=(sbatch --parsable --job-name=four_4GHz_precision scripts/train_fixed_four_laser_great_lakes.slurm
    --span-ghz 4 --precision-checkpoint "$checkpoint" --iterations "$iterations" --actions-per-iteration 128)
if "$smoke_test"; then command+=(--smoke-test); fi
if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
    job=$("${command[@]}")
    echo "Submitted four_4GHz_precision: $job"
fi
echo 'Mac plot sync: python3 scripts/sync_great_lakes_plot.py --fixed-context-tests --include-asymmetric --include-precision'
