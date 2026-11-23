#!/bin/bash
# Three new jobs; never cancel, replace or overwrite existing training jobs.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=rl/rl_api/runs/initialization/density_asym_laser1_83ad841f91a2.pt
updates=1000
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --checkpoint) checkpoint="${2:?--checkpoint needs a path}"; shift 2 ;;
        --updates) updates="${2:?--updates needs a count}"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_density_physical_bounds_tests.sh [--checkpoint PATH] [--updates N] [--dry-run] [--smoke-test]'
            echo 'Three additional warm-started runs: physical bounds on both values, strengths only, phases only.'
            echo 'Each requests 32 CPUs, 31 workers, 64 GB and seven days; unique run and plot names.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$updates" =~ ^[1-9][0-9]*$ ]] || { echo '--updates must be positive' >&2; exit 2; }
if ! "$dry_run"; then
    [[ -f "$checkpoint" ]] || { echo "Working checkpoint missing: $checkpoint" >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'Submit on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
components=(both strengths phases)
names=(density_bound_both density_bound_kappa density_bound_phase)
for i in 0 1 2; do
    command=(sbatch --parsable --job-name="${names[$i]}" scripts/train_density_physical_bounds.slurm
        --components "${components[$i]}" --initial-checkpoint "$checkpoint" --updates "$updates")
    if "$smoke_test"; then command+=(--smoke-test); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        job=$("${command[@]}"); echo "Submitted ${names[$i]}: $job"
    fi
done
echo 'Existing jobs are unchanged. Plot sync automatically discovers density_bound_*_progress.png.'
