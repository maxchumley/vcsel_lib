#!/bin/bash
# One additional fresh full-density model; existing jobs continue unchanged.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_density_conditioned_fresh.sh [--dry-run] [--smoke-test]'
            echo 'Fresh 3-9 laser full-density model with symmetric pair values and conditional phases.'
            echo '1000 iterations, 32 CPUs, 31 workers, 64 GB and seven days; no initialization checkpoint.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
command=(sbatch --parsable --job-name=density_cond_fresh --cpus-per-task=32 --time=7-00:00:00
    scripts/train_rl_great_lakes.slurm train_density_conditioned_fresh_1000
    --workers 31 --pairs-per-job 5 --actions-per-job 8)
if "$smoke_test"; then command+=(--smoke-test); fi
if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    job=$("${command[@]}"); echo "Submitted density_cond_fresh: $job"
fi
echo 'Existing jobs are unchanged. Plot sync discovers density_conditioned_sym_fresh_progress.png.'
