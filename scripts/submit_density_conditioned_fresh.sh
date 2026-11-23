#!/bin/bash
# One additional fresh full-density model; existing jobs continue unchanged.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
smoke_test=false
zero_reference=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        --zero-reference) zero_reference=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_density_conditioned_fresh.sh [--zero-reference] [--dry-run] [--smoke-test]'
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
name=density_cond_fresh
recipe=train_density_conditioned_fresh_1000
figure=density_conditioned_sym_fresh_progress.png
if "$zero_reference"; then
    name=density_cond_zero
    recipe=train_density_conditioned_fresh_zeroref_1000
    figure=density_conditioned_sym_zero_fresh_progress.png
fi
command=(sbatch --parsable --job-name="$name" --cpus-per-task=32 --time=7-00:00:00
    scripts/train_rl_great_lakes.slurm "$recipe"
    --workers 31 --pairs-per-job 5 --actions-per-job 8)
if "$smoke_test"; then command+=(--smoke-test); fi
if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    job=$("${command[@]}"); echo "Submitted $name: $job"
fi
echo "Existing jobs are unchanged. Plot sync discovers $figure."
