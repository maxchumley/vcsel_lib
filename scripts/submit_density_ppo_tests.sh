#!/bin/bash
# Submit two independent matched tests; no running job is cancelled.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
    ''|--dry-run|--smoke-test) ;;
    *) echo 'Usage: bash scripts/submit_density_ppo_tests.sh [--dry-run|--smoke-test]' >&2; exit 2 ;;
esac
for variant in base offset; do
    if [[ "$variant" == base ]]; then
        name=density_ppo
        recipe=train_density_ppo_1000
    else
        name=density_ppo_offset
        recipe=train_density_ppo_target_offset_1000
    fi
    command=(sbatch --parsable --job-name="$name" --cpus-per-task=32 --mem=64G --time=7-00:00:00
        scripts/train_rl_great_lakes.slurm "$recipe" --workers 31 --pairs-per-job 5 --actions-per-job 8)
    if [[ "$mode" == --dry-run ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        mkdir -p logs
        if [[ "$mode" == --smoke-test ]]; then command+=(--smoke-test); fi
        printf '%s: ' "$name"
        "${command[@]}"
    fi
done
