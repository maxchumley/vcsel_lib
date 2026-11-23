#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command=(sbatch --parsable --job-name=density_ac_asymphase --cpus-per-task=32 --mem=64G --time=7-00:00:00
    scripts/train_rl_great_lakes.slurm train_density_actor_critic_replay_asymphase_1000
    --workers 31 --pairs-per-job 5 --actions-per-job 8)
case "${1:-}" in
    --dry-run) printf '%q ' "${command[@]}"; printf '\n' ;;
    --smoke-test) mkdir -p logs; "${command[@]}" --smoke-test ;;
    '') mkdir -p logs; "${command[@]}" ;;
    *) echo 'Usage: bash scripts/submit_density_ac_replay_asymphase.sh [--dry-run|--smoke-test]' >&2; exit 2 ;;
esac
