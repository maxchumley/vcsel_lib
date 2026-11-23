#!/bin/bash
# Matched fresh actors: current-batch critic versus 100k-design replay critic.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
smoke_test=false
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_density_actor_critic_tests.sh [--dry-run] [--smoke-test]'
            echo 'Two fresh 3-9 laser symmetric density actors, both with laser-1 reference.'
            echo 'Each: 1000 iterations, 1400 real simulations/iteration, 500 ns.'
            echo '32 CPUs, 31 simulation workers, 64 GB and seven days per job.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if ! "$dry_run"; then
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
fi
for arm in current replay; do
    name=density_ac_${arm}
    recipe=train_density_actor_critic_1000
    if [[ $arm == replay ]]; then recipe=train_density_actor_critic_replay_1000; fi
    command=(sbatch --parsable --job-name="$name" --cpus-per-task=32 --mem=64G --time=7-00:00:00
        scripts/train_rl_great_lakes.slurm "$recipe"
        --workers 31 --pairs-per-job 5 --actions-per-job 8)
    if "$smoke_test"; then command+=(--smoke-test); fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        job=$("${command[@]}"); echo "Submitted $name: $job"
    fi
    echo "Plot sync discovers density_ac_${arm}_progress.png."
done
echo 'Existing jobs are unchanged.'
