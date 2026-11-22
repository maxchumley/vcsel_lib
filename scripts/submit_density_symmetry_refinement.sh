#!/bin/bash
# Continue the working asymmetric density model; one job replaces the fresh pair.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=rl/rl_api/runs/initialization/density_asym_laser1_83ad841f91a2.pt
dry_run=false
smoke_test=false
updates=1000
while (($#)); do
    case "$1" in
        --checkpoint) checkpoint="${2:?--checkpoint needs a path}"; shift 2 ;;
        --updates) updates="${2:?--updates needs a value}"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --smoke-test) smoke_test=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/submit_density_symmetry_refinement.sh [--checkpoint PATH] [--updates N] [--dry-run] [--smoke-test]'
            echo 'One warm-started refinement: laser-1 reference, 32 CPUs, 31 workers, 64 GB, seven days.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$updates" =~ ^[1-9][0-9]*$ ]] || { echo '--updates must be positive' >&2; exit 2; }
command=(sbatch --parsable scripts/train_density_symmetry_refinement.slurm --initial-checkpoint "$checkpoint" --updates "$updates")
if "$smoke_test"; then command+=(--smoke-test); fi
if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    [[ -f "$checkpoint" ]] || { echo "Working checkpoint missing: $checkpoint" >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'Run this launcher on Great Lakes.' >&2; exit 1; }
    mkdir -p logs
    job=$("${command[@]}"); echo "Submitted density_sym_refine: $job"
fi
echo 'Progress image: density_symmetry_refinement_progress.png (existing plot sync discovers it)'
