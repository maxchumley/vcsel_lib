#!/bin/bash
# Superseded by the requested warm-started symmetry refinement.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo 'Replaced the fresh asymmetric runs with refinement of the working density checkpoint.'
exec bash scripts/submit_density_symmetry_refinement.sh "$@"
