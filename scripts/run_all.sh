#!/usr/bin/env bash
# Run from a deployed code snapshot; all measurements use one native thread.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PROVPUB_PYTHON:-../.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
MODE="${1:-reference}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${PROVPUB_RUN_ROOT:-../runs/$STAMP-$MODE}"
mkdir -p "$RUN_ROOT"
case "$MODE" in
  reference) CONFIGS=(e1_e3 e2 e4) ;;
  e6) CONFIGS=(e4_extended) ;;
  *) echo 'Usage: bash scripts/run_all.sh {reference|e6}' >&2; exit 2 ;;
esac
for CONFIG in "${CONFIGS[@]}"; do
  "$PY" scripts/run_experiment.py --config "configs/$CONFIG.yaml" --out "$RUN_ROOT/$CONFIG" \
    2>&1 | tee "$RUN_ROOT/$CONFIG.log"
done
echo "Results: $RUN_ROOT"
