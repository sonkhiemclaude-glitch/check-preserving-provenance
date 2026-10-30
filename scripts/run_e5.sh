#!/usr/bin/env bash
# Bedrock E5: credentials remain outside the source snapshot and captured results.
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
PY="${PROVPUB_PYTHON:-../.venv/bin/python}"
ENV_FILE="${PROVPUB_E5_ENV:-../private/bedrock.env}"
CONFIG_FILE="${PROVPUB_E5_CONFIG:-configs/e5/pilot.yaml}"
MODE="${1:-check}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
case "$MODE" in
  check) exec "$PY" scripts/run_e5.py --env "$ENV_FILE" --config "$CONFIG_FILE" --check-config ;;
  discover) exec "$PY" scripts/run_e5.py --env "$ENV_FILE" --config "$CONFIG_FILE" --discover ;;
  probe) exec "$PY" scripts/run_e5.py --env "$ENV_FILE" --config "$CONFIG_FILE" --probe ;;
  live|mock|resume) ;;
  *) echo 'Usage: bash scripts/run_e5.sh {check|discover|probe|mock|live|resume} [output]' >&2; exit 2 ;;
esac
DATA_ROOT="${PROVPUB_E5_DATA:-../data/diabetes}"
if [[ ! -f "$DATA_ROOT/manifest.yaml" ]]; then
  "$PY" scripts/run_e5.py --prepare-benchmark "$DATA_ROOT"
fi
if [[ "$MODE" == resume && -z "${2:-}" ]]; then
  echo 'resume requires the existing output directory as its second argument' >&2
  exit 2
fi
OUT="${2:-../runs/e5_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p ../runs/logs
EXTRA=()
if [[ "$MODE" == mock ]]; then EXTRA+=(--mock); fi
if [[ "$MODE" == resume ]]; then EXTRA+=(--resume); fi
"$PY" scripts/run_e5.py --env "$ENV_FILE" --config "$CONFIG_FILE" \
  --datasets "$DATA_ROOT/manifest.yaml" --out "$OUT" "${EXTRA[@]}" \
  2>&1 | tee "../runs/logs/$(basename "$OUT").log"
