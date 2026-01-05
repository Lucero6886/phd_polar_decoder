#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_SCRIPT="run_bler_sc.py"
RESULTS_CSV="10_baseline_sc/results.csv"
LOG_DIR="10_baseline_sc/logs"
RUN_DIR="10_baseline_sc/runs"
FIG_DIR="10_baseline_sc/figs"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$FIG_DIR"

TMP_OUT="$(mktemp)"
python3 -u "$PY_SCRIPT" | tee "$TMP_OUT"

RUN_ID="$(grep -m1 '^\[RUN\]' "$TMP_OUT" | awk '{print $2}')"
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] Cannot find RUN_ID from output."
  exit 1
fi

LOG_PATH="$LOG_DIR/run_${RUN_ID}.txt"
cp "$TMP_OUT" "$LOG_PATH"
rm -f "$TMP_OUT"
echo "[Saved] Log: $LOG_PATH"

RUN_CSV="$RUN_DIR/results_${RUN_ID}.csv"
{ head -n 1 "$RESULTS_CSV"; grep ",${RUN_ID}," "$RESULTS_CSV"; } > "$RUN_CSV"
echo "[Saved] Run CSV: $RUN_CSV"

python3 -u scripts/plot_bler_ber.py --input "$RUN_CSV" --outdir "$FIG_DIR"
echo "[Done] Figures saved to: $FIG_DIR"
