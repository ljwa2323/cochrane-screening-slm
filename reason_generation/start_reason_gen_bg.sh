#!/usr/bin/env bash
# Resume-safe background launcher for gpt-oss-120b reason generation.
# Usage:
#   bash start_reason_gen_bg.sh
#   bash start_reason_gen_bg.sh --rps 20 --concurrency 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/reason_gen"
LOG_DIR="${OUT_DIR}/logs"
PID_FILE="${OUT_DIR}/reason_gen.pid"
SCRIPT="${SCRIPT_DIR}/generate_reasons_gpt_oss.py"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running: pid=${old_pid}"
    echo "Log: $(ls -t ${LOG_DIR}/reason_gen_*.log | head -1)"
    exit 0
  fi
fi

# Stop any stray previous process for this script.
pkill -f "python ${SCRIPT}" 2>/dev/null || true
sleep 2

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/reason_gen_${TS}.log"

# Extra CLI args are forwarded; defaults keep resume via --out-dir.
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  EXTRA_ARGS=(--rps 20 --concurrency 30 --max-retries 10 --log-every 100)
fi

cd "${PROJECT_DIR}"
nohup env PYTHONUNBUFFERED=1 python "${SCRIPT}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started pid=$(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Progress: ${OUT_DIR}/progress.jsonl"
echo "Resume later by running this script again (skips completed idxs)."
