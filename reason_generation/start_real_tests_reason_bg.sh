#!/usr/bin/env bash
# Background launcher for real_test1/2/3 reason generation + jsonl build.
# Mapping:
#   real_test1.jsonl <- 20240827_random_test_set.csv
#   real_test2.jsonl <- 20240827_HIV_test_set.csv
#   real_test3.jsonl <- 20240827_heart_test_set.csv
#
# Usage:
#   bash start_real_tests_reason_bg.sh
#   bash start_real_tests_reason_bg.sh --rps 10 --concurrency 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/reason_gen_real_tests"
LOG_DIR="${OUT_DIR}/logs"
PID_FILE="${OUT_DIR}/pipeline.pid"
SCRIPT="${SCRIPT_DIR}/run_real_tests_reason_pipeline.py"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running: pid=${old_pid}"
    echo "Log: $(ls -t ${LOG_DIR}/pipeline_*.log | head -1)"
    exit 0
  fi
fi

# Avoid colliding with another generate_reasons job on the same API budget.
if pgrep -f "generate_reasons_gpt_oss.py" >/dev/null 2>&1; then
  echo "Another generate_reasons_gpt_oss.py process is already running. Abort."
  pgrep -af "generate_reasons_gpt_oss.py" || true
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${TS}.log"

EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  # Match reason_gen_val defaults for stability.
  EXTRA_ARGS=(--rps 5 --concurrency 2 --max-retries 10 --log-every 100)
fi

cd "${PROJECT_DIR}"
nohup env PYTHONUNBUFFERED=1 python "${SCRIPT}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started pid=$(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Outputs:"
echo "  ${PROJECT_DIR}/sft_data/real_test1.jsonl  # random"
echo "  ${PROJECT_DIR}/sft_data/real_test2.jsonl  # HIV"
echo "  ${PROJECT_DIR}/sft_data/real_test3.jsonl  # heart"
echo "Progress dirs:"
echo "  ${PROJECT_DIR}/reason_gen_real_test1"
echo "  ${PROJECT_DIR}/reason_gen_real_test2"
echo "  ${PROJECT_DIR}/reason_gen_real_test3"
