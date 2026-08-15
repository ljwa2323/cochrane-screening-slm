#!/usr/bin/env bash
# Launch Qwen3-1.7B LoRA SFT on 2 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/sft_runs/qwen3_1_7b_lora"
LOG_DIR="${OUT_DIR}/logs"
PID_FILE="${OUT_DIR}/train.pid"
SCRIPT="${SCRIPT_DIR}/train_qwen3_lora.py"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running: pid=${old_pid}"
    exit 0
  fi
fi

pkill -f "train_qwen3_lora.py" 2>/dev/null || true
sleep 1

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TS}.log"

cd "${PROJECT_DIR}"
nohup env PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 \
  "${SCRIPT}" \
  --model-path "${PROJECT_DIR}/models/Qwen3-1.7B" \
  --train-file "${PROJECT_DIR}/sft_data/train.jsonl" \
  --val-file "${PROJECT_DIR}/sft_data/val.jsonl" \
  --output-dir "${OUT_DIR}" \
  --max-length 2048 \
  --num-train-epochs 2 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  "$@" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started pid=$(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUT_DIR}"
