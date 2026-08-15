#!/usr/bin/env bash
# Evaluate Qwen3 1.7B / 4B / 8B (base + LoRA) on stratified 10% of test.jsonl.
# Uses 2 GPUs in parallel: GPU0=1.7B then 8B; GPU1=4B.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT_DIR="${PROJECT_DIR}/results/test_base_vs_lora_all"
LOG_DIR="${OUT_DIR}/logs"
TEST_FILE="${PROJECT_DIR}/sft_data/test_strat10.jsonl"
PID_FILE="${OUT_DIR}/eval.pid"
LIMIT="${LIMIT:-0}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "Missing ${TEST_FILE}. Run make_test_strat10.py first."
  exit 1
fi

run_one () {
  local gpu="$1"
  local tag="$2"
  local base="$3"
  local adapter="$4"
  local bs="$5"
  local logfile="${LOG_DIR}/${tag}.log"
  echo "[$(date '+%F %T')] START ${tag} on GPU ${gpu}" | tee -a "${logfile}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${PROJECT_DIR}/eval_base_vs_lora_val.py" \
    --val-file "${TEST_FILE}" \
    --base-model "${base}" \
    --adapter "${adapter}" \
    --out-dir "${OUT_DIR}" \
    --tag "${tag}" \
    --mode both \
    --file-stem test \
    --batch-size "${bs}" \
    --log-every 100 \
    --limit "${LIMIT}" \
    >> "${logfile}" 2>&1
  echo "[$(date '+%F %T')] DONE ${tag}" | tee -a "${logfile}"
}

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/launch_${TS}.log"

{
  echo "[$(date '+%F %T')] Launching test_strat10 evals. OUT_DIR=${OUT_DIR} LIMIT=${LIMIT}"

  run_one 0 "Qwen3-1.7B" \
    "${PROJECT_DIR}/models/Qwen3-1.7B" \
    "${PROJECT_DIR}/sft_runs/qwen3_1_7b_lora" \
    16 &
  PID17=$!

  run_one 1 "Qwen3-4B" \
    "${PROJECT_DIR}/models/Qwen3-4B" \
    "${PROJECT_DIR}/sft_runs/qwen3_4b_lora" \
    8 &
  PID4=$!

  wait "${PID17}"
  wait "${PID4}"

  run_one 0 "Qwen3-8B" \
    "${PROJECT_DIR}/models/Qwen3-8B" \
    "${PROJECT_DIR}/sft_runs/qwen3_8b_lora" \
    4

  "${PY}" - <<PY
from pathlib import Path
import pandas as pd
out = Path(r"${PROJECT_DIR}/results/test_base_vs_lora_all")
dfs = []
for p in sorted(out.glob("*_test_accuracy_summary.csv")):
    dfs.append(pd.read_csv(p))
if dfs:
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out / "all_test_accuracy_summary.csv", index=False)
    print(all_df.to_string(index=False))
else:
    print("No summary files found")
PY

  echo "[$(date '+%F %T')] All evals finished."
} > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started pid=$(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUT_DIR}"
