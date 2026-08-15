#!/usr/bin/env bash
# DEPRECATED: val.jsonl is the validation split and must not be used as a test set.
# Use start_eval_test_all.sh (sft_data/test_strat10.jsonl) instead.
set -euo pipefail
echo "Refusing to evaluate on val. Use start_eval_test_all.sh" >&2
exit 1
# Evaluate Qwen3 1.7B / 4B / 8B (base + LoRA) on sft_data/val.jsonl.
# Uses 2 GPUs in parallel: GPU0=1.7B then 8B; GPU1=4B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python}"
OUT_DIR="${PROJECT_DIR}/results/val_base_vs_lora_all"
LOG_DIR="${OUT_DIR}/logs"
VAL_FILE="${PROJECT_DIR}/sft_data/val_strat10.jsonl"
LIMIT="${LIMIT:-0}"
mkdir -p "${LOG_DIR}"

run_one () {
  local gpu="$1"
  local tag="$2"
  local base="$3"
  local adapter="$4"
  local bs="$5"
  local logfile="${LOG_DIR}/${tag}.log"
  echo "[$(date '+%F %T')] START ${tag} on GPU ${gpu}" | tee -a "${logfile}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT_DIR}/eval_base_vs_lora_val.py" \
    --val-file "${VAL_FILE}" \
    --base-model "${base}" \
    --adapter "${adapter}" \
    --out-dir "${OUT_DIR}" \
    --tag "${tag}" \
    --mode both \
    --batch-size "${bs}" \
    --log-every 100 \
    --limit "${LIMIT}" \
    >> "${logfile}" 2>&1
  echo "[$(date '+%F %T')] DONE ${tag}" | tee -a "${logfile}"
}

echo "[$(date '+%F %T')] Launching evals. OUT_DIR=${OUT_DIR} LIMIT=${LIMIT}"

# Parallel: 1.7B on GPU0, 4B on GPU1
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

# Then 8B on GPU0 (larger model; keep batch smaller)
run_one 0 "Qwen3-8B" \
  "${PROJECT_DIR}/models/Qwen3-8B" \
  "${PROJECT_DIR}/sft_runs/qwen3_8b_lora" \
  4

# Merge summaries
"${PY}" - <<PY
from pathlib import Path
import pandas as pd
out = Path(r"${PROJECT_DIR}/results/val_base_vs_lora_all")
dfs = []
for p in sorted(out.glob("*_val_accuracy_summary.csv")):
    dfs.append(pd.read_csv(p))
if dfs:
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out / "all_val_accuracy_summary.csv", index=False)
    print(all_df.to_string(index=False))
else:
    print("No summary files found")
PY

echo "[$(date '+%F %T')] All evals finished."
