# Cochrane Screening SLM

Code for **reason generation**, **Qwen3 LoRA SFT**, and **inference/evaluation** for title/abstract screening of systematic reviews.

Repository: https://github.com/ljwa2323/cochrane-screening-slm

## Pipeline overview

1. **Reason generation** (`generate_reasons_gpt_oss.py`)  
   Call NVIDIA `openai/gpt-oss-120b` to write justifications for gold labels.
2. **Build SFT data** (`build_sft_dataset.py`)  
   Merge reasons with the screening CSV into chat-format `train.jsonl` / `val.jsonl` / `test.jsonl`.
3. **LoRA fine-tuning** (`train_qwen3_lora.py`)  
   Train Qwen3-1.7B / 4B / 8B adapters (assistant-token loss only).
4. **Inference / eval** (`eval_base_vs_lora_val.py`)  
   Compare base vs LoRA models; output label + reason JSON.

## Layout

```
.
|-- generate_reasons_gpt_oss.py      # reason generation (API)
|-- start_reason_gen_bg.sh
|-- run_real_tests_reason_pipeline.py
|-- start_real_tests_reason_bg.sh
|-- build_real_test_jsonl.py
|-- build_sft_dataset.py             # build train/val/test jsonl
|-- make_test_strat10.py
|-- make_val_strat10.py
|-- train_qwen3_lora.py              # LoRA SFT
|-- start_sft_1_7b.sh
|-- start_sft_4b.sh
|-- start_sft_8b.sh
|-- eval_base_vs_lora_val.py         # main inference/eval
|-- eval_qwen3_lora_90.py
|-- test_qwen3_screening.py          # base-model smoke eval
|-- start_eval_test_all.sh
|-- start_eval_val_all.sh            # deprecated (refuses val as test)
|-- requirements.txt
|-- data/                            # place CSV inputs here (not shipped)
|-- models/                          # place Qwen3 base models here
|-- sft_data/                        # generated train/val/test jsonl
|-- sft_runs/                        # LoRA adapter outputs
|-- reason_gen/                      # reason generation checkpoints
|-- results/                         # eval outputs
|-- api_key.txt                      # NVIDIA API key (do not commit)
```

## Setup

```bash
pip install -r requirements.txt
```

Put your NVIDIA API key in `api_key.txt` (single line).  
Download Qwen3 base models into `models/Qwen3-1.7B`, `models/Qwen3-4B`, `models/Qwen3-8B`.  
Place screening CSVs under `data/` (or pass `--input` / CLI paths).

Expected CSV columns include: `Selection_criteria`, `Title`, `Abstract_clean`, `label`.

## Quick start

### 1) Generate reasons

```bash
bash start_reason_gen_bg.sh
# or
python generate_reasons_gpt_oss.py --input data/20240827_dev_set.csv --out-dir reason_gen
```

### 2) Build SFT dataset

```bash
python build_sft_dataset.py --csv data/20240827_dev_set.csv --progress reason_gen/progress.jsonl --out-dir sft_data
```

### 3) LoRA fine-tune

```bash
bash start_sft_1_7b.sh
bash start_sft_4b.sh
bash start_sft_8b.sh
```

### 4) Inference / evaluation

```bash
python make_test_strat10.py
bash start_eval_test_all.sh
```

Single-model eval example:

```bash
python eval_base_vs_lora_val.py \
  --val-file sft_data/test_strat10.jsonl \
  --base-model models/Qwen3-4B \
  --adapter sft_runs/qwen3_4b_lora \
  --out-dir results/test_base_vs_lora_all \
  --tag Qwen3-4B \
  --mode both \
  --file-stem test \
  --batch-size 8
```

## Notes

- Large artifacts (base models, full `sft_data`, LoRA weights, reason progress dumps) are **not** included in this repo.
- `start_eval_val_all.sh` intentionally refuses to run; use the held-out test split via `start_eval_test_all.sh`.
- Override the Python interpreter with `PYTHON=/path/to/python` when launching eval scripts.
