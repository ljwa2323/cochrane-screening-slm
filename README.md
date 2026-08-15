# Cochrane Screening SLM

Pipeline code for Cochrane-style title/abstract screening with small language models:
reason generation, SFT data preparation, Qwen3 LoRA fine-tuning, and inference evaluation.

## Repository layout

```
reason_generation/   # generate screening reasons via API
data_prep/           # build train/val/test chat jsonl
training/            # LoRA SFT for Qwen3-1.7B / 4B / 8B
inference/           # base vs LoRA screening evaluation
data/                # place input CSVs here (not shipped)
models/              # place Qwen3 base models here
sft_data/            # generated SFT jsonl
sft_runs/            # LoRA adapter outputs
reason_gen/          # reason-generation checkpoints
results/             # evaluation outputs
```

## Setup

```bash
pip install -r requirements.txt
```

Put the NVIDIA API key in `api_key.txt` (single line).  
Download Qwen3 base models into `models/Qwen3-1.7B`, `models/Qwen3-4B`, `models/Qwen3-8B`.  
Place screening CSVs under `data/`.

Expected CSV columns include: `Selection_criteria`, `Title`, `Abstract_clean`, `label`.

## Usage

### 1. Reason generation

```bash
bash reason_generation/start_reason_gen_bg.sh
# or
python reason_generation/generate_reasons_gpt_oss.py \
  --input data/20240827_dev_set.csv \
  --out-dir reason_gen
```

Held-out external reviews (random / HIV / heart):

```bash
bash reason_generation/start_heldout_reviews_reason_bg.sh
```

This writes `sft_data/heldout_reviews{1,2,3}.jsonl`.

### 2. Build SFT dataset

```bash
python data_prep/build_sft_dataset.py \
  --csv data/20240827_dev_set.csv \
  --progress reason_gen/progress.jsonl \
  --out-dir sft_data
```

### 3. LoRA fine-tuning

```bash
bash training/start_sft_1_7b.sh
bash training/start_sft_4b.sh
bash training/start_sft_8b.sh
```

### 4. Inference / evaluation

```bash
python data_prep/make_test_strat10.py
bash inference/start_eval_test_all.sh
```

Single-model example:

```bash
python inference/eval_base_vs_lora_val.py \
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

- Large artifacts (base models, full datasets, LoRA weights, progress dumps) are not included.
- `inference/start_eval_val_all.sh` refuses to run; use the held-out test split instead.
- Override the Python interpreter with `PYTHON=/path/to/python` when launching eval scripts.
