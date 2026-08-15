"""
Evaluate a LoRA-finetuned Qwen3 screening model on a user-provided CSV eval set.

Expected CSV columns: Selection_criteria, Title, Abstract_clean, label
Gold label may be numeric (0/0.5/1) or text (exclude/uncertain/include).
Model outputs text labels: include | exclude | uncertain.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = PROJECT_DIR / "models" / "Qwen3-1.7B"
DEFAULT_ADAPTER = PROJECT_DIR / "sft_runs" / "qwen3_1_7b_lora"
RESULTS_DIR = PROJECT_DIR / "results"

VALID_LABELS = {"include", "exclude", "uncertain"}
MAX_NEW_TOKENS = 256

NUMERIC_TO_TEXT = {
    0.0: "exclude",
    0.5: "uncertain",
    1.0: "include",
}

SYSTEM_PROMPT = """You are an expert systematic reviewer performing title and abstract screening.
Given the review Selection_criteria, the study Title, and the Abstract, decide whether the study should be included.

Labels:
- include: clearly meets selection criteria
- exclude: clearly does not meet selection criteria
- uncertain: insufficient information to decide

Respond with ONLY a JSON object in this exact format:
{"label": "include" | "exclude" | "uncertain", "reason": "<brief explanation>"}
Do not output any other text."""


def normalize_label(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in VALID_LABELS:
            return text
        try:
            value = float(text)
        except Exception:
            return None
    try:
        num = float(value)
    except Exception:
        return None
    return NUMERIC_TO_TEXT.get(num)


def build_user_prompt(row: pd.Series) -> str:
    criteria = "" if pd.isna(row.get("Selection_criteria")) else str(row.get("Selection_criteria"))
    title = "" if pd.isna(row.get("Title")) else str(row.get("Title"))
    abstract = "" if pd.isna(row.get("Abstract_clean")) else str(row.get("Abstract_clean"))
    return (
        f"Selection_criteria:\n{criteria}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Decide the screening label and provide a brief reason."
    )


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response(text: str) -> tuple[str | None, str]:
    text = strip_thinking(text.strip())
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    brace = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            label = normalize_label(obj.get("label"))
            reason = str(obj.get("reason", "")).strip()
            return label, reason
        except Exception:
            continue
    # Fallback: look for bare label words
    lower = text.lower()
    for lab in ("include", "exclude", "uncertain"):
        if re.search(rf"\b{lab}\b", lower):
            return lab, text
    return None, text


def generate(model, tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA screening model on a user-provided CSV eval set."
    )
    parser.add_argument("--base-model", type=str, default=str(BASE_MODEL))
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Eval CSV with columns Selection_criteria, Title, Abstract_clean, label",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "Qwen3-1.7B-LoRA_eval_results.csv"),
    )
    args = parser.parse_args()

    sample_path = Path(args.sample)
    if not sample_path.exists():
        raise FileNotFoundError(f"Eval sample CSV not found: {sample_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_path)
    print(f"Sample size={len(sample)}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    records = []
    t0 = time.time()
    for i, row in sample.iterrows():
        gold_label = normalize_label(row.get("label"))
        try:
            raw = generate(model, tokenizer, build_user_prompt(row))
            pred_label, reason = parse_response(raw)
            error = ""
        except Exception as exc:
            raw, pred_label, reason, error = "", None, "", str(exc)
        records.append(
            {
                "model": "Qwen3-1.7B-LoRA",
                "row_id": int(row["Unnamed: 0"]) if "Unnamed: 0" in row else i,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "reason": reason,
                "raw_output": raw,
                "error": error,
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(sample)}] pred={pred_label} gold={gold_label}")

    result_df = pd.DataFrame(records)
    result_df.to_csv(args.output, index=False)
    comparable = result_df.dropna(subset=["pred_label", "gold_label"])
    acc = (
        float((comparable["pred_label"] == comparable["gold_label"]).mean())
        if len(comparable)
        else None
    )
    print(f"Saved {args.output}")
    print(f"Parsed={len(comparable)}/{len(result_df)} accuracy={acc}")
    if len(comparable):
        print(
            pd.crosstab(
                comparable["gold_label"],
                comparable["pred_label"],
                rownames=["gold"],
                colnames=["pred"],
            )
        )
    print(f"Elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
