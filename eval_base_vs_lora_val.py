"""
Compare base Qwen3 vs LoRA-finetuned model on sft_data/val.jsonl.
Supports 1.7B / 4B / 8B via CLI args, with optional batched generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
VALID_LABELS = {"include", "exclude", "uncertain"}
MAX_NEW_TOKENS = 256

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
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in VALID_LABELS:
            return text
    return None


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
    lower = text.lower()
    for lab in ("include", "exclude", "uncertain"):
        if re.search(rf"\b{lab}\b", lower):
            return lab, text
    return None, text


def load_val(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold = normalize_label(obj.get("label"))
            if gold is None:
                try:
                    gold = normalize_label(
                        json.loads(obj["messages"][-1]["content"]).get("label")
                    )
                except Exception:
                    gold = None
            user = ""
            for msg in obj["messages"]:
                if msg.get("role") == "user":
                    user = msg.get("content", "")
                    break
            rec = {
                "row_id": obj.get("row_id"),
                "gold_label": gold,
                "user_content": user,
            }
            for key in ("dataset", "review_url", "review_title"):
                if key in obj:
                    rec[key] = obj.get(key)
            rows.append(rec)
    return rows


def build_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_batch(model, tokenizer, user_contents: list[str]) -> list[str]:
    prompts = [build_prompt(tokenizer, u) for u in user_contents]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    texts = []
    for i, ids in enumerate(output_ids):
        # Left padding: generated tokens start after the full padded length
        # Prefer slicing by original non-pad length from the right of the prompt region.
        prompt_len = int(input_lens[i])
        # With left padding, non-pad tokens are at the end of the prompt region.
        # input_ids shape[1] is the padded prompt length for all rows.
        padded_prompt_len = inputs["input_ids"].shape[1]
        gen = ids[padded_prompt_len:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
        _ = prompt_len  # kept for clarity / future debug
    return texts


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(
    name: str,
    model,
    tokenizer,
    rows: list[dict],
    log_every: int,
    batch_size: int,
) -> pd.DataFrame:
    records = []
    t0 = time.time()
    n = len(rows)
    for start in range(0, n, batch_size):
        batch = rows[start : start + batch_size]
        try:
            raws = generate_batch(
                model, tokenizer, [r["user_content"] for r in batch]
            )
            for row, raw in zip(batch, raws):
                pred, reason = parse_response(raw)
                rec = {
                    "model": name,
                    "row_id": row["row_id"],
                    "gold_label": row["gold_label"],
                    "pred_label": pred,
                    "reason": reason,
                    "raw_output": raw,
                    "error": "",
                }
                for key in ("dataset", "review_url", "review_title"):
                    if key in row:
                        rec[key] = row[key]
                records.append(rec)
        except Exception as exc:
            for row in batch:
                rec = {
                    "model": name,
                    "row_id": row["row_id"],
                    "gold_label": row["gold_label"],
                    "pred_label": None,
                    "reason": "",
                    "raw_output": "",
                    "error": str(exc),
                }
                for key in ("dataset", "review_url", "review_title"):
                    if key in row:
                        rec[key] = row[key]
                records.append(rec)
        done = min(start + batch_size, n)
        if done % log_every < batch_size or done == n or start == 0:
            elapsed = time.time() - t0
            last = records[-1]
            print(
                f"[{name}] {done}/{n} "
                f"elapsed={elapsed:.1f}s pred={last['pred_label']} gold={last['gold_label']}",
                flush=True,
            )
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, name: str) -> dict:
    comparable = df.dropna(subset=["pred_label", "gold_label"])
    acc = (
        float((comparable["pred_label"] == comparable["gold_label"]).mean())
        if len(comparable)
        else None
    )
    print(f"\n===== {name} =====")
    print(f"Parsed={len(comparable)}/{len(df)} accuracy={acc}")
    if len(comparable):
        print("Confusion (gold x pred):")
        print(
            pd.crosstab(
                comparable["gold_label"],
                comparable["pred_label"],
                rownames=["gold"],
                colnames=["pred"],
            )
        )
        print("Accuracy by gold:")
        print(
            comparable.assign(ok=comparable["pred_label"] == comparable["gold_label"])
            .groupby("gold_label")["ok"]
            .mean()
        )
        print("Pred distribution:", Counter(comparable["pred_label"]))
        if "dataset" in comparable.columns:
            print("Accuracy by dataset:")
            print(
                comparable.assign(ok=comparable["pred_label"] == comparable["gold_label"])
                .groupby("dataset")["ok"]
                .agg(["count", "mean"])
            )
    return {
        "model": name,
        "n_total": len(df),
        "n_parsed": int(len(comparable)),
        "accuracy": acc,
    }


def load_base(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-file",
        type=str,
        default=str(PROJECT_DIR / "sft_data" / "val.jsonl"),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(PROJECT_DIR / "models" / "Qwen3-1.7B"),
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(PROJECT_DIR / "sft_runs" / "qwen3_1_7b_lora"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_DIR / "results" / "val_base_vs_lora"),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Qwen3-1.7B",
        help="Name prefix, e.g. Qwen3-1.7B / Qwen3-4B / Qwen3-8B",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "base", "lora"],
        default="both",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional subset for debug")
    parser.add_argument(
        "--file-stem",
        type=str,
        default="val",
        help="Output name infix, e.g. val or real_test",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_val(Path(args.val_file))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"Val size={len(rows)} from {args.val_file}")
    print("Gold distribution:", Counter(r["gold_label"] for r in rows))
    print(f"tag={args.tag} mode={args.mode} batch_size={args.batch_size}")

    summaries = []
    base_name = f"{args.tag}-base"
    lora_name = f"{args.tag}-LoRA"

    if args.mode in ("both", "base"):
        print("\nLoading base model...")
        tokenizer, model = load_base(args.base_model)
        base_df = run_model(
            base_name, model, tokenizer, rows, args.log_every, args.batch_size
        )
        base_path = out_dir / f"{args.tag}_{args.file_stem}_base_results.csv"
        base_df.to_csv(base_path, index=False)
        summaries.append(summarize(base_df, base_name))
        unload(model)

    if args.mode in ("both", "lora"):
        print("\nLoading LoRA model...")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(base_model, args.adapter)
        lora_model.eval()
        lora_df = run_model(
            lora_name, lora_model, tokenizer, rows, args.log_every, args.batch_size
        )
        lora_path = out_dir / f"{args.tag}_{args.file_stem}_lora_results.csv"
        lora_df.to_csv(lora_path, index=False)
        summaries.append(summarize(lora_df, lora_name))
        unload(lora_model)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / f"{args.tag}_{args.file_stem}_accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n===== Summary =====")
    print(summary_df.to_string(index=False))
    print(f"Saved details to {out_dir}")


if __name__ == "__main__":
    main()
