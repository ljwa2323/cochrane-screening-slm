"""
Test Qwen3 models (1.7B / 4B / 8B / 14B) on the fixed 90-sample screening set.
Uses the same stratified sample as the previous Qwen2.5-7B run.
"""

import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "qwen2p5_7b_screening_90_sample.csv"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

MODEL_NAMES = [
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
]

MAX_NEW_TOKENS = 256
VALID_LABELS = {0.0, 0.5, 1.0}

SYSTEM_PROMPT = """You are an expert systematic reviewer performing title and abstract screening.
Given the review Selection_criteria, the study Title, and the Abstract, decide whether the study should be included.

Labels:
- 1.0: Include (clearly meets selection criteria)
- 0.0: Exclude (clearly does not meet selection criteria)
- 0.5: Uncertain (insufficient information to decide)

Respond with ONLY a JSON object in this exact format:
{"label": <0.0 or 0.5 or 1.0>, "reason": "<brief explanation>"}
Do not output any other text."""


def build_user_prompt(row: pd.Series) -> str:
    criteria = row.get("Selection_criteria")
    title = row.get("Title")
    abstract = row.get("Abstract_clean")
    if pd.isna(criteria):
        criteria = ""
    if pd.isna(title):
        title = ""
    if pd.isna(abstract):
        abstract = ""
    return (
        f"Selection_criteria:\n{criteria}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Decide the screening label and provide a brief reason."
    )


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_response(text: str) -> tuple[float | None, str]:
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
            label = float(obj.get("label"))
            reason = str(obj.get("reason", "")).strip()
            return label, reason
        except Exception:
            continue

    label_match = re.search(r"(?:label|Label)\s*[:=]\s*(0\.5|0\.0|1\.0|0|1|0\.50)", text)
    reason_match = re.search(r"(?:reason|Reason)\s*[:=]\s*[\"']?(.*)", text, re.DOTALL)
    pred_label = None
    if label_match:
        pred_label = float(label_match.group(1))
    reason = reason_match.group(1).strip().strip('"').strip("'") if reason_match else text
    return pred_label, reason


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
    gen_ids = output_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def summarize_results(result_df: pd.DataFrame, model_name: str) -> dict:
    comparable = result_df.dropna(subset=["pred_label"])
    summary = {
        "model": model_name,
        "n_total": len(result_df),
        "n_parsed": int(len(comparable)),
        "accuracy": None,
    }
    print(f"\n===== {model_name} =====")
    if len(comparable):
        acc = float((comparable["pred_label"] == comparable["gold_label"]).mean())
        summary["accuracy"] = acc
        print(f"Parsed predictions: {len(comparable)}/{len(result_df)}")
        print(f"Exact-match accuracy: {acc:.3f}")
        print("Confusion (gold x pred):")
        print(
            pd.crosstab(
                comparable["gold_label"],
                comparable["pred_label"],
                rownames=["gold"],
                colnames=["pred"],
            )
        )
        print("Accuracy by gold label:")
        print(
            comparable.assign(ok=comparable["pred_label"] == comparable["gold_label"])
            .groupby("gold_label")["ok"]
            .mean()
        )
    else:
        print("No successfully parsed predictions.")
    return summary


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_model(model_name: str, sample: pd.DataFrame) -> dict:
    model_path = MODELS_DIR / model_name
    output_path = RESULTS_DIR / f"{model_name}_screening_90_results.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model: {model_name} from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = []
    t0 = time.time()
    for i, row in sample.iterrows():
        user_prompt = build_user_prompt(row)
        try:
            raw = generate(model, tokenizer, user_prompt)
            pred_label, reason = parse_response(raw)
            error = ""
        except Exception as exc:
            raw = ""
            pred_label, reason = None, ""
            error = str(exc)

        records.append(
            {
                "model": model_name,
                "row_id": int(row["Unnamed: 0"])
                if "Unnamed: 0" in row and not pd.isna(row["Unnamed: 0"])
                else i,
                "gold_label": float(row["label"]),
                "pred_label": pred_label,
                "reason": reason,
                "raw_output": raw,
                "error": error,
                "Selection_criteria": row["Selection_criteria"],
                "Title": row["Title"],
                "Abstract_clean": row["Abstract_clean"],
                "Review_Title": row.get("Review_Title", ""),
            }
        )

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"[{model_name}] [{i + 1}/{len(sample)}] "
                f"elapsed={elapsed:.1f}s pred={pred_label} gold={row['label']}"
            )

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    summary = summarize_results(result_df, model_name)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["output_path"] = str(output_path)

    unload_model(model, tokenizer)
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading fixed sample from {SAMPLE_PATH}")
    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Sample size: {len(sample)}")
    print("Sample label counts:\n", sample["label"].value_counts().sort_index())

    summaries = []
    for model_name in MODEL_NAMES:
        summary = run_one_model(model_name, sample)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "qwen3_screening_90_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n===== Overall summary =====")
    print(summary_df.to_string(index=False))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
