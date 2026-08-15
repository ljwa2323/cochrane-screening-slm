"""
Build a held-out test jsonl from a Chan CSV + gpt-oss reason progress file.

Output format matches sft_data/test.jsonl:
  {"messages": [...], "row_id": int, "label": "include"|"exclude"|"uncertain"}
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """You are an expert systematic reviewer performing title and abstract screening.
Given the review Selection_criteria, the study Title, and the Abstract, decide whether the study should be included.

Labels:
- include: clearly meets selection criteria
- exclude: clearly does not meet selection criteria
- uncertain: insufficient information to decide

Respond with ONLY a JSON object in this exact format:
{"label": "include" | "exclude" | "uncertain", "reason": "<brief explanation>"}
Do not output any other text."""

LABEL_TEXT = {
    0.0: "exclude",
    0.5: "uncertain",
    1.0: "include",
}


def label_to_text(label: float) -> str:
    key = float(label)
    if key not in LABEL_TEXT:
        raise ValueError(f"Unexpected label: {label}")
    return LABEL_TEXT[key]


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_reasons(progress_path: Path) -> tuple[dict[int, str], int, int]:
    reasons: dict[int, str] = {}
    n_fail = 0
    n_lines = 0
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_fail += 1
                continue
            if obj.get("ok") and obj.get("reason"):
                reasons[int(obj["idx"])] = str(obj["reason"]).strip()
            else:
                n_fail += 1
    return reasons, n_lines, n_fail


def build_user_content(criteria: str, title: str, abstract: str) -> str:
    return (
        f"Selection_criteria:\n{criteria}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Decide the screening label and provide a brief reason."
    )


def build_assistant_content(label: float, reason: str) -> str:
    return json.dumps(
        {"label": label_to_text(label), "reason": reason},
        ensure_ascii=False,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build heldout_reviews*.jsonl from CSV + reason progress"
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--progress", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest path; default is <out-jsonl>_manifest.json",
    )
    parser.add_argument(
        "--keep-missing-text",
        action="store_true",
        help="Keep rows with empty Selection_criteria/Title/Abstract_clean",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    progress_path = Path(args.progress)
    out_jsonl = Path(args.out_jsonl)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else out_jsonl.with_name(out_jsonl.stem + "_manifest.json")
    )

    print(f"Loading reasons from {progress_path}")
    reasons, n_lines, n_fail = load_reasons(progress_path)
    print(f"progress lines={n_lines} fail_or_empty={n_fail} unique_ok={len(reasons)}")

    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV rows={len(df)}")

    message_rows: list[dict] = []
    skipped_missing = 0
    skipped_no_reason = 0
    skipped_oob = 0
    label_counts: Counter[str] = Counter()

    for idx, reason in sorted(reasons.items()):
        if idx < 0 or idx >= len(df):
            skipped_oob += 1
            continue
        row = df.iloc[idx]
        criteria = safe_text(row.get("Selection_criteria"))
        title = safe_text(row.get("Title"))
        abstract = safe_text(row.get("Abstract_clean"))
        if not args.keep_missing_text and (not criteria or not title or not abstract):
            skipped_missing += 1
            continue
        label_num = float(row["label"])
        label_text = label_to_text(label_num)
        unnamed = row.get("Unnamed: 0")
        row_id = int(unnamed) if unnamed is not None and not pd.isna(unnamed) else int(idx)
        message_rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_content(criteria, title, abstract),
                    },
                    {
                        "role": "assistant",
                        "content": build_assistant_content(label_num, reason),
                    },
                ],
                "row_id": row_id,
                "label": label_text,
            }
        )
        label_counts[label_text] += 1

    # Count rows present in CSV but missing a usable reason.
    for i in range(len(df)):
        if i not in reasons:
            skipped_no_reason += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, message_rows)
    print(
        f"Wrote {out_jsonl} n={len(message_rows)} "
        f"skipped_missing={skipped_missing} skipped_no_reason={skipped_no_reason} "
        f"skipped_oob={skipped_oob}"
    )

    manifest = {
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_csv": str(csv_path),
        "source_progress": str(progress_path),
        "progress_lines": n_lines,
        "progress_fail_or_empty": n_fail,
        "unique_ok_reasons": len(reasons),
        "matched_usable": len(message_rows),
        "skipped_missing_text": skipped_missing,
        "skipped_no_reason": skipped_no_reason,
        "skipped_oob": skipped_oob,
        "test_n": len(message_rows),
        "label_mapping": {str(k): v for k, v in LABEL_TEXT.items()},
        "label_counts": {k: int(v) for k, v in sorted(label_counts.items())},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
