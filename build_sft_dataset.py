"""
Build matched SFT dataset from reason_gen/progress.jsonl + dev_set.csv.

Outputs under sft_data/:
  - screening_sft_snapshot.csv / .jsonl  (necessary columns only)
  - screening_sft_messages.jsonl        (full chat records)
  - train.jsonl / val.jsonl             (stratified 90/10)
  - manifest.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "data" / "20240827_dev_set.csv"
DEFAULT_PROGRESS = PROJECT_DIR / "reason_gen" / "progress.jsonl"
DEFAULT_OUT_DIR = PROJECT_DIR / "sft_data"

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


def load_reasons(progress_path: Path) -> dict[int, str]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--progress", type=str, default=str(DEFAULT_PROGRESS))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-missing-text",
        action="store_true",
        help="Keep rows with empty Selection_criteria/Title/Abstract_clean",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reasons from {args.progress}")
    reasons, n_lines, n_fail = load_reasons(Path(args.progress))
    print(f"progress lines={n_lines} fail_or_empty={n_fail} unique_ok={len(reasons)}")

    print(f"Loading CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"CSV rows={len(df)}")

    records = []
    skipped_missing = 0
    skipped_oob = 0
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
        records.append(
            {
                "row_id": row_id,
                "idx": int(idx),
                "Selection_criteria": criteria,
                "Title": title,
                "Abstract_clean": abstract,
                "label": label_text,
                "label_numeric": label_num,
                "reason": reason,
            }
        )

    print(
        f"Matched usable={len(records)} "
        f"skipped_missing={skipped_missing} skipped_oob={skipped_oob}"
    )
    if not records:
        raise SystemExit("No usable records.")

    snap_df = pd.DataFrame(records)[
        [
            "row_id",
            "Selection_criteria",
            "Title",
            "Abstract_clean",
            "label",
            "reason",
        ]
    ]
    snap_csv = out_dir / "screening_sft_snapshot.csv"
    snap_jsonl = out_dir / "screening_sft_snapshot.jsonl"
    snap_df.to_csv(snap_csv, index=False)
    write_jsonl(snap_jsonl, snap_df.to_dict(orient="records"))
    print(f"Wrote {snap_csv}")
    print(f"Wrote {snap_jsonl}")

    message_rows = []
    for rec in records:
        assistant = build_assistant_content(rec["label_numeric"], rec["reason"])
        message_rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_content(
                            rec["Selection_criteria"],
                            rec["Title"],
                            rec["Abstract_clean"],
                        ),
                    },
                    {"role": "assistant", "content": assistant},
                ],
                "row_id": rec["row_id"],
                "label": rec["label"],
            }
        )

    all_msg_path = out_dir / "screening_sft_messages.jsonl"
    write_jsonl(all_msg_path, message_rows)
    print(f"Wrote {all_msg_path}")

    labels = [r["label"] for r in message_rows]
    train_rows, val_rows = train_test_split(
        message_rows,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    print(f"Wrote {train_path} n={len(train_rows)}")
    print(f"Wrote {val_path} n={len(val_rows)}")

    label_counts = Counter(snap_df["label"].tolist())
    manifest = {
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_csv": str(args.csv),
        "source_progress": str(args.progress),
        "progress_lines": n_lines,
        "progress_fail_or_empty": n_fail,
        "unique_ok_reasons": len(reasons),
        "matched_usable": len(records),
        "skipped_missing_text": skipped_missing,
        "skipped_oob": skipped_oob,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "label_mapping": {str(k): v for k, v in LABEL_TEXT.items()},
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "columns": [
            "row_id",
            "Selection_criteria",
            "Title",
            "Abstract_clean",
            "label",
            "reason",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
