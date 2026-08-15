"""Stratified 10% sample from val.jsonl by 3-class label."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC = PROJECT_DIR / "sft_data" / "val.jsonl"
OUT = PROJECT_DIR / "sft_data" / "val_strat10.jsonl"
MANIFEST = PROJECT_DIR / "sft_data" / "val_strat10_manifest.json"
SEED = 42
FRAC = 0.10


def main() -> None:
    rows = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get("label")
            if label not in {"include", "exclude", "uncertain"}:
                try:
                    label = json.loads(obj["messages"][-1]["content"]).get("label")
                except Exception:
                    label = None
            obj["_label"] = label
            rows.append(obj)

    df = pd.DataFrame(rows)
    before = Counter(df["_label"])
    sampled = (
        df.groupby("_label", group_keys=False)
        .apply(lambda g: g.sample(frac=FRAC, random_state=SEED))
        .reset_index(drop=True)
    )
    after = Counter(sampled["_label"])

    with OUT.open("w", encoding="utf-8") as f:
        for obj in sampled.to_dict(orient="records"):
            obj.pop("_label", None)
            # pandas may convert ints; keep JSON-serializable
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(SRC),
        "output": str(OUT),
        "seed": SEED,
        "frac": FRAC,
        "n_source": int(len(df)),
        "n_sample": int(len(sampled)),
        "source_label_counts": dict(before),
        "sample_label_counts": dict(after),
        "source_label_frac": {k: v / len(df) for k, v in before.items()},
        "sample_label_frac": {k: v / len(sampled) for k, v in after.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
