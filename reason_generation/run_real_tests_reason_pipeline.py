"""
Generate gpt-oss-120b reasons for random/HIV/heart test CSVs, then build
sft_data/real_test{1,2,3}.jsonl in the same format as test.jsonl.

Runs the three sets sequentially so API rate limits stay stable.
Resume-safe: reason generation skips idxs already present in progress.jsonl.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
GEN_SCRIPT = Path(__file__).resolve().parent / "generate_reasons_gpt_oss.py"
BUILD_SCRIPT = Path(__file__).resolve().parent / "build_real_test_jsonl.py"
SFT_DIR = PROJECT_DIR / "sft_data"

TASKS = [
    {
        "name": "real_test1",
        "csv": DATA_DIR / "20240827_random_test_set.csv",
        "reason_dir": PROJECT_DIR / "reason_gen_real_test1",
        "out_jsonl": SFT_DIR / "real_test1.jsonl",
    },
    {
        "name": "real_test2",
        "csv": DATA_DIR / "20240827_HIV_test_set.csv",
        "reason_dir": PROJECT_DIR / "reason_gen_real_test2",
        "out_jsonl": SFT_DIR / "real_test2.jsonl",
    },
    {
        "name": "real_test3",
        "csv": DATA_DIR / "20240827_heart_test_set.csv",
        "reason_dir": PROJECT_DIR / "reason_gen_real_test3",
        "out_jsonl": SFT_DIR / "real_test3.jsonl",
    },
]


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"CMD: {' '.join(cmd)}")
    print(f"LOG: {log_path}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        logf.write("CMD: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def count_ok_reasons(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    n = 0
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '"ok": true' in line or '"ok":true' in line:
                n += 1
    return n


def process_task(task: dict, args: argparse.Namespace) -> None:
    name = task["name"]
    csv_path: Path = task["csv"]
    reason_dir: Path = task["reason_dir"]
    out_jsonl: Path = task["out_jsonl"]
    reason_dir.mkdir(parents=True, exist_ok=True)
    log_dir = reason_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print(f"[{name}] csv={csv_path}")
    print(f"[{name}] reason_dir={reason_dir}")
    print(f"[{name}] out_jsonl={out_jsonl}")

    if not args.skip_generate:
        # First pass
        run_cmd(
            [
                sys.executable,
                str(GEN_SCRIPT),
                "--input",
                str(csv_path),
                "--out-dir",
                str(reason_dir),
                "--api-key-file",
                str(PROJECT_DIR / "api_key.txt"),
                "--rps",
                str(args.rps),
                "--concurrency",
                str(args.concurrency),
                "--max-retries",
                str(args.max_retries),
                "--log-every",
                str(args.log_every),
            ],
            log_dir / f"reason_gen_{ts}.log",
        )
        # Second pass to retry failures / incomplete rows
        run_cmd(
            [
                sys.executable,
                str(GEN_SCRIPT),
                "--input",
                str(csv_path),
                "--out-dir",
                str(reason_dir),
                "--api-key-file",
                str(PROJECT_DIR / "api_key.txt"),
                "--rps",
                str(args.rps),
                "--concurrency",
                str(args.concurrency),
                "--max-retries",
                str(args.max_retries),
                "--log-every",
                str(args.log_every),
            ],
            log_dir / f"reason_gen_retry_{ts}.log",
        )

    ok_n = count_ok_reasons(reason_dir / "progress.jsonl")
    print(f"[{name}] ok reasons in progress: {ok_n}")

    if not args.skip_build:
        run_cmd(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--csv",
                str(csv_path),
                "--progress",
                str(reason_dir / "progress.jsonl"),
                "--out-jsonl",
                str(out_jsonl),
            ],
            log_dir / f"build_jsonl_{ts}.log",
        )
        print(f"[{name}] done -> {out_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reasons and build real_test1/2/3.jsonl"
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset: real_test1,real_test2,real_test3",
    )
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    selected = {t.strip() for t in args.only.split(",") if t.strip()}
    tasks = [t for t in TASKS if not selected or t["name"] in selected]
    if not tasks:
        raise SystemExit("No tasks selected.")

    t0 = time.time()
    for task in tasks:
        process_task(task, args)
    print("=" * 72)
    print(f"All selected tasks finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
