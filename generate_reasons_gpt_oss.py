"""
Generate screening reasons with NVIDIA openai/gpt-oss-120b API.

Input columns: Selection_criteria, Title, Abstract_clean, label
Output: reason column (for later Qwen3 SFT)

Features:
- Process rows in ascending index order
- Async concurrency with RPS rate limit (default 40)
- Checkpoint / resume via append-only JSONL progress file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "20240827_dev_set.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "reason_gen"
DEFAULT_API_KEY_FILE = PROJECT_DIR / "api_key.txt"

MODEL_ID = "openai/gpt-oss-120b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

SYSTEM_PROMPT = """You are an expert systematic reviewer.
Given Selection_criteria, Title, Abstract, and the gold screening label, write a brief justification for that label.

Label meanings:
- 1.0: Include (clearly meets selection criteria)
- 0.0: Exclude (clearly does not meet selection criteria)
- 0.5: Uncertain (insufficient information to decide)

Respond with ONLY a JSON object:
{"reason": "<1-3 sentence explanation justifying the given label>"}
Do not change the label. Do not output any other text."""


class RateLimiter:
    """Token-bucket limiter for max `rate` acquires per second."""

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self.tokens = 0.0  # no burst on startup
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.updated = now
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait)

    async def penalize(self, seconds: float = 0.0) -> None:
        """Drain tokens after a 429; caller should also sleep."""
        async with self._lock:
            self.tokens = 0.0
            self.updated = time.monotonic()


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def build_user_prompt(criteria: str, title: str, abstract: str, label: float) -> str:
    return (
        f"Selection_criteria:\n{criteria}\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Gold label: {label}\n\n"
        "Write the reason that justifies this gold label."
    )


def parse_reason(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            reason = str(obj.get("reason", "")).strip()
            if reason:
                return reason
        except Exception:
            pass
    # Fallback: use raw content
    return text


def load_done_indices(progress_path: Path) -> set[int]:
    done: set[int] = set()
    if not progress_path.exists():
        return done
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("ok") and "idx" in obj and obj.get("reason"):
                    done.add(int(obj["idx"]))
            except Exception:
                continue
    return done


async def call_one(
    client: AsyncOpenAI,
    limiter: RateLimiter,
    sem: asyncio.Semaphore,
    idx: int,
    criteria: str,
    title: str,
    abstract: str,
    label: float,
    max_tokens: int,
    max_retries: int,
) -> dict:
    user_prompt = build_user_prompt(criteria, title, abstract, label)
    last_err = ""
    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                await limiter.acquire()
                resp = await client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    extra_body={"reasoning_effort": "low"},
                )
                content = resp.choices[0].message.content or ""
                reason = parse_reason(content)
                if not reason:
                    raise ValueError("empty reason")
                return {
                    "idx": idx,
                    "ok": True,
                    "reason": reason,
                    "error": "",
                    "attempts": attempt,
                    "ts": time.time(),
                }
            except Exception as exc:
                last_err = str(exc)
                err_l = last_err.lower()
                if "429" in err_l or "too many requests" in err_l:
                    backoff = min(10 * attempt, 60)
                    await limiter.penalize(backoff)
                    await asyncio.sleep(backoff)
                else:
                    await asyncio.sleep(min(2 ** attempt, 30))
    return {
        "idx": idx,
        "ok": False,
        "reason": "",
        "error": last_err,
        "attempts": max_retries,
        "ts": time.time(),
    }


async def writer_loop(
    queue: asyncio.Queue,
    progress_path: Path,
    stats: dict,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            stats["written"] += 1
            if item.get("ok"):
                stats["ok"] += 1
            else:
                stats["fail"] += 1
            queue.task_done()


async def run_generation(args: argparse.Namespace) -> None:
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    meta_path = out_dir / "run_meta.json"

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)
    n_total = len(df)
    start = max(0, args.start)
    end = n_total if args.end is None else min(args.end, n_total)
    print(f"Row range: [{start}, {end}) of {n_total}")

    done = load_done_indices(progress_path)
    pending = [i for i in range(start, end) if i not in done]
    print(f"Already done in progress: {len(done)}")
    print(f"Pending in range: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    meta = {
        "input": str(args.input),
        "model": MODEL_ID,
        "start": start,
        "end": end,
        "rps": args.rps,
        "concurrency": args.concurrency,
        "started_at": time.time(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key, timeout=180.0)
    limiter = RateLimiter(args.rps)
    sem = asyncio.Semaphore(args.concurrency)
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    work_queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    stats = {"written": 0, "ok": 0, "fail": 0, "completed": 0}
    writer_task = asyncio.create_task(writer_loop(result_queue, progress_path, stats))

    t0 = time.time()
    total_pending = len(pending)

    async def worker() -> None:
        while True:
            idx = await work_queue.get()
            if idx is None:
                work_queue.task_done()
                break
            row = df.iloc[idx]
            result = await call_one(
                client=client,
                limiter=limiter,
                sem=sem,
                idx=idx,
                criteria=safe_text(row.get("Selection_criteria")),
                title=safe_text(row.get("Title")),
                abstract=safe_text(row.get("Abstract_clean")),
                label=float(row.get("label")),
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
            await result_queue.put(result)
            stats["completed"] += 1
            completed = stats["completed"]
            if completed % args.log_every == 0 or completed == total_pending:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] completed={completed}/{total_pending} "
                    f"ok={stats['ok']} fail={stats['fail']} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} rows/s "
                    f"last_idx={idx}"
                )
            work_queue.task_done()

    # Fixed worker pool keeps memory bounded; indices are enqueued in order.
    n_workers = args.concurrency
    workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
    for idx in pending:
        await work_queue.put(idx)
    for _ in range(n_workers):
        await work_queue.put(None)
    await asyncio.gather(*workers)
    await result_queue.put(None)
    await writer_task
    await client.close()

    elapsed = time.time() - t0
    print(
        f"Finished pending batch. completed={stats['completed']} "
        f"ok={stats['ok']} fail={stats['fail']} elapsed={elapsed:.1f}s"
    )


def merge_to_csv(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    progress_path = out_dir / "progress.jsonl"
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "dev_set_with_reason.csv"

    print(f"Loading reasons from {progress_path}")
    reasons: dict[int, str] = {}
    n_fail = 0
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["idx"])
            if obj.get("ok") and obj.get("reason"):
                reasons[idx] = obj["reason"]
            else:
                n_fail += 1
    print(f"Loaded ok reasons: {len(reasons)}; failed records seen: {n_fail}")

    print(f"Loading input CSV: {args.input}")
    df = pd.read_csv(args.input)
    df["reason"] = [reasons.get(i, "") for i in range(len(df))]
    n_empty = int((df["reason"] == "").sum())
    print(f"Empty reason rows: {n_empty}/{len(df)}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate reasons with gpt-oss-120b")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--api-key-file", type=str, default=str(DEFAULT_API_KEY_FILE))
    p.add_argument("--start", type=int, default=0, help="Inclusive start index")
    p.add_argument("--end", type=int, default=None, help="Exclusive end index")
    p.add_argument("--rps", type=float, default=40.0, help="Max request send rate")
    p.add_argument(
        "--concurrency",
        type=int,
        default=40,
        help="Worker / in-flight request pool size",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge progress.jsonl into CSV with reason column",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Final CSV path when merging",
    )
    return p


def main() -> None:
    # Ensure progress logs appear immediately when stdout is redirected.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = build_parser().parse_args()
    if args.merge_only:
        merge_to_csv(args)
        return
    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
