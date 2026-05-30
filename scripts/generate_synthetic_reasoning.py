"""Generate local reasoning traces with an Ollama teacher model.

This is the distillation entry point for a 35B-teacher demonstration. It asks a
local Qwen/Ollama teacher to solve varied tasks with compact reasoning and writes
JSONL records that can be tokenized by ``scripts/build_token_shards.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = """You are generating high-quality supervised fine-tuning data for a small research language model. Produce clear reasoning, but keep it concise. End with a line starting with 'Final:' that gives the answer or deliverable."""

BUILTIN_SEEDS = [
    "Solve step by step: A train travels 84 km in 1.5 hours, then 126 km in 2 hours. What is its average speed?",
    "A function f satisfies f(x)=3x+2. If g(x)=f(f(x))-x, simplify g(x).",
    "Write a Python function that returns the first non-repeated character in a string. Explain edge cases briefly.",
    "A warehouse has red, blue, and green boxes in ratio 3:5:7. If there are 210 boxes total, how many are blue?",
    "Find the flaw in this argument: All cats are animals. Some animals are dogs. Therefore some cats are dogs.",
    "Given the requirements, design a small API for tracking experiment runs and checkpoint metadata.",
    "Explain why validation loss can improve while training loss appears flat near the end of a run.",
    "A rectangle has perimeter 50 and length 15. What is its area? Show the calculation.",
    "Implement binary search and explain why the loop terminates.",
    "Summarize the tradeoff between model size, token count, and data quality for language model training.",
]

TASK_MUTATORS = [
    "Make the reasoning explicit but concise.",
    "Include a sanity check of the answer.",
    "Prefer a robust method over a shortcut.",
    "If there is ambiguity, state the assumption and proceed.",
    "Give the final answer in one sentence after the reasoning.",
]


def read_seed_prompts(path: str | None) -> list[str]:
    if not path:
        return list(BUILTIN_SEEDS)
    prompts = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if p.suffix == ".jsonl":
                obj = json.loads(line)
                prompts.append(str(obj.get("prompt") or obj.get("text") or obj))
            else:
                prompts.append(line.strip())
    return prompts


def make_prompt(seed: str, rng: random.Random) -> str:
    mutator = rng.choice(TASK_MUTATORS)
    return f"{seed}\n\n{mutator}"


def ollama_chat(host: str, model: str, prompt: str, temperature: float,
                num_predict: int, timeout: int, think: bool = False) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "think": think,  # Qwen3 / DeepSeek-R1 / other reasoning models default to thinking, which
                          # routes the answer into `message.thinking` and leaves `content` empty.
                          # Setting think=False makes the model emit the answer directly.
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        obj = json.loads(response.read().decode("utf-8"))
    msg = obj.get("message", {}) or {}
    content = msg.get("content", "") or ""
    if not content and msg.get("thinking"):
        # Fallback: some servers ignore think=False on older models; salvage the answer
        # so we don't write empty records. Strip everything before the final answer.
        thinking = msg.get("thinking", "")
        content = _strip_thinking_prefix(thinking)
    return content


def _strip_thinking_prefix(text: str) -> str:
    # Models often emit "...long monologue... Final: <answer>" inside their thinking trace.
    marker = "Final:"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx:].strip()
    return text.strip()


def build_record(i: int, seed: str, model: str, host: str, temperature: float,
                 num_predict: int, timeout: int, base_seed: int,
                 think: bool = False) -> dict[str, Any]:
    rng = random.Random(base_seed + i)
    prompt = make_prompt(seed, rng)
    response = ollama_chat(host, model, prompt, temperature, num_predict, timeout, think=think)
    if not response.strip():
        raise RuntimeError("empty response (model may be in thinking mode; pass --think to keep traces)")
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
    return {
        "id": i,
        "teacher": model,
        "prompt": prompt,
        "response": response,
        "text": text,
        "created_at": time.time(),
    }


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def check_ollama(host: str, model: str) -> tuple[bool, str]:
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            obj = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return False, f"unreachable at {host}: {exc}"
    names = [m.get("name", "") for m in obj.get("models", [])]
    if model in names:
        return True, f"reachable; model '{model}' present"
    base = model.split(":", 1)[0]
    near = [n for n in names if n.startswith(base)]
    suffix = f"; nearest: {', '.join(near[:3])}" if near else ""
    return True, f"reachable; model '{model}' NOT in local tags ({len(names)} models){suffix}"


def write_progress(progress_path: Path, payload: dict) -> None:
    try:
        progress_path.write_text(json.dumps(payload) + "\n")
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6:35b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--seed-prompts", default=None, help="txt or jsonl prompt seed file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num-predict", type=int, default=768)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat-every", type=int, default=10,
                    help="print throughput summary every N completions")
    ap.add_argument("--strict-model-check", action="store_true",
                    help="abort if the requested model is not present in the local Ollama tags")
    ap.add_argument("--think", action="store_true",
                    help="keep the model's <think> traces in the output (default off — most teachers waste tokens in thinking mode and return empty content)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = read_seed_prompts(args.seed_prompts)
    if not seeds:
        raise SystemExit("no seed prompts available")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out.with_suffix(out.suffix + ".progress.json")

    pid = os.getpid()
    print(f"=== generate_synthetic_reasoning (pid {pid}) ===", flush=True)
    print(f"model    : {args.model}", flush=True)
    print(f"host     : {args.host}", flush=True)
    print(f"seeds    : {len(seeds)} prompts (source={args.seed_prompts or 'BUILTIN'})", flush=True)
    print(f"count    : {args.count}", flush=True)
    print(f"workers  : {args.workers}", flush=True)
    print(f"temp     : {args.temperature}  num_predict: {args.num_predict}  timeout: {args.timeout}s", flush=True)
    print(f"out      : {out}", flush=True)
    print(f"progress : {progress_path}", flush=True)

    ok, note = check_ollama(args.host, args.model)
    print(f"ollama   : {note}", flush=True)
    if not ok:
        print("HINT: start the daemon with `ollama serve` or pass --host", flush=True)
        return 2
    if args.strict_model_check and "NOT in local tags" in note:
        print(f"HINT: pull the model with `ollama pull {args.model}`", flush=True)
        return 3

    jobs = [(i, seeds[i % len(seeds)]) for i in range(args.count)]
    if args.dry_run:
        for i, seed in jobs[: min(5, len(jobs))]:
            print(make_prompt(seed, random.Random(args.seed + i)))
            print("---")
        return 0

    n_total = args.count
    n_ok = 0
    n_fail = 0
    t0 = time.monotonic()
    last_hb_t = t0
    last_hb_count = 0
    tokens_total = 0
    tokens_recent: list[int] = []
    heartbeat_every = max(1, args.heartbeat_every)

    def heartbeat(now: float, force: bool = False) -> None:
        nonlocal last_hb_t, last_hb_count
        done = n_ok + n_fail
        if not force and done - last_hb_count < heartbeat_every:
            return
        elapsed = max(now - t0, 1e-6)
        rate = done / elapsed
        remaining = n_total - done
        eta = remaining / rate if rate > 0 else float("inf")
        avg_tok = (tokens_total / max(n_ok, 1)) if n_ok else 0
        msg = (f"~~ progress: {done}/{n_total} ({100*done/n_total:.1f}%) "
               f"ok={n_ok} fail={n_fail} | {rate:.2f}/s | eta {fmt_eta(eta)} "
               f"| avg_resp_chars={avg_tok:.0f}")
        print(msg, flush=True)
        write_progress(progress_path, {
            "pid": pid, "done": done, "total": n_total, "ok": n_ok, "fail": n_fail,
            "rate_per_s": rate, "eta_s": eta if eta != float("inf") else None,
            "avg_response_chars": avg_tok, "started_at": t0, "updated_at": now,
            "model": args.model, "out": str(out),
        })
        last_hb_t = now
        last_hb_count = done

    print(f"~~ start | t={time.strftime('%H:%M:%S')}", flush=True)
    heartbeat(t0, force=True)

    try:
        with out.open("a", encoding="utf-8") as f:
            if args.workers == 1:
                for i, seed in jobs:
                    t_call = time.monotonic()
                    try:
                        rec = build_record(i, seed, args.model, args.host, args.temperature,
                                           args.num_predict, args.timeout, args.seed, args.think)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        n_ok += 1
                        tokens_total += len(rec.get("response", ""))
                        tokens_recent.append(len(rec.get("response", "")))
                        dt = time.monotonic() - t_call
                        print(f"[{i+1}/{n_total}] ok | {dt:.1f}s | resp_chars={len(rec.get('response',''))}", flush=True)
                    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                        n_fail += 1
                        dt = time.monotonic() - t_call
                        print(f"[{i+1}/{n_total}] FAIL | {dt:.1f}s | {type(exc).__name__}: {exc}", flush=True)
                    heartbeat(time.monotonic())
            else:
                with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    pending = [
                        ex.submit(build_record, i, seed, args.model, args.host, args.temperature,
                                  args.num_predict, args.timeout, args.seed, args.think)
                        for i, seed in jobs
                    ]
                    for j, fut in enumerate(futures.as_completed(pending), start=1):
                        try:
                            rec = fut.result()
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            n_ok += 1
                            tokens_total += len(rec.get("response", ""))
                            print(f"[{j}/{n_total}] ok | resp_chars={len(rec.get('response',''))}", flush=True)
                        except Exception as exc:  # noqa: BLE001 — worker pool boundary
                            n_fail += 1
                            print(f"[{j}/{n_total}] FAIL | {type(exc).__name__}: {exc}", flush=True)
                        heartbeat(time.monotonic())
    except KeyboardInterrupt:
        print("~~ interrupted by user; finalising progress", flush=True)
    finally:
        now = time.monotonic()
        heartbeat(now, force=True)
        elapsed = now - t0
        print(f"=== done | ok={n_ok} fail={n_fail} of {n_total} | {elapsed:.1f}s "
              f"| {n_ok / max(elapsed, 1e-6):.2f} ok/s ===", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
