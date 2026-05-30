"""Generate local reasoning traces with an Ollama teacher model.

This is the distillation entry point for a 35B-teacher demonstration. It asks a
local Qwen/Ollama teacher to solve varied tasks with compact reasoning and writes
JSONL records that can be tokenized by ``scripts/build_token_shards.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import random
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
                num_predict: int, timeout: int) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
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
    return obj.get("message", {}).get("content", "")


def build_record(i: int, seed: str, model: str, host: str, temperature: float,
                 num_predict: int, timeout: int, base_seed: int) -> dict[str, Any]:
    rng = random.Random(base_seed + i)
    prompt = make_prompt(seed, rng)
    response = ollama_chat(host, model, prompt, temperature, num_predict, timeout)
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
    return {
        "id": i,
        "teacher": model,
        "prompt": prompt,
        "response": response,
        "text": text,
        "created_at": time.time(),
    }


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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = read_seed_prompts(args.seed_prompts)
    if not seeds:
        raise SystemExit("no seed prompts available")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(i, seeds[i % len(seeds)]) for i in range(args.count)]
    if args.dry_run:
        for i, seed in jobs[: min(5, len(jobs))]:
            print(make_prompt(seed, random.Random(args.seed + i)))
            print("---")
        return 0

    with out.open("a", encoding="utf-8") as f:
        if args.workers == 1:
            for i, seed in jobs:
                try:
                    rec = build_record(i, seed, args.model, args.host, args.temperature,
                                       args.num_predict, args.timeout, args.seed)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{i+1}/{args.count}] ok", flush=True)
                except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                    print(f"[{i+1}/{args.count}] failed: {exc}", flush=True)
        else:
            with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                pending = [
                    ex.submit(build_record, i, seed, args.model, args.host, args.temperature,
                              args.num_predict, args.timeout, args.seed)
                    for i, seed in jobs
                ]
                for j, fut in enumerate(futures.as_completed(pending), start=1):
                    try:
                        rec = fut.result()
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"[{j}/{args.count}] ok", flush=True)
                    except Exception as exc:
                        print(f"[{j}/{args.count}] failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
