"""Generate text or token IDs from a Phi-Plasma checkpoint.

Examples:
    PYTHONPATH=src python scripts/generate.py \
        --ckpt logs/a100_3gpu_plasma_byte_infer/ckpt_final.pt \
        --prompt "The meaning of life is" --device cuda --max-new-tokens 200

For byte-tokenized checkpoints, text prompts are UTF-8 bytes and output is
UTF-8 decoded. For legacy token-ID checkpoints, pass whitespace-separated token
IDs as the prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi_plasma.dashboard import CheckpointRuntime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path relative to repo root or absolute")
    ap.add_argument("--prompt", required=True, help="text prompt or whitespace-separated token IDs")
    ap.add_argument("--device", default="auto", help="cpu, cuda, auto, mps, or explicit torch device")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    runtime = CheckpointRuntime(ROOT, args.device)
    ckpt = Path(args.ckpt)
    rel = ckpt.resolve().relative_to(ROOT).as_posix() if ckpt.is_absolute() else ckpt.as_posix()
    result = runtime.generate(
        rel_path=rel,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(result["text"])
    print()
    print(f"[codec] {result['codec']}  [device] {result['device']}")
    print("[tokens] " + " ".join(str(t) for t in result["generated_tokens"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
