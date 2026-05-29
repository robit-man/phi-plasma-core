"""Estimate rough training cost for Phi-Plasma scaling runs."""

from __future__ import annotations

import argparse
import math


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True, help="parameter count, e.g. 3e8")
    ap.add_argument("--tokens", type=float, required=True, help="training tokens")
    ap.add_argument("--tokens-per-second", type=float, default=170000,
                    help="measured useful throughput across the box")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--global-batch", type=int, default=12)
    args = ap.parse_args()

    steps = args.tokens / max(1, args.seq_len * args.global_batch)
    seconds = args.tokens / max(1.0, args.tokens_per_second)
    flops = 6.0 * args.params * args.tokens
    print(f"steps: {steps:,.0f}")
    print(f"wall time: {seconds/3600:.2f} hours ({seconds/86400:.2f} days)")
    print(f"training flops: {flops:.3e}")
    print(f"tokens/parameter: {args.tokens / args.params:.2f}")
    print(f"compute-optimal-ish tokens at 20 tokens/param: {20 * args.params:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
