"""Read the final metrics from an MVP run and report whether the four
'success diagnostics' show up:

  1. Loss-curve plateau structure (step-wise drops)
  2. Eigensheaf penalty actively constraining heads
  3. Head isotypic distance staying low (<0.2)
  4. Training completion without divergence

Usage:
    scripts/py scripts/read_results.py [--run logs/mvp_18m]
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def load_log(log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def detect_plateaus(nll_series: list[float], window: int = 50,
                    drop_threshold: float = 0.15) -> list[int]:
    """Detect step indices where the trailing-window median drops by ≥drop_threshold."""
    medians = []
    drops = []
    for i in range(window, len(nll_series)):
        prev = sorted(nll_series[i-window:i])[window//2]
        curr = sorted(nll_series[max(0, i-window//2):i+1])[len(nll_series[max(0, i-window//2):i+1])//2]
        if prev - curr > drop_threshold:
            drops.append(i)
    return drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/mvp_18m")
    args = ap.parse_args()

    run_dir = Path(args.run)
    log_path = run_dir / "metrics.jsonl"
    if not log_path.exists():
        print(f"[error] no log at {log_path}")
        return 1

    rows = load_log(log_path)
    train_rows = [r for r in rows if "nll" in r]
    eval_rows = [r for r in rows if "val_ppl" in r]

    if not train_rows:
        print(f"[error] no train rows in {log_path}")
        return 1

    print(f"=== run: {args.run} ===")
    print(f"steps: {train_rows[-1]['step']}  ({len(train_rows)} log entries)")
    print(f"elapsed: {train_rows[-1].get('elapsed_s', 0):.0f}s")
    print()

    nll = [r["nll"] for r in train_rows]
    iso = [r.get("head_iso_mean", float("nan")) for r in train_rows if "head_iso_mean" in r]
    eig = [r.get("eig", float("nan")) for r in train_rows]

    print(f"-- NLL --")
    print(f"  initial: {nll[0]:.4f}  ({math.exp(nll[0]):.1f} PPL)")
    print(f"  final:   {nll[-1]:.4f}  ({math.exp(nll[-1]):.1f} PPL)")
    print(f"  min:     {min(nll):.4f}  ({math.exp(min(nll)):.1f} PPL)")

    if eval_rows:
        ppls = [r["val_ppl"] for r in eval_rows]
        print()
        print(f"-- Val PPL --")
        for r in eval_rows[-5:]:
            print(f"  step {r['step']:>7d}: {r['val_ppl']:.2f}")

    print()
    print(f"-- Head isotypic distance --")
    if iso:
        print(f"  initial: {iso[0]:.4f}")
        print(f"  final:   {iso[-1]:.4f}")
        print(f"  max:     {max(iso):.4f}  (target: <0.2)")
        print(f"  VERDICT: {'PASS' if max(iso) < 0.2 else 'FAIL'} — no mode collapse")

    print()
    print(f"-- Eigensheaf penalty --")
    print(f"  initial: {eig[0]:.4e}")
    print(f"  final:   {eig[-1]:.4e}")

    print()
    print(f"-- Plateau detection (NLL drops >0.15 over 50-step window) --")
    plateaus = detect_plateaus(nll)
    if plateaus:
        step_pairs = [(p, train_rows[p]["step"]) for p in plateaus]
        unique_steps = sorted(set(s for _, s in step_pairs))
        print(f"  {len(unique_steps)} plateau-drop events detected")
        for s in unique_steps[:10]:
            print(f"    drop near step {s}")
    else:
        print(f"  no plateau structure detected at this threshold")

    print()
    print(f"=== summary ===")
    success = (
        train_rows[-1]["nll"] < 7.5 and          # meaningful learning
        (not iso or max(iso) < 0.2) and          # no mode collapse
        not any(math.isnan(r["nll"]) for r in train_rows)  # no divergence
    )
    print(f"  Tier 1 MVP completion: {'PASS' if success else 'INVESTIGATE'}")
    return 0 if success else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
