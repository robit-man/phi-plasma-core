"""Assess whether a training run has converged from metrics.jsonl.

This is intentionally separate from headline-result scripts: it answers the
operational question "should this run stop, continue, or resume with a changed
schedule?"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from phi_plasma.train import build_convergence_report


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_run_segment(rows: list[dict]) -> list[dict]:
    starts = [i for i, row in enumerate(rows) if row.get("step") == 1 and "nll" in row]
    return rows[starts[-1]:] if starts else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/a100_3gpu_plasma_d704",
                    help="run directory containing metrics.jsonl")
    ap.add_argument("--window", type=int, default=40,
                    help="number of train log rows for final slope")
    ap.add_argument("--train-slope-epsilon", type=float, default=0.005,
                    help="absolute NLL slope per 1K steps considered flat")
    ap.add_argument("--val-min-delta", type=float, default=0.002,
                    help="relative val PPL improvement threshold")
    ap.add_argument("--patience", type=int, default=3,
                    help="validation intervals for recent-improvement check")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.run) / "metrics.jsonl"
    if not log_path.exists():
        print(f"[error] no metrics log at {log_path}")
        return 1

    rows = latest_run_segment(load_rows(log_path))
    train_rows = [r for r in rows if "nll" in r]
    eval_rows = [r for r in rows if "val_ppl" in r]
    cfg = {
        "convergence_window": args.window,
        "convergence_train_slope_epsilon": args.train_slope_epsilon,
        "convergence_val_min_delta": args.val_min_delta,
        "convergence_patience": args.patience,
    }
    final_step = int(train_rows[-1]["step"] if train_rows else rows[-1].get("step", 0))
    report = build_convergence_report(train_rows, eval_rows, cfg, final_step)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== convergence: {args.run} ===")
    print(f"status: {report['status']}")
    print(f"reason: {report['reason']}")
    print(f"recommendation: {report['recommendation']}")
    print(f"train rows: {report['train_points']}  eval rows: {report['eval_points']}  window: {report['window']}")
    if "train_nll_final" in report:
        print(f"train nll: final={report['train_nll_final']:.4f} best={report['train_nll_best']:.4f}")
        print(f"train slope: {report['train_slope_nll_per_1k']:+.6f} nll / 1K steps")
        print(f"train recent std: {report['train_nll_recent_std']:.6f}")
    if "val_ppl_final" in report:
        print(f"val ppl: final={report['val_ppl_final']:.6f} best={report['val_ppl_best']:.6f} at step {report['val_best_step']}")
        recent = report.get("val_recent_rel_improvements", [])
        if recent:
            print("recent val relative improvements: " + ", ".join(f"{x*100:.3f}%" for x in recent))
    if "train_nll_final" in report:
        print(f"train ppl final: {math.exp(report['train_nll_final']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
