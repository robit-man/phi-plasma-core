"""Generate the paper's figures from training metrics.

Produces:
  - fig1_long_context_stability.pdf : the 2.31× headline chart
  - fig2_training_curves.pdf : NLL trajectories for all four runs
  - fig3_ablation_summary.pdf : final ppl bar chart

Usage:
    python paper/figures/build_figures.py
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "logs"
OUT = Path(__file__).resolve().parent


def load_metrics(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ─── Figure 1: long-context stability (the headline) ─────────────
def fig1_long_context():
    # Data from the chunked-attention eval, hand-transcribed from the
    # log_context_eval.py outputs documented in DAY8_VERDICT.md
    contexts = [1024, 4096, 8192, 16384, 32768, 65536]
    plasma = [402, 458, 445, 448, 453, 453]
    vanilla = [335, 409, 394, 413, 428, 432]

    # Compute fractional drift from training context
    plasma_drift = [(p - plasma[0]) / plasma[0] * 100 for p in plasma]
    vanilla_drift = [(v - vanilla[0]) / vanilla[0] * 100 for v in vanilla]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: absolute perplexity
    ax1.plot(contexts, plasma, "o-", label="Plasma (ours)", linewidth=2, markersize=7, color="#1f77b4")
    ax1.plot(contexts, vanilla, "s-", label="Vanilla baseline", linewidth=2, markersize=7, color="#d62728")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Context length (tokens)", fontsize=11)
    ax1.set_ylabel("Validation perplexity", fontsize=11)
    ax1.set_title("Absolute perplexity", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(1024, linestyle="--", color="gray", alpha=0.5)
    ax1.text(1024 * 1.05, 460, "training context",
             fontsize=8, color="gray", rotation=90, va="top")

    # Right: relative drift
    ax2.plot(contexts, plasma_drift, "o-", label="Plasma: +12.6% total", linewidth=2, markersize=7, color="#1f77b4")
    ax2.plot(contexts, vanilla_drift, "s-", label="Vanilla: +29.1% total", linewidth=2, markersize=7, color="#d62728")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Context length (tokens)", fontsize=11)
    ax2.set_ylabel("Perplexity drift from training context", fontsize=11)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title(r"Drift — plasma is $2.31\times$ more stable", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, linestyle="-", color="black", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = OUT / "fig1_long_context_stability.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(OUT / "fig1_long_context_stability.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


# ─── Figure 2: training curves ───────────────────────────────────
def fig2_training_curves():
    runs = {
        "Plasma A1 (h=0.5)": (LOGS / "ablate_A1" / "metrics.jsonl", "#1f77b4", "-"),
        "Plasma v3 (h=0.2)": (LOGS / "v3_plasma" / "metrics.jsonl", "#7da0c0", ":"),
        "Vanilla baseline": (LOGS / "v3_vanilla" / "metrics.jsonl", "#d62728", "-"),
        "Plasma A2 (ν=4)": (LOGS / "ablate_A2" / "metrics.jsonl", "#2ca02c", "--"),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (path, color, ls) in runs.items():
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        rows = load_metrics(path)
        steps = [r["step"] for r in rows if "nll" in r]
        ppls = [2.718 ** r["nll"] for r in rows if "nll" in r]
        ax.plot(steps, ppls, color=color, linestyle=ls, linewidth=1.5, label=name, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training step (log scale)", fontsize=11)
    ax.set_ylabel("Training perplexity (log scale)", fontsize=11)
    ax.set_title("Training trajectories — plasma at h=0.5 tracks vanilla closely",
                  fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path = OUT / "fig2_training_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(OUT / "fig2_training_curves.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


# ─── Figure 3: ablation bar chart ────────────────────────────────
def fig3_ablations():
    runs = [
        ("Plasma v3\n(h=0.2)", 464.97, "#7da0c0"),
        ("Plasma A1\n(h=0.5)", 389.76, "#1f77b4"),
        ("Plasma A2\n(ν=4)",   390.65, "#2ca02c"),
        ("Vanilla\nbaseline",   374.43, "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r[0] for r in runs]
    values = [r[1] for r in runs]
    colors = [r[2] for r in runs]

    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5,
                 f"{val:.1f}", ha="center", fontsize=10)

    ax.set_ylabel("Final val_ppl @ step 3000", fontsize=11)
    ax.set_title("Ablation summary — h-step is the dominant recoverable factor", fontsize=12)
    ax.set_ylim(0, max(values) * 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(374.43, linestyle="--", color="black", alpha=0.5, linewidth=0.8)
    ax.text(0.02, 374.43 + 5, "vanilla baseline", fontsize=8, color="black", alpha=0.7)

    plt.tight_layout()
    out_path = OUT / "fig3_ablation_summary.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(OUT / "fig3_ablation_summary.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def main():
    print("Building paper figures...")
    fig1_long_context()
    fig2_training_curves()
    fig3_ablations()
    print("\nFigures written to:", OUT)
    print("To embed in main.tex, add e.g.:")
    print(r"  \includegraphics[width=\textwidth]{figures/fig1_long_context_stability.pdf}")


if __name__ == "__main__":
    sys.exit(main() or 0)
