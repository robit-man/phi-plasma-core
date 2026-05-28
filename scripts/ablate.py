"""Run plasma + vanilla controls sequentially and write a comparison summary.

Usage:
    scripts/py scripts/ablate.py
"""

from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    "configs/mvp_18m.yaml",
    "configs/vanilla_18m.yaml",
]


def main():
    summary_path = ROOT / "logs" / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in CONFIGS:
        cfg_path = ROOT / cfg
        print(f"\n{'='*72}\n[ablate] launching: {cfg}\n{'='*72}", flush=True)
        t0 = time.time()
        rc = subprocess.call([
            sys.executable, "-m", "phi_plasma.train", "--config", str(cfg_path)
        ], cwd=str(ROOT))
        dt = time.time() - t0
        print(f"[ablate] {cfg} → returncode={rc} in {dt:.1f}s", flush=True)

        # Read final metrics from the run's log.
        with (cfg_path).open() as f:
            import yaml
            cfg_obj = yaml.safe_load(f)
        log_path = ROOT / cfg_obj["out_dir"] / "metrics.jsonl"
        last_eval_ppl = None
        last_train_nll = None
        last_iso = None
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                row = json.loads(line)
                if "val_ppl" in row:
                    last_eval_ppl = row["val_ppl"]
                if "nll" in row:
                    last_train_nll = row["nll"]
                if "head_iso_mean" in row:
                    last_iso = row["head_iso_mean"]
        results.append({
            "config": cfg,
            "arch": cfg_obj["arch"],
            "returncode": rc,
            "wall_seconds": dt,
            "final_train_nll": last_train_nll,
            "final_val_ppl": last_eval_ppl,
            "final_head_iso_mean": last_iso,
        })

    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[ablate] summary written: {summary_path}", flush=True)
    for r in results:
        print(f"  {r['config']:30s}  arch={r['arch']:8s}  "
              f"val_ppl={r['final_val_ppl']}  iso={r['final_head_iso_mean']}", flush=True)


if __name__ == "__main__":
    main()
