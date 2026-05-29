"""Training loop. CLI: phi-plasma-train --config configs/mvp.yaml"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

from .constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM, SEQ_LEN
from .data import make_loaders
from .losses import combined_loss
from .plasma_core import PlasmaCore
from .symplectic_adamw import SymplecticAdamW
from .vanilla_baseline import VanillaTransformer
from .concentrate_model import ConcentrateModel, combined_concentrate_loss


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: dict, device: torch.device):
    arch = cfg["arch"]
    common = dict(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        d_hidden=cfg.get("d_hidden", D_HIDDEN),
        n_layers=cfg.get("n_layers", N_LAYERS),
        n_heads=cfg.get("n_heads", N_HEADS),
        head_dim=cfg.get("head_dim", HEAD_DIM),
        seq_len=cfg.get("seq_len", SEQ_LEN),
        tie_embeddings=cfg.get("tie_embeddings", True),
    )
    if arch == "plasma":
        m = PlasmaCore(h_step=cfg.get("h_step", 0.5),
                       v_hidden_mult=cfg.get("v_hidden_mult", 2),
                       **common)
    elif arch == "vanilla":
        m = VanillaTransformer(ffn_mult=cfg.get("ffn_mult", 4), **common)
    elif arch == "concentrate":
        m = ConcentrateModel(
            h_step=cfg.get("h_step", 0.5),
            v_hidden_mult=cfg.get("v_hidden_mult", 2),
            use_koopman=cfg.get("use_koopman", True),
            koopman_latent_dim=cfg.get("koopman_latent_dim", 32),
            use_iit_pid=cfg.get("use_iit_pid", True),
            iit_n_partitions=cfg.get("iit_n_partitions", 4),
            use_sheaf=cfg.get("use_sheaf", True),
            sheaf_n_edge_types=cfg.get("sheaf_n_edge_types", 8),
            sheaf_max_skip=cfg.get("sheaf_max_skip", 8),
            **common,
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m.to(device)


def build_optimizer(model, cfg: dict):
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 0.05)
    if cfg.get("optimizer", "symplectic_adamw") == "symplectic_adamw":
        return SymplecticAdamW(model.parameters(), lr=lr,
                                betas=tuple(cfg.get("betas", (0.9, 0.95))),
                                weight_decay=wd,
                                lucas_decay=cfg.get("lucas_decay", True))
    return torch.optim.AdamW(model.parameters(), lr=lr,
                              betas=tuple(cfg.get("betas", (0.9, 0.95))),
                              weight_decay=wd)


def cosine_warmup_lr(step: int, warmup: int, total: int,
                     base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def log_row(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def eval_perplexity(model, val_loader, device, max_batches: int = 32) -> float:
    model.eval()
    total_nll = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        # concentrate returns dict; plasma/vanilla return raw logits tensor
        logits = out["logits"] if isinstance(out, dict) else out
        B, T, V = logits.shape
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        total_nll += float(nll) * B * T
        n += B * T
    model.train()
    return math.exp(total_nll / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml config path")
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke mode: 500 steps, smaller batch, more logging")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.steps is not None:
        cfg["total_steps"] = args.steps
    if args.smoke:
        cfg["total_steps"] = 500
        cfg["batch_size"] = min(cfg.get("batch_size", 8), 4)
        cfg["log_every"] = 10
        cfg["eval_every"] = 100
        cfg["ckpt_every"] = 500

    device = pick_device()
    torch.manual_seed(cfg.get("seed", 0))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"

    print(f"[init] device={device}  config={args.config}", flush=True)
    print(f"[init] cfg={cfg}", flush=True)

    seq_len = cfg.get("seq_len", SEQ_LEN)
    train_loader, val_loader, meta = make_loaders(
        seq_len=seq_len, batch_size=cfg.get("batch_size", 8),
        stride=cfg.get("stride", None),
    )
    print(f"[data] {meta}", flush=True)

    cfg.setdefault("vocab_size", max(VOCAB_SIZE, meta["vocab_inferred_max"]))
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] arch={cfg['arch']}  params={n_params/1e6:.2f}M", flush=True)

    opt = build_optimizer(model, cfg)

    total_steps = cfg.get("total_steps", 30000)
    warmup = cfg.get("warmup", 500)
    eig_warmup = cfg.get("eig_warmup", 2000)
    eig_max = cfg.get("eig_max", 1e-3)
    base_lr = cfg.get("lr", 3e-4)
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 1000)
    ckpt_every = cfg.get("ckpt_every", 5000)
    grad_clip = cfg.get("grad_clip", 1.0)

    step = 0
    t_start = time.time()
    train_iter = iter(train_loader)

    model.train()
    print(f"[train] starting {total_steps} steps", flush=True)

    while step < total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        lr_now = cosine_warmup_lr(step, warmup, total_steps, base_lr)
        set_lr(opt, lr_now)

        opt.zero_grad(set_to_none=True)
        if cfg.get("arch") == "concentrate":
            output = model(x)
            losses = combined_concentrate_loss(
                model, output, y, step,
                eig_warmup=eig_warmup, eig_max=eig_max,
                koopman_weight=cfg.get("koopman_weight", 1e-3),
                iit_weight=cfg.get("iit_weight", 1e-4),
                sheaf_weight=cfg.get("sheaf_weight", 1e-3),
                koopman_warmup=cfg.get("koopman_warmup", 1000),
                iit_warmup=cfg.get("iit_warmup", 1000),
                sheaf_warmup=cfg.get("sheaf_warmup", 500),
            )
        else:
            logits = model(x)
            losses = combined_loss(model, logits, y, step,
                                    warmup=eig_warmup, lam_max=eig_max)
        loss = losses["total"]
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step += 1

        if step % log_every == 0 or step == 1:
            dt = time.time() - t_start
            sps = step / dt

            def _flt(v):
                if isinstance(v, torch.Tensor):
                    return float(v.detach())
                return float(v)

            row = {
                "step": step,
                "loss": float(loss.detach()),
                "nll": _flt(losses["nll"]),
                "eig": _flt(losses.get("eigensheaf", 0.0)),
                "lam": _flt(losses.get("lambda", losses.get("lam_eig", 0.0))),
                "lr": lr_now,
                "sps": sps,
                "elapsed_s": dt,
            }
            # Concentrate-specific probe outputs
            for k in ("koopman_residual", "phi_surrogate", "sheaf_loss",
                       "lam_koopman", "lam_iit", "lam_sheaf"):
                if k in losses:
                    row[k] = _flt(losses[k])
            if hasattr(model, "head_isotypic_distances"):
                dists = model.head_isotypic_distances()
                row["head_iso_mean"] = sum(dists) / len(dists)
                row["head_iso_max"] = max(dists)
            log_row(log_path, row)
            extras = ""
            if "koopman_residual" in row:
                extras = (f"  koop_res={row['koopman_residual']:.3e}"
                          f"  phi={row.get('phi_surrogate', 0.0):.3e}"
                          f"  sheaf={row.get('sheaf_loss', 0.0):.3e}")
            print(f"[step {step:>6d}] nll={row['nll']:.4f}  ppl={math.exp(row['nll']):.1f}  "
                  f"eig={row['eig']:.2e}  iso={row.get('head_iso_mean', float('nan')):.3f}"
                  f"{extras}  lr={lr_now:.2e}  sps={sps:.2f}", flush=True)

        if step % eval_every == 0:
            ppl = eval_perplexity(model, val_loader, device)
            row = {"step": step, "val_ppl": ppl, "elapsed_s": time.time() - t_start}
            log_row(log_path, row)
            print(f"[eval  {step:>6d}] val_ppl={ppl:.2f}", flush=True)

        if step % ckpt_every == 0:
            ckpt_path = out_dir / f"ckpt_{step:08d}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg}, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}", flush=True)

    # Final save.
    torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                out_dir / "ckpt_final.pt")
    print(f"[done] {step} steps in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
