"""Training loop. CLI: phi-plasma-train --config configs/mvp.yaml"""

from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM, SEQ_LEN
from .data import make_loaders, ensure_cache_files
from .packed_data import make_packed_loaders
from .losses import combined_loss
from .plasma_core import PlasmaCore
from .symplectic_adamw import SymplecticAdamW
from .vanilla_baseline import VanillaTransformer
from .concentrate_model import ConcentrateModel, combined_concentrate_loss


def pick_device(local_rank: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def setup_distributed(local_rank_arg: int | None = None) -> dict[str, Any]:
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": local_rank_arg,
            "world_size": 1,
            "device": pick_device(local_rank_arg),
        }

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA in this trainer")

    rank = _env_int("RANK", 0)
    local_rank = local_rank_arg
    if local_rank is None:
        local_rank = _env_int("LOCAL_RANK", rank % torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
    try:
        dist.init_process_group(backend=backend, device_id=torch.device("cuda", local_rank))
    except TypeError:
        dist.init_process_group(backend=backend)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.device("cuda", local_rank),
    }


def cleanup_distributed(dist_ctx: dict[str, Any]) -> None:
    if dist_ctx.get("enabled") and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_ctx: dict[str, Any]) -> bool:
    return int(dist_ctx.get("rank", 0)) == 0


def rank_print(dist_ctx: dict[str, Any], *args, **kwargs) -> None:
    if is_main_process(dist_ctx):
        print(*args, **kwargs)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def resolve_batch_size(cfg: dict, world_size: int) -> tuple[int, int]:
    if cfg.get("global_batch_size") is not None:
        global_batch = int(cfg["global_batch_size"])
        if global_batch % world_size != 0:
            raise ValueError(
                f"global_batch_size={global_batch} must divide world_size={world_size}"
            )
        return global_batch // world_size, global_batch
    per_process = int(cfg.get("per_device_batch_size", cfg.get("batch_size", 8)))
    return per_process, per_process * world_size


def autocast_context(device: torch.device, precision: str):
    precision = (precision or "float32").lower()
    if device.type != "cuda":
        return contextlib.nullcontext()
    if precision in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision in ("fp16", "float16"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and (precision or "").lower() in ("fp16", "float16")
    return torch.cuda.amp.GradScaler(enabled=enabled)


def reduce_row_values(row: dict, device: torch.device, dist_ctx: dict[str, Any],
                      keys: list[str]) -> dict:
    if not dist_ctx.get("enabled"):
        return row
    active_keys = [k for k in keys if k in row]
    if not active_keys:
        return row
    values = torch.tensor([float(row[k]) for k in active_keys],
                          dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= int(dist_ctx["world_size"])
    reduced = dict(row)
    for key, value in zip(active_keys, values.cpu().tolist()):
        reduced[key] = value
    return reduced


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def build_convergence_report(train_history: list[dict], eval_history: list[dict],
                             cfg: dict, final_step: int) -> dict:
    window = int(cfg.get("convergence_window", 40))
    min_points = int(cfg.get("convergence_min_points", 12))
    train_slope_eps = float(cfg.get("convergence_train_slope_epsilon", 0.005))
    val_min_delta = float(cfg.get("convergence_val_min_delta", 0.002))
    patience = int(cfg.get("convergence_patience", 3))

    train_rows = [r for r in train_history if "nll" in r]
    eval_rows = [r for r in eval_history if "val_ppl" in r]
    report = {
        "step": final_step,
        "train_points": len(train_rows),
        "eval_points": len(eval_rows),
        "window": min(window, len(train_rows)),
        "status": "insufficient_data",
        "reason": "not enough logged points for convergence assessment",
        "recommendation": "continue_or_run_longer_with_more_eval_points",
    }

    if len(train_rows) >= min_points:
        recent = train_rows[-min(window, len(train_rows)):]
        xs = [float(r["step"]) for r in recent]
        ys = [float(r["nll"]) for r in recent]
        slope_per_1k = linear_slope(xs, ys) * 1000.0
        mean_recent = sum(ys) / len(ys)
        variance = sum((y - mean_recent) ** 2 for y in ys) / len(ys)
        report.update({
            "train_nll_final": float(train_rows[-1]["nll"]),
            "train_nll_best": min(float(r["nll"]) for r in train_rows),
            "train_slope_nll_per_1k": slope_per_1k,
            "train_nll_recent_std": variance ** 0.5,
            "train_plateau": abs(slope_per_1k) <= train_slope_eps,
        })

    if eval_rows:
        best = min(eval_rows, key=lambda r: float(r["val_ppl"]))
        final = eval_rows[-1]
        report.update({
            "val_ppl_final": float(final["val_ppl"]),
            "val_ppl_best": float(best["val_ppl"]),
            "val_best_step": int(best["step"]),
            "val_final_is_best": int(final["step"]) == int(best["step"]),
        })
        if len(eval_rows) >= 2:
            rel_improvements = []
            for prev, curr in zip(eval_rows, eval_rows[1:]):
                prev_ppl = float(prev["val_ppl"])
                curr_ppl = float(curr["val_ppl"])
                rel_improvements.append((prev_ppl - curr_ppl) / max(prev_ppl, 1e-12))
            recent_improvements = rel_improvements[-patience:]
            report.update({
                "val_recent_rel_improvements": recent_improvements,
                "val_improving": any(x > val_min_delta for x in recent_improvements),
                "val_min_delta": val_min_delta,
                "patience": patience,
            })

    train_plateau = bool(report.get("train_plateau", False))
    val_improving = bool(report.get("val_improving", False))
    val_final_is_best = bool(report.get("val_final_is_best", False))

    if "train_plateau" not in report:
        return report
    if train_plateau and not val_improving:
        report.update({
            "status": "converged_or_lr_limited",
            "reason": "train slope is flat and validation has not improved beyond threshold recently",
            "recommendation": "stop_or_resume_with_lower_lr_if_more_quality_is_needed",
        })
    elif train_plateau and val_final_is_best:
        report.update({
            "status": "validation_still_creeping",
            "reason": "train NLL is flat but latest validation is still the best observed",
            "recommendation": "extend modestly or lower eval interval before declaring convergence",
        })
    else:
        report.update({
            "status": "not_converged",
            "reason": "train or validation trend still exceeds convergence threshold",
            "recommendation": "continue_training_and_reassess",
        })
    return report


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


def gradient_heatmap_summary(model, max_groups: int = 96) -> dict[str, Any]:
    groups: dict[str, dict[str, float]] = {}
    total_sq = 0.0
    max_abs = 0.0

    def group_name(name: str) -> str:
        parts = name.split(".")
        if parts and parts[0] == "module":
            parts = parts[1:]
        if len(parts) >= 2 and parts[0] in {"layers", "blocks", "hblocks", "transformer"}:
            return ".".join(parts[:2])
        if len(parts) >= 3 and parts[0] in {"model", "net"} and parts[1] in {"layers", "blocks"}:
            return ".".join(parts[:3])
        return parts[0] if parts else name

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        detached = grad.detach().float()
        sq = float(detached.pow(2).sum().cpu())
        group = groups.setdefault(group_name(name), {"sq": 0.0, "max_abs": 0.0, "tensors": 0.0, "params": 0.0})
        group["sq"] += sq
        group["max_abs"] = max(group["max_abs"], float(detached.abs().max().cpu()))
        group["tensors"] += 1.0
        group["params"] += float(param.numel())
        total_sq += sq
        max_abs = max(max_abs, group["max_abs"])

    rows = []
    for group, stats in groups.items():
        norm = math.sqrt(max(stats["sq"], 0.0))
        rows.append({
            "group": group,
            "norm": norm,
            "log10_norm": math.log10(max(norm, 1e-30)),
            "max_abs": stats["max_abs"],
            "tensors": int(stats["tensors"]),
            "params": int(stats["params"]),
        })
    rows.sort(key=lambda item: item["group"])
    if max_groups > 0 and len(rows) > max_groups:
        rows = rows[:max_groups]
    return {
        "grad_total_norm": math.sqrt(max(total_sq, 0.0)),
        "grad_max_abs": max_abs,
        "grad_groups": rows,
    }


@torch.no_grad()
def eval_perplexity(model, val_loader, device, max_batches: int = 32,
                    dist_ctx: dict[str, Any] | None = None) -> float:
    model.eval()
    total_nll = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        # concentrate returns dict; plasma/vanilla return raw logits tensor
        logits = out["logits"] if isinstance(out, dict) else out
        B, T, V = logits.shape
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        total_nll += float(nll) * B * T
        n += B * T

    if dist_ctx and dist_ctx.get("enabled"):
        totals = torch.tensor([total_nll, float(n)], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_nll = float(totals[0].cpu())
        n = int(totals[1].cpu())

    model.train()
    return math.exp(total_nll / max(1, n))


def checkpoint_payload(model, opt, step: int, cfg: dict,
                       dist_ctx: dict[str, Any], include_opt: bool = True) -> dict:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "step": step,
        "cfg": cfg,
        "distributed": {
            "world_size": int(dist_ctx.get("world_size", 1)),
            "rank0_device": str(dist_ctx.get("device", "cpu")),
        },
    }
    if include_opt:
        payload["opt"] = opt.state_dict()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml config path")
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke mode: 500 steps, smaller batch, more logging")
    ap.add_argument("--local-rank", "--local_rank", type=int, default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    dist_ctx = setup_distributed(args.local_rank)
    device = dist_ctx["device"]

    try:
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

        if device.type == "cuda":
            torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
        torch.manual_seed(cfg.get("seed", 0))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.get("seed", 0))

        batch_size, global_batch_size = resolve_batch_size(cfg, dist_ctx["world_size"])
        cfg["batch_size_per_process"] = batch_size
        cfg["global_batch_size_runtime"] = global_batch_size
        cfg["world_size"] = dist_ctx["world_size"]

        out_dir = Path(cfg["out_dir"])
        if is_main_process(dist_ctx):
            out_dir.mkdir(parents=True, exist_ok=True)
        if dist_ctx.get("enabled"):
            dist.barrier()
        log_path = out_dir / "metrics.jsonl"

        rank_print(dist_ctx, f"[init] device={device}  world_size={dist_ctx['world_size']}  config={args.config}", flush=True)
        rank_print(dist_ctx, f"[init] cfg={cfg}", flush=True)

        seq_len = cfg.get("seq_len", SEQ_LEN)
        data_source = cfg.get("data_source", "wikitext_cache")
        if data_source == "packed":
            packed_dir = cfg.get("packed_data_dir")
            if not packed_dir:
                raise ValueError("data_source=packed requires packed_data_dir")
            train_loader, val_loader, meta = make_packed_loaders(
                data_dir=packed_dir,
                seq_len=seq_len,
                batch_size=batch_size,
                num_workers=cfg.get("num_workers", 0),
                stride=cfg.get("stride", None),
                distributed=dist_ctx.get("enabled", False),
                rank=dist_ctx["rank"],
                world_size=dist_ctx["world_size"],
                pin_memory=cfg.get("pin_memory", device.type == "cuda"),
                seed=cfg.get("seed", 0),
            )
        else:
            data_cache_dir = cfg.get("data_cache_dir")
            seq_in_cache = cfg.get("seq_in_cache", 256)
            data_auto_prepare = cfg.get("data_auto_prepare", True)
            if dist_ctx.get("enabled"):
                if is_main_process(dist_ctx):
                    cache_meta = ensure_cache_files(
                        cache_dir=data_cache_dir,
                        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
                        seq_in_cache=seq_in_cache,
                        auto_prepare=data_auto_prepare,
                    )
                    rank_print(dist_ctx, f"[data] cache={cache_meta}", flush=True)
                dist.barrier()

            train_loader, val_loader, meta = make_loaders(
                seq_len=seq_len,
                batch_size=batch_size,
                num_workers=cfg.get("num_workers", 0),
                stride=cfg.get("stride", None),
                distributed=dist_ctx.get("enabled", False),
                rank=dist_ctx["rank"],
                world_size=dist_ctx["world_size"],
                pin_memory=cfg.get("pin_memory", device.type == "cuda"),
                seed=cfg.get("seed", 0),
                cache_dir=data_cache_dir,
                vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
                seq_in_cache=seq_in_cache,
                auto_prepare=(not dist_ctx.get("enabled", False)) and data_auto_prepare,
            )
        meta["global_batch_size"] = global_batch_size
        rank_print(dist_ctx, f"[data] {meta}", flush=True)

        cfg.setdefault("vocab_size", max(VOCAB_SIZE, meta["vocab_inferred_max"]))
        if meta.get("tokenizer_name") and not cfg.get("tokenizer_name"):
            cfg["tokenizer_name"] = meta["tokenizer_name"]
        if meta.get("tokenizer_path") and not cfg.get("tokenizer_path"):
            cfg["tokenizer_path"] = meta["tokenizer_path"]
        model = build_model(cfg, device)
        n_params = sum(p.numel() for p in model.parameters())
        rank_print(dist_ctx, f"[model] arch={cfg['arch']}  params={n_params/1e6:.2f}M", flush=True)

        if dist_ctx.get("enabled"):
            ddp_kwargs = {
                "device_ids": [dist_ctx["local_rank"]],
                "output_device": dist_ctx["local_rank"],
                "find_unused_parameters": cfg.get("find_unused_parameters", False),
            }
            if cfg.get("ddp_static_graph", True):
                ddp_kwargs["static_graph"] = True
            try:
                model = DDP(model, **ddp_kwargs)
            except TypeError:
                ddp_kwargs.pop("static_graph", None)
                model = DDP(model, **ddp_kwargs)
                if cfg.get("ddp_static_graph", True) and hasattr(model, "_set_static_graph"):
                    model._set_static_graph()

        raw_model = unwrap_model(model)
        opt = build_optimizer(raw_model, cfg)
        precision = cfg.get("precision", "float32")
        scaler = make_grad_scaler(device, precision)

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
        epoch = 0
        t_start = time.time()
        train_sampler = getattr(train_loader, "sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        train_history: list[dict] = []
        eval_history: list[dict] = []

        model.train()
        rank_print(dist_ctx, f"[train] starting {total_steps} steps", flush=True)

        while step < total_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                if hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lr_now = cosine_warmup_lr(step, warmup, total_steps, base_lr)
            set_lr(opt, lr_now)

            opt.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                if cfg.get("arch") == "concentrate":
                    output = model(x)
                    losses = combined_concentrate_loss(
                        raw_model, output, y, step,
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
                    losses = combined_loss(raw_model, logits, y, step,
                                            warmup=eig_warmup, lam_max=eig_max)
                loss = losses["total"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
                opt.step()

            step += 1

            if step % log_every == 0 or step == 1:
                sync_device(device)
                dt = time.time() - t_start
                sps = step / dt
                samples_s = (step * global_batch_size) / dt
                tokens_s = samples_s * seq_len

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
                    "samples_s": samples_s,
                    "tokens_s": tokens_s,
                    "elapsed_s": dt,
                    "world_size": dist_ctx["world_size"],
                    "batch_size_per_process": batch_size,
                    "global_batch_size": global_batch_size,
                }
                for k in ("koopman_residual", "phi_surrogate", "sheaf_loss",
                           "lam_koopman", "lam_iit", "lam_sheaf"):
                    if k in losses:
                        row[k] = _flt(losses[k])
                if hasattr(raw_model, "head_isotypic_distances"):
                    dists = raw_model.head_isotypic_distances()
                    row["head_iso_mean"] = sum(dists) / len(dists)
                    row["head_iso_max"] = max(dists)

                row = reduce_row_values(
                    row, device, dist_ctx,
                    [
                        "loss", "nll", "eig", "koopman_residual",
                        "phi_surrogate", "sheaf_loss", "head_iso_mean",
                        "head_iso_max",
                    ],
                )
                if is_main_process(dist_ctx):
                    if cfg.get("grad_heatmap", True):
                        row.update(gradient_heatmap_summary(raw_model, int(cfg.get("grad_heatmap_max_groups", 96))))
                    train_history.append(dict(row))
                    log_row(log_path, row)
                    extras = ""
                    if "koopman_residual" in row:
                        extras = (f"  koop_res={row['koopman_residual']:.3e}"
                                  f"  phi={row.get('phi_surrogate', 0.0):.3e}"
                                  f"  sheaf={row.get('sheaf_loss', 0.0):.3e}")
                    print(f"[step {step:>6d}] nll={row['nll']:.4f}  ppl={math.exp(row['nll']):.1f}  "
                          f"eig={row['eig']:.2e}  iso={row.get('head_iso_mean', float('nan')):.3f}"
                          f"  grad={row.get('grad_total_norm', float('nan')):.2e}"
                          f"{extras}  lr={lr_now:.2e}  sps={sps:.2f}  tok/s={tokens_s:.0f}", flush=True)

            if step % eval_every == 0:
                ppl = eval_perplexity(model, val_loader, device, dist_ctx=dist_ctx)
                if is_main_process(dist_ctx):
                    row = {"step": step, "val_ppl": ppl, "elapsed_s": time.time() - t_start}
                    eval_history.append(dict(row))
                    log_row(log_path, row)
                    print(f"[eval  {step:>6d}] val_ppl={ppl:.2f}", flush=True)
                    report = build_convergence_report(train_history, eval_history, cfg, step)
                    if report.get("status") != "insufficient_data":
                        log_row(log_path, {"step": step, "convergence": report, "elapsed_s": time.time() - t_start})
                        print(f"[convergence] {report['status']}: {report['reason']}", flush=True)

            if step % ckpt_every == 0:
                if is_main_process(dist_ctx):
                    ckpt_path = out_dir / f"ckpt_{step:08d}.pt"
                    torch.save(checkpoint_payload(model, opt, step, cfg, dist_ctx), ckpt_path)
                    print(f"[ckpt] saved {ckpt_path}", flush=True)
                if dist_ctx.get("enabled"):
                    dist.barrier()

        if is_main_process(dist_ctx):
            report = build_convergence_report(train_history, eval_history, cfg, step)
            log_row(log_path, {"step": step, "convergence": report, "elapsed_s": time.time() - t_start})
            print(f"[convergence] {report['status']}: {report['recommendation']}", flush=True)
            torch.save(checkpoint_payload(model, opt, step, cfg, dist_ctx, include_opt=False),
                       out_dir / "ckpt_final.pt")
            print(f"[done] {step} steps in {time.time() - t_start:.1f}s", flush=True)
        if dist_ctx.get("enabled"):
            dist.barrier()
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
