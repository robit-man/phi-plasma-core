"""Evaluate trained checkpoints at sequence lengths longer than training context.

Plasma's volume-preserving Hamiltonian flow should preserve information across
the deeper-than-trained context. Vanilla transformers degrade past their
training context due to positional encoding limits and attention's
unconditional pairwise interaction.

The test: evaluate val_ppl at seq_len = {1024 (train), 1536, 2048, 3072}.
Measure relative degradation. The architecture that degrades less is the
information-preserving one.

Usage:
    scripts/py scripts/long_context_eval.py --ckpt logs/v3_plasma/ckpt_final.pt
    scripts/py scripts/long_context_eval.py --ckpt logs/v3_vanilla/ckpt_final.pt
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phi_plasma.constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM
from phi_plasma.data import find_cache, load_token_stream
from phi_plasma.plasma_core import PlasmaCore
from phi_plasma.vanilla_baseline import VanillaTransformer
from phi_plasma.chunked_attention import patch_for_long_context


def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_cfg(cfg):
    common = dict(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        d_hidden=cfg.get("d_hidden", D_HIDDEN),
        n_layers=cfg.get("n_layers", N_LAYERS),
        n_heads=cfg.get("n_heads", N_HEADS),
        head_dim=cfg.get("head_dim", HEAD_DIM),
        seq_len=cfg.get("seq_len", 1024),
        tie_embeddings=cfg.get("tie_embeddings", True),
    )
    if cfg["arch"] == "plasma":
        return PlasmaCore(h_step=cfg.get("h_step", 0.5),
                          v_hidden_mult=cfg.get("v_hidden_mult", 2),
                          **common)
    return VanillaTransformer(ffn_mult=cfg.get("ffn_mult", 4), **common)


def make_model_for_eval(model_cfg, eval_seq_len: int, device):
    """Build a model with seq_len ≥ eval target — pos_embed must cover full length.

    For this MVP both models use learned positional embeddings sized to seq_len.
    To evaluate at eval_seq_len > train_seq_len, we'd need to either:
      (a) extend the positional embedding by interpolation
      (b) evaluate sliding-window style

    We use (b): chunk the test data into eval_seq_len overlapping windows and
    process each with a model whose seq_len matches eval_seq_len, but reuse the
    trained model's weights with positional embedding EXTRAPOLATED via linear
    interpolation. This is a standard NTK-aware eval; better than refusing to
    test past trained context."""
    train_seq = model_cfg.get("seq_len", 1024)
    # Build model with eval_seq_len capacity
    eval_cfg = dict(model_cfg)
    eval_cfg["seq_len"] = eval_seq_len
    model = build_model_from_cfg(eval_cfg).to(device)
    return model, train_seq


@torch.no_grad()
def eval_at_seq_len(model, tokens, seq_len, device, max_windows=20):
    """Stride-based eval over the val set at the requested sequence length."""
    n_windows = min(max_windows, (len(tokens) - seq_len - 1) // seq_len)
    if n_windows < 1:
        return float("nan"), 0
    total_nll, total_tokens = 0.0, 0
    model.eval()
    for w in range(n_windows):
        start = w * seq_len
        x = tokens[start:start+seq_len].unsqueeze(0).to(device)
        y = tokens[start+1:start+seq_len+1].unsqueeze(0).to(device)
        logits = model(x)
        B, T, V = logits.shape
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="sum"
        )
        total_nll += float(nll)
        total_tokens += B * T
    return math.exp(total_nll / total_tokens), n_windows


def remap_pos_embedding(state_dict, old_seq, new_seq):
    """Linearly interpolate positional embedding from old_seq → new_seq tokens.

    Standard NTK-aware extrapolation: scale positions by old/new ratio so that
    relative phase relationships are preserved."""
    key = "pos_embed.weight"
    if key not in state_dict:
        return state_dict
    old_pe = state_dict[key]   # (old_seq, d_hidden)
    if old_pe.shape[0] == new_seq:
        return state_dict
    # Interpolate along the sequence dimension.
    old_pe = old_pe.unsqueeze(0).unsqueeze(0)   # (1, 1, old_seq, d_hidden)
    new_pe = torch.nn.functional.interpolate(
        old_pe.float(), size=(new_seq, old_pe.shape[-1]), mode="bilinear", align_corners=False
    )
    state_dict = dict(state_dict)
    state_dict[key] = new_pe.squeeze(0).squeeze(0).to(old_pe.dtype)
    return state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-lens", default="1024,1536,2048,3072")
    ap.add_argument("--max-windows", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=0,
                    help="If >0, patch attention to chunked (eval-only, fits long contexts)")
    args = ap.parse_args()

    device = pick_device()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[error] ckpt not found: {ckpt_path}")
        return 1
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    if "arch" not in cfg:
        # Try to infer from state dict
        sd = ck["model"] if "model" in ck else ck
        cfg["arch"] = "plasma" if any("hamiltonian" in k or "hecke" in k for k in sd.keys()) else "vanilla"

    train_seq = cfg.get("seq_len", 1024)
    print(f"[ckpt] {ckpt_path}  arch={cfg['arch']}  train_seq={train_seq}", flush=True)

    val_path = find_cache("validation")
    tokens = load_token_stream(val_path)
    print(f"[data] {len(tokens):,} validation tokens", flush=True)

    results = []
    for s in (int(x) for x in args.seq_lens.split(",")):
        # Build fresh model for this seq_len + remap positional embedding
        model, _ = make_model_for_eval(cfg, s, device)
        sd = ck["model"] if "model" in ck else ck
        sd = remap_pos_embedding(sd, train_seq, s)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"  seq={s}: load failed: {e}", flush=True)
            continue
        if args.chunk > 0:
            n_patched = patch_for_long_context(model, query_chunk=args.chunk)
            print(f"  seq={s}: patched {n_patched} attention layers, chunk={args.chunk}", flush=True)
        t0 = time.time()
        ppl, n = eval_at_seq_len(model, tokens, s, device, args.max_windows)
        dt = time.time() - t0
        results.append({"seq_len": s, "val_ppl": ppl, "n_windows": n, "time_s": dt})
        ratio = "" if s == train_seq else f"  (× {ppl / results[0]['val_ppl']:.2f} vs train_seq)"
        print(f"  seq={s:>4d}  val_ppl={ppl:>8.2f}  ({n} windows in {dt:.1f}s){ratio}", flush=True)
        del model

    print()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
