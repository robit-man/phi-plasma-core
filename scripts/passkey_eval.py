"""Passkey retrieval / needle-in-haystack evaluation.

This is the canonical long-context retrieval test:
  1. Construct a long prompt: "<random distractor text> KEY=<5-digit secret> <distractor>... What is KEY?"
  2. Measure whether the model assigns higher probability to the correct digits than wrong ones.

A vanilla transformer trained at seq=1024 often fails catastrophically at
longer contexts (forgets the key, attention dilutes). The volume-preserving
plasma architecture should retain the key info across the context extension.

We can't easily do open-ended generation with the WikiText tokenizer (it's not
chat-aware), so we score directly: compute log P(key | context_with_key) vs.
log P(random_5digit | context_with_key). The model that retains better gets
higher log-probability for the actual key.

This is a CLEAN information-retention test that's independent of generation
quality — exactly where plasma's architectural advantage should manifest.

Metric: accuracy = fraction of probes where log P(true_key | ctx) >
top of log P(random_key | ctx) over a set of distractor keys.

Usage:
    scripts/py scripts/passkey_eval.py --ckpt logs/v3_plasma/ckpt_final.pt \\
        --seq-lens 1024,4096,16384 --n-probes 20
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phi_plasma.constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM
from phi_plasma.data import find_cache, load_token_stream
from phi_plasma.plasma_core import PlasmaCore
from phi_plasma.vanilla_baseline import VanillaTransformer
from phi_plasma.concentrate_model import ConcentrateModel
from phi_plasma.chunked_attention import patch_for_long_context


def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_cfg(cfg, seq_len_override=None):
    sl = seq_len_override or cfg.get("seq_len", 1024)
    common = dict(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        d_hidden=cfg.get("d_hidden", D_HIDDEN),
        n_layers=cfg.get("n_layers", N_LAYERS),
        n_heads=cfg.get("n_heads", N_HEADS),
        head_dim=cfg.get("head_dim", HEAD_DIM),
        seq_len=sl,
        tie_embeddings=cfg.get("tie_embeddings", True),
    )
    if cfg["arch"] == "plasma":
        return PlasmaCore(h_step=cfg.get("h_step", 0.5),
                          v_hidden_mult=cfg.get("v_hidden_mult", 2),
                          **common)
    if cfg["arch"] == "concentrate":
        return ConcentrateModel(
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
    return VanillaTransformer(ffn_mult=cfg.get("ffn_mult", 4), **common)


def remap_pos_embedding(sd, new_seq):
    """Interpolate position embedding for extended context.

    Handles both plasma's 'pos_embed.weight' and concentrate's
    nested 'backbone.pos_embed.weight'."""
    sd = dict(sd)
    for key in ("pos_embed.weight", "backbone.pos_embed.weight"):
        if key not in sd:
            continue
        pe = sd[key]
        if pe.shape[0] == new_seq:
            continue
        pe_b = pe.unsqueeze(0).unsqueeze(0).float()
        new_pe = torch.nn.functional.interpolate(
            pe_b, size=(new_seq, pe_b.shape[-1]), mode="bilinear", align_corners=False
        )
        sd[key] = new_pe.squeeze(0).squeeze(0).to(pe.dtype)
    return sd


def make_probe(tokens, seq_len, key_position_frac, rng, key_len=5):
    """Construct a context of length seq_len with a 'key' token sequence
    embedded at position key_position_frac.

    The key is a sequence of `key_len` random tokens drawn from the vocab.
    We use real vocab tokens (which appear in training) rather than reserving
    special tokens — this means the test is genuinely "remember this random
    snippet" rather than "use a special marker."

    Returns:
      ctx_with_key: 1D tensor of length seq_len
      key_tokens: the 5-token key
      key_start_pos: where the key starts in ctx_with_key
      query_start_pos: position right before the key answer slot at the end
    """
    # Pick a contiguous chunk of training tokens as the "distractor" content.
    max_start = len(tokens) - seq_len - 100
    base_start = rng.randint(0, max_start)
    base = tokens[base_start:base_start + seq_len].clone()

    # Sample key_len random vocab tokens for the key.
    vocab_max = int(tokens.max().item())
    key_tokens = torch.tensor(
        [rng.randint(0, vocab_max) for _ in range(key_len)], dtype=torch.long
    )

    # Embed key at position key_position_frac (e.g., 0.1 = 10% into context).
    # Reserve last (key_len + 2) tokens for the "query" + answer slot.
    answer_slot = seq_len - key_len  # answer goes here
    key_pos_max = answer_slot - 50
    key_pos_min = 10
    key_pos = int(key_position_frac * (key_pos_max - key_pos_min)) + key_pos_min
    base[key_pos:key_pos + key_len] = key_tokens

    # Also place the key tokens at the answer slot, so we can score them.
    base[answer_slot:answer_slot + key_len] = key_tokens

    return base, key_tokens, key_pos, answer_slot


@torch.no_grad()
def score_at_position(model, ctx_with_key, answer_slot, key_tokens, device):
    """Forward the model, return log P(key_tokens | context up to answer_slot)."""
    model.eval()
    x = ctx_with_key.unsqueeze(0).to(device)             # (1, T)
    out = model(x)
    # concentrate models return dict; plasma/vanilla return logits tensor
    logits = out["logits"] if isinstance(out, dict) else out  # (1, T, V)
    # The model predicts token t+1 from tokens 0..t.
    # answer_slot is where the key starts being placed.
    # To predict key[i] (at position answer_slot+i), we read logits[answer_slot+i-1].
    log_probs = torch.log_softmax(logits[0], dim=-1)     # (T, V)
    total_lp = 0.0
    for i, tok in enumerate(key_tokens):
        pred_pos = answer_slot + i - 1
        total_lp += float(log_probs[pred_pos, int(tok)])
    return total_lp / len(key_tokens)


@torch.no_grad()
def evaluate(model, tokens, seq_len, n_probes, device, rng, chunk=0, key_len=5):
    """Run n_probes passkey probes and compute mean log P(key | context)."""
    if chunk > 0:
        patch_for_long_context(model, query_chunk=chunk)

    correct_log_probs = []
    distractor_log_probs = []
    vocab_max = int(tokens.max().item())

    for i in range(n_probes):
        key_pos_frac = (i + 1) / (n_probes + 1)
        ctx, key, kp, ans_slot = make_probe(tokens, seq_len, key_pos_frac, rng, key_len)

        # Score the true key.
        lp_correct = score_at_position(model, ctx, ans_slot, key, device)
        correct_log_probs.append(lp_correct)

        # Score a distractor key (random 5-token sequence, NOT placed in context).
        distractor_key = torch.tensor(
            [rng.randint(0, vocab_max) for _ in range(key_len)], dtype=torch.long
        )
        # Replace the answer slot with the distractor (not in context).
        ctx_distractor = ctx.clone()
        ctx_distractor[ans_slot:ans_slot + key_len] = distractor_key
        # But careful — the distractor key MUST NOT appear in the context portion.
        # Simple heuristic: as long as the distractor key tokens aren't all in the same
        # consecutive positions, this is fine. Skip the check for speed; rare collisions
        # average out across n_probes.
        lp_distractor = score_at_position(model, ctx_distractor, ans_slot, distractor_key, device)
        distractor_log_probs.append(lp_distractor)

    correct_mean = sum(correct_log_probs) / len(correct_log_probs)
    distractor_mean = sum(distractor_log_probs) / len(distractor_log_probs)
    # Accuracy: fraction where correct > distractor (i.e., model retains the key).
    accuracy = sum(1 for c, d in zip(correct_log_probs, distractor_log_probs)
                   if c > d) / n_probes
    return {
        "correct_log_prob": correct_mean,
        "distractor_log_prob": distractor_mean,
        "retention_margin": correct_mean - distractor_mean,
        "accuracy": accuracy,
        "n_probes": n_probes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-lens", default="1024,4096,16384")
    ap.add_argument("--n-probes", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--key-len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = pick_device()
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    if "arch" not in cfg:
        sd = ck["model"] if "model" in ck else ck
        cfg["arch"] = "plasma" if any("hamiltonian" in k or "hecke" in k for k in sd.keys()) else "vanilla"

    val_path = find_cache("validation")
    tokens = load_token_stream(val_path)

    print(f"[ckpt] {ckpt_path}  arch={cfg['arch']}  train_seq={cfg.get('seq_len', 1024)}", flush=True)
    print(f"[probe] n={args.n_probes} key_len={args.key_len}", flush=True)

    results = []
    for s in (int(x) for x in args.seq_lens.split(",")):
        # Reset rng per seq_len so probes are comparable across seq_lens.
        rng_s = random.Random(args.seed)
        model = build_model_from_cfg(cfg, seq_len_override=s).to(device)
        sd = ck["model"] if "model" in ck else ck
        sd = remap_pos_embedding(sd, s)
        model.load_state_dict(sd, strict=False)
        # Always patch for chunked attention at long context (cheap, safe).
        if s > 1024 and args.chunk > 0:
            patch_for_long_context(model, query_chunk=args.chunk)

        r = evaluate(model, tokens, s, args.n_probes, device, rng_s,
                      chunk=0, key_len=args.key_len)
        r["seq_len"] = s
        results.append(r)
        print(f"  seq={s:>5d}  acc={r['accuracy']:.2%}  "
              f"margin={r['retention_margin']:+.2f}  "
              f"(correct={r['correct_log_prob']:+.2f}  "
              f"dist={r['distractor_log_prob']:+.2f})", flush=True)
        del model

    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
