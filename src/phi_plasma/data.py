"""WikiText-2 data loader.

The original experiments used a pre-tokenized WikiText-2 cache, but the cache
location is machine-specific. This loader now resolves the cache in this order:

1. explicit config value passed to ``make_loaders(data_cache_dir=...)``
2. ``PHI_PLASMA_CACHE_DIR``
3. the legacy PHI_AEON cache location
4. ``./.token_cache`` in this repository

If the requested cache files are missing, it can bootstrap a byte-tokenized
WikiText-2 raw cache into the resolved directory. Byte tokenization is not the
same tokenizer used for the paper numbers; it is a dependency-free fallback
that keeps training/evaluation operational on a fresh checkout.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[2]
LEGACY_CACHE_DIR = (
    Path.home() / "Desktop" / "phi_training_workspace" / "PHI_AEON" / ".token_cache"
)
LOCAL_CACHE_DIR = ROOT / ".token_cache"
CACHE_ENV = "PHI_PLASMA_CACHE_DIR"
DEFAULT_VOCAB_SIZE = 28657
DEFAULT_SEQ_IN_CACHE = 256
WIKITEXT_RAW_URLS = {
    "train": "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw",
    "validation": "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.valid.raw",
}


def cache_filename(split: str, vocab_size: int = DEFAULT_VOCAB_SIZE,
                   seq_in_cache: int = DEFAULT_SEQ_IN_CACHE) -> str:
    return f"wikitext__wikitext-2-raw-v1__{split}__v{vocab_size}__s{seq_in_cache}.pt"


def candidate_cache_dirs(cache_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if cache_dir:
        candidates.append(Path(cache_dir).expanduser())
    env_cache = os.environ.get(CACHE_ENV)
    if env_cache:
        candidates.append(Path(env_cache).expanduser())
    candidates.extend([LEGACY_CACHE_DIR, LOCAL_CACHE_DIR])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def resolve_cache_dir(cache_dir: str | Path | None = None,
                      vocab_size: int = DEFAULT_VOCAB_SIZE,
                      seq_in_cache: int = DEFAULT_SEQ_IN_CACHE) -> Path:
    candidates = candidate_cache_dirs(cache_dir)
    expected = cache_filename("train", vocab_size, seq_in_cache)
    for candidate in candidates:
        if (candidate / expected).exists():
            return candidate
    return candidates[0]


def find_cache(split: str = "train", seq_in_cache: int = DEFAULT_SEQ_IN_CACHE,
               cache_dir: str | Path | None = None,
               vocab_size: int = DEFAULT_VOCAB_SIZE,
               auto_prepare: bool = True) -> Path:
    resolved = resolve_cache_dir(cache_dir, vocab_size, seq_in_cache)
    path = resolved / cache_filename(split, vocab_size, seq_in_cache)
    if path.exists():
        return path
    if auto_prepare:
        ensure_cache_files(
            cache_dir=resolved,
            vocab_size=vocab_size,
            seq_in_cache=seq_in_cache,
            splits=(split,),
        )
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in candidate_cache_dirs(cache_dir))
    raise FileNotFoundError(
        f"token cache missing for split={split!r}: {path}. "
        f"Searched: {searched}. Set {CACHE_ENV} or data_cache_dir to an existing cache."
    )


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def raw_text_path(cache_dir: Path, split: str) -> Path:
    suffix = "valid" if split == "validation" else split
    return cache_dir / "raw" / f"wiki.{suffix}.raw"


def byte_tokenize(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def build_cache_file(cache_dir: Path, split: str,
                     vocab_size: int = DEFAULT_VOCAB_SIZE,
                     seq_in_cache: int = DEFAULT_SEQ_IN_CACHE) -> Path:
    if split not in WIKITEXT_RAW_URLS:
        raise ValueError(f"unsupported WikiText split for bootstrap: {split}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_text_path(cache_dir, split)
    if not raw_path.exists():
        download_file(WIKITEXT_RAW_URLS[split], raw_path)
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    tokens = byte_tokenize(text)
    out_path = cache_dir / cache_filename(split, vocab_size, seq_in_cache)
    payload = {
        "input_ids": tokens,
        "meta": {
            "dataset": "wikitext-2-raw-v1",
            "split": split,
            "tokenizer": "byte_fallback",
            "vocab_size_requested": vocab_size,
            "vocab_size_observed": 256,
            "source_url": WIKITEXT_RAW_URLS[split],
        },
    }
    torch.save(payload, out_path)
    return out_path


def ensure_cache_files(cache_dir: str | Path | None = None,
                       vocab_size: int = DEFAULT_VOCAB_SIZE,
                       seq_in_cache: int = DEFAULT_SEQ_IN_CACHE,
                       splits: tuple[str, ...] = ("train", "validation"),
                       auto_prepare: bool = True) -> dict:
    resolved = resolve_cache_dir(cache_dir, vocab_size, seq_in_cache)
    paths: dict[str, str] = {}
    missing = []
    for split in splits:
        path = resolved / cache_filename(split, vocab_size, seq_in_cache)
        if not path.exists():
            missing.append(split)
        paths[split] = str(path)

    if missing and not auto_prepare:
        raise FileNotFoundError(
            f"token cache missing for splits={missing} in {resolved}. "
            f"Set {CACHE_ENV} or data_cache_dir to an existing cache."
        )

    for split in missing:
        path = build_cache_file(resolved, split, vocab_size, seq_in_cache)
        paths[split] = str(path)

    return {
        "cache_dir": str(resolved),
        "paths": paths,
        "prepared_splits": missing,
    }


def load_token_stream(path: Path) -> torch.Tensor:
    """Load cache and flatten to a 1-D int64 tensor of tokens."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        flat = obj.flatten().long()
    elif isinstance(obj, list):
        flat = torch.cat([t.flatten().long() for t in obj])
    elif isinstance(obj, dict):
        for key in ("input_ids", "tokens", "ids"):
            if key in obj:
                v = obj[key]
                if isinstance(v, torch.Tensor):
                    flat = v.flatten().long()
                elif isinstance(v, list):
                    flat = torch.cat([t.flatten().long() for t in v])
                break
        else:
            raise KeyError(f"dict cache: no known key. Keys: {list(obj.keys())}")
    else:
        raise TypeError(f"unknown cache type: {type(obj)}")
    return flat


class WikiTextWindows(Dataset):
    """Contiguous sliding windows over the flattened token stream."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, stride: int | None = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.n_windows = max(0, (len(tokens) - seq_len - 1) // self.stride)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def make_loaders(seq_len: int, batch_size: int, num_workers: int = 0,
                 stride: int | None = None, distributed: bool = False,
                 rank: int = 0, world_size: int = 1,
                 pin_memory: bool = False, seed: int = 0,
                 cache_dir: str | Path | None = None,
                 vocab_size: int = DEFAULT_VOCAB_SIZE,
                 seq_in_cache: int = DEFAULT_SEQ_IN_CACHE,
                 auto_prepare: bool = True
                 ) -> tuple[DataLoader, DataLoader, dict]:
    cache_meta = ensure_cache_files(
        cache_dir=cache_dir,
        vocab_size=vocab_size,
        seq_in_cache=seq_in_cache,
        auto_prepare=auto_prepare,
    )
    train_path = Path(cache_meta["paths"]["train"])
    val_path = Path(cache_meta["paths"]["validation"])
    train_tokens = load_token_stream(train_path)
    val_tokens = load_token_stream(val_path)

    train_ds = WikiTextWindows(train_tokens, seq_len, stride)
    val_ds = WikiTextWindows(val_tokens, seq_len, stride)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, seed=seed, drop_last=False,
        )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    meta = {
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "seq_len": seq_len,
        "batch_size_per_process": batch_size,
        "world_size": world_size,
        "cache_dir": cache_meta["cache_dir"],
        "cache_prepared_splits": cache_meta["prepared_splits"],
        "vocab_inferred_max": int(max(train_tokens.max().item(),
                                       val_tokens.max().item())) + 1,
    }
    return train_loader, val_loader, meta
