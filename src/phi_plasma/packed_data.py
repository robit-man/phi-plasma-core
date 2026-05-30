"""Packed-token dataset loader for large-corpus training.

The production path for meaningful pretraining/distillation is:

1. build binary token streams with ``scripts/build_token_shards.py``
2. point a training config at the resulting directory with ``data_source: packed``
3. train with torchrun/DDP

The binary format is deliberately simple: ``train.bin`` and ``validation.bin``
are contiguous token-id arrays, with dtype and tokenizer metadata in
``meta.json``. This avoids loading multi-GB token tensors into RAM on each rank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


class PackedTokenWindows(Dataset):
    """Sliding windows over a memory-mapped token stream."""

    def __init__(self, path: str | Path, seq_len: int, stride: int | None = None,
                 dtype: str = "uint32"):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.stride = int(stride if stride is not None else seq_len)
        if dtype not in DTYPES:
            raise ValueError(f"unsupported packed dtype {dtype!r}; expected one of {sorted(DTYPES)}")
        self.dtype_name = dtype
        self.tokens = np.memmap(self.path, dtype=DTYPES[dtype], mode="r")
        self.n_windows = max(0, (len(self.tokens) - self.seq_len - 1) // self.stride)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = int(idx) * self.stride
        # Copy before torch conversion so the tensors are writable/owning and
        # DataLoader workers do not retain a view into the memmap page.
        x = np.asarray(self.tokens[start:start + self.seq_len], dtype=np.int64).copy()
        y = np.asarray(self.tokens[start + 1:start + self.seq_len + 1], dtype=np.int64).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def load_meta(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"packed dataset metadata missing: {meta_path}")
    return json.loads(meta_path.read_text())


def split_path(data_dir: str | Path, meta: dict[str, Any], split: str) -> Path:
    data_dir = Path(data_dir)
    split_info = meta.get("splits", {}).get(split)
    if isinstance(split_info, dict) and split_info.get("path"):
        return data_dir / split_info["path"]
    fallback = "validation.bin" if split in ("validation", "val") else f"{split}.bin"
    return data_dir / fallback


def make_packed_loaders(data_dir: str | Path, seq_len: int, batch_size: int,
                        num_workers: int = 0, stride: int | None = None,
                        distributed: bool = False, rank: int = 0,
                        world_size: int = 1, pin_memory: bool = False,
                        seed: int = 0) -> tuple[DataLoader, DataLoader, dict]:
    meta = load_meta(data_dir)
    dtype = meta.get("dtype", "uint32")
    train_path = split_path(data_dir, meta, "train")
    val_path = split_path(data_dir, meta, "validation")
    if not train_path.exists():
        raise FileNotFoundError(f"packed train split missing: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"packed validation split missing: {val_path}")

    train_ds = PackedTokenWindows(train_path, seq_len=seq_len, stride=stride, dtype=dtype)
    val_ds = PackedTokenWindows(val_path, seq_len=seq_len, stride=stride, dtype=dtype)

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

    vocab_size = int(meta.get("vocab_size", 0) or 0)
    observed_max = int(meta.get("max_token_id", vocab_size - 1 if vocab_size else 0))
    info = {
        "data_source": "packed",
        "data_dir": str(Path(data_dir).resolve()),
        "dtype": dtype,
        "tokenizer_name": meta.get("tokenizer_name"),
        "tokenizer_path": meta.get("tokenizer_path"),
        "train_tokens": int(meta.get("splits", {}).get("train", {}).get("tokens", len(train_ds.tokens))),
        "val_tokens": int(meta.get("splits", {}).get("validation", {}).get("tokens", len(val_ds.tokens))),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "seq_len": seq_len,
        "batch_size_per_process": batch_size,
        "world_size": world_size,
        "vocab_inferred_max": max(vocab_size, observed_max + 1),
    }
    return train_loader, val_loader, info
