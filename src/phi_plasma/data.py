"""WikiText-2 data loader — reuses existing token cache at
~/Desktop/phi_training_workspace/PHI_AEON/.token_cache/

Cache file: wikitext__wikitext-2-raw-v1__train__v28657__s256.pt
Format probed during smoke test; loader handles list-of-tensors, single
tensor, or dict format."""

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


CACHE_DIR = Path.home() / "Desktop" / "phi_training_workspace" / "PHI_AEON" / ".token_cache"


def find_cache(split: str = "train", seq_in_cache: int = 256) -> Path:
    pattern = f"wikitext__wikitext-2-raw-v1__{split}__v28657__s{seq_in_cache}.pt"
    p = CACHE_DIR / pattern
    if not p.exists():
        raise FileNotFoundError(f"token cache missing: {p}")
    return p


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
                 stride: int | None = None) -> tuple[DataLoader, DataLoader, dict]:
    train_path = find_cache("train")
    val_path = find_cache("validation")
    train_tokens = load_token_stream(train_path)
    val_tokens = load_token_stream(val_path)

    train_ds = WikiTextWindows(train_tokens, seq_len, stride)
    val_ds = WikiTextWindows(val_tokens, seq_len, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True,
                            pin_memory=False)

    meta = {
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "seq_len": seq_len,
        "vocab_inferred_max": int(max(train_tokens.max().item(),
                                       val_tokens.max().item())) + 1,
    }
    return train_loader, val_loader, meta
