"""End-to-end smoke: PlasmaCore + Vanilla forward + backward both work."""

import torch

from phi_plasma.constants import VOCAB_SIZE, D_HIDDEN
from phi_plasma.plasma_core import PlasmaCore
from phi_plasma.vanilla_baseline import VanillaTransformer
from phi_plasma.losses import combined_loss


def test_plasma_forward_backward():
    torch.manual_seed(0)
    model = PlasmaCore(vocab_size=512, d_hidden=D_HIDDEN, n_layers=2,
                        n_heads=11, head_dim=32, seq_len=32)
    idx = torch.randint(0, 512, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, 512)
    losses = combined_loss(model, logits, idx, step=100, warmup=50, lam_max=1e-3)
    losses["total"].backward()
    # At least one parameter should have non-zero grad.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients computed"
    assert any(g.abs().sum().item() > 0 for g in grads), "all grads are zero"


def test_vanilla_forward_backward():
    torch.manual_seed(0)
    model = VanillaTransformer(vocab_size=512, d_hidden=D_HIDDEN, n_layers=2,
                                n_heads=11, head_dim=32, seq_len=32)
    idx = torch.randint(0, 512, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, 512)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), idx.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum().item() > 0 for g in grads)


def test_plasma_param_count_reasonable():
    model = PlasmaCore(vocab_size=28657, d_hidden=352, n_layers=6,
                        n_heads=11, head_dim=32, seq_len=1024)
    n = sum(p.numel() for p in model.parameters())
    # Sanity: should be in the 10–30M range with tied embeddings.
    assert 5e6 < n < 50e6, f"param count out of expected range: {n/1e6:.2f}M"
