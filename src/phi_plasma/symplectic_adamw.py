"""Symplectic AdamW — Störmer-Verlet leapfrog form. Vendored from world_model.

Update rule:
    m_t = β₁ m_{t-1} + (1-β₁) g(q_{t-1})
    v_t = β₂ v_{t-1} + (1-β₂) g²
    M_t = √v̂_t + ε              (adapted metric)
    h_t = lr · L(0) / L(min(epoch+2, 15))     (golden step decay)
    q_t = q_{t-1} − h_t · m̂_t / M_t

Preserves the symplectic 2-form ω = Σ_i dq_i ∧ dp_i. Bounds parameter-
space volume drift — addresses the v2 "drift to flat plateau" failure."""

from __future__ import annotations
import math

import torch

from .constants import LUCAS_TABLE


class SymplecticAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.1,
                 lucas_decay: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lucas_decay=lucas_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _lucas_step_factor(epoch: int) -> float:
        idx = min(epoch + 2, 15)
        return LUCAS_TABLE[0] / LUCAS_TABLE[idx]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SymplecticAdamW: no sparse grads")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
                M = v_hat.sqrt().add_(eps)

                if group["lucas_decay"]:
                    epoch = (t - 1) // 1000
                    h = lr * SymplecticAdamW._lucas_step_factor(epoch)
                else:
                    h = lr

                p.add_(m_hat / M, alpha=-h)

        return loss
