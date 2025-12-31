# sparsity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


class SparsityPenalty:
    """Interface for sparsity penalties."""
    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError


@dataclass
class UniformL1Penalty(SparsityPenalty):
    """Classic uniform L1 penalty: lambda * mean(sum_i |a_i|)."""
    lambda_base: float

    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        # a is ReLU => abs(a) == a, but keep abs() for generality
        l = self.lambda_base * a.abs().sum(dim=1).mean()
        return l, {"lambda_mean": float(self.lambda_base)}


@dataclass
class FrequencyWeightedL1Penalty(SparsityPenalty):
    """Frequency-weighted L1: sum_i lambda_i |a_i|, where lambda_i ~ 1/(p_i+eps)^alpha.

    This is your TF-IDF-ish idea in its simplest stable form:
      - We estimate p_i using an EMA over batch "active" rate.
      - We optionally warm up with uniform lambdas, clip lambdas, and normalize mean(lambda_i).
    """
    n_latents: int
    lambda_base: float
    ema_beta: float = 0.99
    eps: float = 1e-4
    alpha: float = 0.5
    warmup_steps: int = 200
    clip_min: float = 1e-4
    clip_max: float = 1e-2
    normalize_mean: bool = True
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        dev = self.device if self.device is not None else torch.device("cpu")
        # Initialize with a modest assumed activation rate (avoid insane early weights).
        self.ema_p = torch.full((self.n_latents,), 1e-2, device=dev)

    @torch.no_grad()
    def update_ema(self, a: torch.Tensor) -> None:
        # Define "active" as >0 (for ReLU). Detached by no_grad.
        batch_p = (a > 0).float().mean(dim=0)
        self.ema_p.mul_(self.ema_beta).add_((1.0 - self.ema_beta) * batch_p)

    def current_lambdas(self, step: int) -> torch.Tensor:
        if step < self.warmup_steps:
            return torch.full_like(self.ema_p, self.lambda_base)

        # weights ~ 1/(p+eps)^alpha
        w = (self.ema_p + self.eps).pow(-self.alpha).detach()
        if self.normalize_mean:
            w = w / w.mean().clamp_min(1e-12)

        lam = (self.lambda_base * w).clamp(self.clip_min, self.clip_max)
        return lam

    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        # Update frequency estimates first (based on current a)
        self.update_ema(a)

        lam = self.current_lambdas(step)  # [n_latents]
        # Weighted L1: mean over examples of sum_i lam_i * |a_i|
        l = (a.abs() * lam.unsqueeze(0)).sum(dim=1).mean()

        stats = {
            "lambda_mean": float(lam.mean().item()),
            "lambda_min": float(lam.min().item()),
            "lambda_max": float(lam.max().item()),
            "p_mean": float(self.ema_p.mean().item()),
            "p_min": float(self.ema_p.min().item()),
            "p_max": float(self.ema_p.max().item()),
        }
        return l, stats
