# sparsity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


def get_current_p(step: int, p_start: float, p_end: float, t_start: int, t_end: int) -> float:
    """Calculate current p value based on linear annealing schedule."""
    if step < t_start:
        return p_start
    if step >= t_end:
        return p_end
    
    # Linear interpolation
    frac = (step - t_start) / float(max(t_end - t_start, 1))
    return p_start + frac * (p_end - p_start)


class SparsityPenalty:
    """Interface for sparsity penalties."""
    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError


@dataclass
class UniformLpPenalty(SparsityPenalty):
    """
    Uniform Lp penalty: lambda * mean(sum_i (|a_i| + eps)^p).
    
    Supports annealing p from p_start -> p_end over time (Idea 1).
    If p_start == p_end == 1.0, this is standard L1.
    """
    lambda_base: float
    p_start: float
    p_end: float
    anneal_start: int
    anneal_end: int
    eps: float = 1e-6

    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        p = get_current_p(step, self.p_start, self.p_end, self.anneal_start, self.anneal_end)
        
        # Calculate Lp norm: sum((|a| + eps)^p)
        # We add eps inside the power to ensure gradient stability near 0 for p < 1
        sparsity_term = (a.abs() + self.eps).pow(p).sum(dim=1).mean()
        
        loss = self.lambda_base * sparsity_term
        
        return loss, {
            "lambda_mean": float(self.lambda_base),
            "p_current": float(p)
        }


@dataclass
class FrequencyWeightedLpPenalty(SparsityPenalty):
    """
    Frequency-weighted Lp penalty (Idea 2 + Idea 1 Support).
    
    Loss = sum_i lambda_i * (|a_i| + eps)^p
    
    Features:
    1. lambda_i scales with 1/(freq_i)^alpha (penalizes rare features more, or less depending on alpha).
    2. Warmup: uniform weights for first N steps.
    3. P-Annealing: p can transition from 1.0 -> 0.5.
    """
    n_latents: int
    lambda_base: float
    
    # Frequency Weighting Params
    ema_beta: float = 0.99
    fw_eps: float = 1e-4
    alpha: float = 0.5
    warmup_steps: int = 200
    clip_min: float = 1e-4
    clip_max: float = 1e-2
    normalize_mean: bool = True
    
    # P-Annealing Params
    p_start: float = 1.0
    p_end: float = 1.0
    anneal_start: int = 0
    anneal_end: int = 0
    sparsity_eps: float = 1e-6

    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        dev = self.device if self.device is not None else torch.device("cpu")
        # Initialize with a modest assumed activation rate.
        self.ema_p = torch.full((self.n_latents,), 1e-2, device=dev)

    @torch.no_grad()
    def update_ema(self, a: torch.Tensor) -> None:
        # Define "active" as >0 (for ReLU). Detached by no_grad.
        batch_p = (a > 0).float().mean(dim=0)
        self.ema_p.mul_(self.ema_beta).add_((1.0 - self.ema_beta) * batch_p)
        
    def current_lambdas(self, step: int) -> torch.Tensor:
        if step < self.warmup_steps:
            return torch.full_like(self.ema_p, self.lambda_base)

        # Calculate frequency weights
        w = (self.ema_p + self.fw_eps).pow(-self.alpha).detach()

        if self.normalize_mean:
            w = w / w.mean().clamp_min(1e-12)

        lam_raw = self.lambda_base * w

        # Clip (relative or absolute)
        if getattr(self, "clip_relative", True):
            clip_min = self.lambda_base * float(self.clip_min)
            clip_max = self.lambda_base * float(self.clip_max)
        else:
            clip_min = float(self.clip_min)
            clip_max = float(self.clip_max)

        if clip_min > clip_max:
            clip_min, clip_max = clip_max, clip_min

        return lam_raw.clamp(clip_min, clip_max)


    def compute(self, a: torch.Tensor, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
        # 1. Update frequency statistics (Idea 2)
        self.update_ema(a)

        # 2. Get current p value (Idea 1)
        p = get_current_p(step, self.p_start, self.p_end, self.anneal_start, self.anneal_end)

        # 3. Get current lambdas (Idea 2 - Warmup handled inside)
        lam = self.current_lambdas(step)  # [n_latents]

        # 4. Compute Weighted Lp Loss
        # L = mean_batch [ sum_latents ( lambda_i * (|a_i| + eps)^p ) ]
        term = (a.abs() + self.sparsity_eps).pow(p)
        weighted_sum = (term * lam.unsqueeze(0)).sum(dim=1).mean()
        
        stats = {
            "lambda_mean": float(lam.mean().item()),
            "lambda_min": float(lam.min().item()),
            "lambda_max": float(lam.max().item()),
            "p_mean": float(self.ema_p.mean().item()),
            "p_min": float(self.ema_p.min().item()),
            "p_max": float(self.ema_p.max().item()),
            "p_current": float(p)
        }
        return weighted_sum, stats