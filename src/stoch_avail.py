# stoch_avail.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch


@torch.no_grad()
def make_p_schedule(n_latents: int, p_min: float, gamma: float, device: torch.device) -> torch.Tensor:
    """Monotone decreasing p_i from 1 -> p_min."""
    i = torch.arange(n_latents, device=device, dtype=torch.float32)
    t = i / max(n_latents - 1, 1)
    p = p_min + (1.0 - p_min) * (1.0 - t).pow(gamma)
    return p.clamp(min=p_min, max=1.0)


@dataclass
class StochasticAvailability:
    n_latents: int
    p_min: float = 0.2
    gamma: float = 2.0
    inverted: bool = True
    shared_across_batch: bool = False
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        dev = self.device if self.device is not None else torch.device("cpu")
        self.p = make_p_schedule(self.n_latents, self.p_min, self.gamma, dev)

    def mask(self, a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply stochastic availability mask to latent activations.

        Args:
            a: [n, n_latents] latent activations

        Returns:
            a_masked: masked activations (same shape)
            stats: diagnostics
        """
        p = self.p.to(a.device)

        if self.shared_across_batch:
            m = torch.bernoulli(p).unsqueeze(0)  # [1, K]
            m = m.expand(a.shape[0], -1)
        else:
            m = torch.bernoulli(p.unsqueeze(0).expand(a.shape[0], -1))

        if self.inverted:
            a_masked = a * (m / p.clamp_min(1e-8))
        else:
            a_masked = a * m

        stats = {
            "sa_p_mean": float(p.mean().item()),
            "sa_p_min": float(p.min().item()),
            "sa_p_max": float(p.max().item()),
            "sa_mask_mean": float(m.mean().item()),
        }
        return a_masked, stats
