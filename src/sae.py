# sae.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn


@dataclass
class SAEOutput:
    a: torch.Tensor      # [n, n_latents]
    x_hat: torch.Tensor  # [n, d_model]


class SparseAutoencoder(nn.Module):
    """A simple ReLU SAE: x -> a -> x_hat.

    Notes:
      - Untied default (W_enc, W_dec independent).
      - Optional tied weights (W_enc == W_dec.T) for fewer params.
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        tied_weights: bool = False,
        activation: Literal["relu"] = "relu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.tied_weights = tied_weights
        self.activation = activation

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # Decoder bias is useful because TL activations are not mean-zero.
        self.b_dec = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
        self.b_enc = nn.Parameter(torch.zeros(n_latents, **factory_kwargs))

        self.W_dec = nn.Parameter(torch.empty(n_latents, d_model, **factory_kwargs))
        if tied_weights:
            self.W_enc = None
        else:
            self.W_enc = nn.Parameter(torch.empty(d_model, n_latents, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier-ish init scaled to keep activations reasonable.
        nn.init.normal_(self.W_dec, std=1.0 / math.sqrt(self.d_model))
        if self.W_enc is not None:
            nn.init.normal_(self.W_enc, std=1.0 / math.sqrt(self.d_model))
        nn.init.zeros_(self.b_dec)
        nn.init.zeros_(self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x - self.b_dec
        if self.tied_weights:
            pre = x0 @ self.W_dec.t() + self.b_enc
        else:
            assert self.W_enc is not None
            pre = x0 @ self.W_enc + self.b_enc

        if self.activation == "relu":
            return torch.relu(pre)
        raise ValueError(f"Unknown activation: {self.activation}")

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        a = self.encode(x)
        x_hat = self.decode(a)
        return SAEOutput(a=a, x_hat=x_hat)


def recon_loss(x: torch.Tensor, x_hat: torch.Tensor, kind: Literal["mse"] = "mse") -> torch.Tensor:
    if kind == "mse":
        return torch.mean((x - x_hat) ** 2)
    raise ValueError(f"Unknown recon loss: {kind}")


@torch.no_grad()
def normalize_decoder_rows(W_dec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return row-normalized decoder vectors for cosine similarity computations."""
    norms = W_dec.norm(dim=1, keepdim=True).clamp_min(eps)
    return W_dec / norms


@torch.no_grad()
def renorm_decoder_rows_(
    sae: "SparseAutoencoder",
    eps: float = 1e-8,
) -> torch.Tensor:
    """In-place: force each decoder row (latent direction) to unit norm.

    Returns:
        norms_pre: [K] row norms before renorm (useful for logging).
    """
    W = sae.W_dec  # [K, d_model]
    norms_pre = W.norm(dim=1)  # [K]
    W.div_(norms_pre.clamp_min(eps).unsqueeze(1))
    return norms_pre
