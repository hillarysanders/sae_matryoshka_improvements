# batchtopk.py
from __future__ import annotations

import math
from typing import Dict, Literal, Tuple, Optional

import torch

TieBreak = Literal["none", "random"]


def apply_batchtopk(
    a: torch.Tensor,          # [..., K]
    *,
    k_per_row: float,
    tie_break: "TieBreak" = "none",
    eps: float = 1e-12,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BatchTopK gating across the whole batch (supports arbitrary leading dims).

    Interprets the last dimension as K and flattens all leading dims into N "rows".
    Keeps the top (N * k_per_row) entries in `a` across all N*K positions and zeroes the rest.

    This enforces *average* L0 ≈ k_per_row per row, but allows per-row variation.

    Args:
        a: activations with shape [..., K] (often [B, T, K] or [N, K]).
        k_per_row: target average number of active latents per row (token).
        tie_break:
            - "none": keep all entries >= threshold (may keep > target due to ties)
            - "random": break ties at the threshold to hit the target count exactly
        eps: numerical guard around comparisons to tau.
        generator: optional torch.Generator for deterministic tie breaking.

    Returns:
        a_sparse: masked activations, same shape as `a`
        stats: diagnostics (tau, keep_target, keep_actual, l0_mean, density)
    """
    if a.ndim < 1:
        raise ValueError(f"apply_batchtopk expects at least 1D tensor, got shape {tuple(a.shape)}")

    K = int(a.shape[-1])
    leading_shape = a.shape[:-1]
    N = math.prod(leading_shape) if len(leading_shape) > 0 else 1

    if N == 0 or K == 0:
        return a, {
            "bt_tau": 0.0,
            "bt_keep_target": 0.0,
            "bt_keep_actual": 0.0,
            "bt_l0_mean": 0.0,
            "bt_density": 0.0,
        }

    # Total number of entries to keep across ALL N*K positions
    keep_target = int(round(float(N) * float(k_per_row)))
    keep_target = max(0, min(keep_target, N * K))

    if keep_target == 0:
        a_sparse = torch.zeros_like(a)
        return a_sparse, {
            "bt_tau": float("inf"),
            "bt_keep_target": float(keep_target),
            "bt_keep_actual": 0.0,
            "bt_l0_mean": 0.0,
            "bt_density": 0.0,
        }

    if keep_target == N * K:
        return a, {
            "bt_tau": 0.0,
            "bt_keep_target": float(keep_target),
            "bt_keep_actual": float(N * K),
            "bt_l0_mean": float(K),
            "bt_density": 1.0,
        }

    # Flatten once; do all selection logic on a detached view under no_grad
    flat = a.reshape(-1)
    flat_det = flat.detach()

    with torch.no_grad():
        # Threshold tau = keep_target-th largest value.
        # Prefer kthvalue on -flat (avoids materializing top-k values).
        try:
            tau = -torch.kthvalue(-flat_det, k=keep_target).values  # scalar tensor
        except RuntimeError:
            # Fallback if kthvalue isn't supported/fast on a given backend
            topv, _ = torch.topk(flat_det, k=keep_target, largest=True, sorted=False)
            tau = topv.min()

        if tie_break == "none":
            # Keep everything >= tau (allowing slight eps slack)
            keep_idx = torch.nonzero(flat_det >= (tau - eps), as_tuple=False).squeeze(1)

        elif tie_break == "random":
            # Keep everything strictly > tau, then randomly choose among ties near tau to hit target exactly.
            gt_idx = torch.nonzero(flat_det > (tau + eps), as_tuple=False).squeeze(1)
            keep_gt = int(gt_idx.numel())
            need = keep_target - keep_gt

            if need <= 0:
                # Rare: if numerical eps makes gt already enough, just take top-k exactly.
                _, keep_idx = torch.topk(flat_det, k=keep_target, largest=True, sorted=False)
            else:
                close_mask = (flat_det >= (tau - eps)) & (flat_det <= (tau + eps))
                close_idx = torch.nonzero(close_mask, as_tuple=False).squeeze(1)

                # Exclude any already in gt_idx by filtering with a strict band; (gt_idx are > tau+eps)
                # So close_idx are disjoint from gt_idx by construction.

                if int(close_idx.numel()) >= need:
                    perm = torch.randperm(close_idx.numel(), device=a.device, generator=generator)
                    chosen = close_idx[perm[:need]]
                    keep_idx = torch.cat([gt_idx, chosen], dim=0)
                else:
                    # If too few "close" values (or weird numeric issues), fall back to exact topk indices.
                    _, keep_idx = torch.topk(flat_det, k=keep_target, largest=True, sorted=False)

        else:
            raise ValueError(f"Unknown tie_break={tie_break!r}")

    # Build output without allocating a full boolean mask:
    # write kept entries into a zero tensor.
    a_sparse = torch.zeros_like(a)
    a_sparse_flat = a_sparse.reshape(-1)
    a_sparse_flat[keep_idx] = flat[keep_idx]

    keep_actual = int(keep_idx.numel())
    stats = {
        "bt_tau": float(tau.detach().item()),
        "bt_keep_target": float(keep_target),
        "bt_keep_actual": float(keep_actual),
        "bt_l0_mean": float(keep_actual / float(N)),       # avg kept per "row"/token
        "bt_density": float(keep_actual / float(N * K)),   # fraction kept overall
    }
    return a_sparse, stats
