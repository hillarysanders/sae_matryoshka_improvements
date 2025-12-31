# batchtopk.py
from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch


TieBreak = Literal["none", "random"]


@torch.no_grad()
def apply_batchtopk(
    a: torch.Tensor,          # [N, K]
    *,
    k_per_row: float,
    tie_break: TieBreak = "none",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BatchTopK gating across the whole batch.

    Keeps the top (N * k_per_row) entries in `a` across all N*K positions.
    Zeroes out everything else.

    This enforces *average* L0 ≈ k_per_row per row (token), but allows per-row
    variation depending on where the largest activations are.

    Args:
        a: nonnegative activations [N, K] (ReLU output).
        k_per_row: target average number of active latents per row.
        tie_break:
            - "none": keep all entries >= threshold (may keep slightly > target due to ties)
            - "random": break ties at the threshold to hit the target count exactly
        eps: numerical guard for topk thresholding.

    Returns:
        a_sparse: masked activations, same shape as `a`
        stats: diagnostics (tau, keep_target, keep_actual, l0_mean)
    """
    if a.ndim != 2:
        raise ValueError(f"apply_batchtopk expects a [N,K] tensor, got shape {tuple(a.shape)}")

    N, K = a.shape
    if N == 0 or K == 0:
        return a, {
            "bt_tau": 0.0,
            "bt_keep_target": 0.0,
            "bt_keep_actual": 0.0,
            "bt_l0_mean": 0.0,
        }

    # Total number of entries to keep
    keep_target = int(round(float(N) * float(k_per_row)))
    keep_target = max(0, min(keep_target, N * K))

    if keep_target == 0:
        a_sparse = torch.zeros_like(a)
        return a_sparse, {
            "bt_tau": float("inf"),
            "bt_keep_target": float(keep_target),
            "bt_keep_actual": 0.0,
            "bt_l0_mean": 0.0,
        }

    if keep_target == N * K:
        # keep everything
        return a, {
            "bt_tau": 0.0,
            "bt_keep_target": float(keep_target),
            "bt_keep_actual": float(N * K),
            "bt_l0_mean": float(K),
        }

    flat = a.reshape(-1)

    # Find the keep_target-th largest value = threshold tau
    # We can do this by taking topk and looking at the smallest among them.
    # topk is O(M log keep_target) and OK for now; later we can optimize if needed.
    topv, _ = torch.topk(flat, k=keep_target, largest=True, sorted=False)
    tau = topv.min().item()

    # Initial mask: keep everything >= tau
    mask = (a >= (tau - eps))

    keep_actual = int(mask.sum().item())

    if tie_break == "random" and keep_actual != keep_target:
        # If we kept too many due to ties at tau, randomly drop some of the tau-equal entries.
        # If we kept too few (rare; eps issues), we can promote a few tau-equal entries.
        # We only adjust among entries very close to tau.
        close = (a >= (tau - eps)) & (a <= (tau + eps))
        close_idx = torch.nonzero(close.reshape(-1), as_tuple=False).squeeze(1)

        if close_idx.numel() > 0:
            mask_flat = mask.reshape(-1)

            if keep_actual > keep_target:
                # Need to drop (keep_actual - keep_target) from the close set that are currently kept
                kept_close = close_idx[mask_flat[close_idx]]
                drop = keep_actual - keep_target
                if kept_close.numel() > 0 and drop > 0:
                    perm = torch.randperm(kept_close.numel(), device=a.device)
                    to_drop = kept_close[perm[:drop]]
                    mask_flat[to_drop] = False
            elif keep_actual < keep_target:
                # Need to add (keep_target - keep_actual) from the close set that are currently not kept
                not_kept_close = close_idx[~mask_flat[close_idx]]
                add = keep_target - keep_actual
                if not_kept_close.numel() > 0 and add > 0:
                    perm = torch.randperm(not_kept_close.numel(), device=a.device)
                    to_add = not_kept_close[perm[:add]]
                    mask_flat[to_add] = True

            mask = mask_flat.reshape(N, K)
            keep_actual = int(mask.sum().item())

    a_sparse = a * mask.to(dtype=a.dtype)

    stats = {
        "bt_tau": float(tau),
        "bt_keep_target": float(keep_target),
        "bt_keep_actual": float(keep_actual),
        "bt_l0_mean": float(keep_actual / float(N)),
        "bt_density": float(keep_actual / float(N * K)),
    }
    return a_sparse, stats
