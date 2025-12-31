# eval_metrics.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch

from config import Config
from data import TokenBatcher
from tl_activations import get_activations, flatten_activations
from sae import normalize_decoder_rows, recon_loss


# -------------------------
# 1) Decoder cosine metric (paper-style)
# -------------------------

@torch.no_grad()
def avg_max_decoder_cosine_similarity(
    W_dec: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> Dict[str, float]:
    """Compute the paper-style 'max decoder cosine similarity'.

    For each latent i, compute max_{j != i} cos(w_i, w_j), then average across i.

    This mirrors the Matryoshka paper's 'decoder cosine similarity' / 'feature composition' proxy
    much more closely than random pair sampling.

    Args:
        W_dec: [K, d_model] decoder matrix.
        chunk_size: compute similarity in blocks to avoid O(K^2) memory.

    Returns:
        Dict with:
            avg_max_decoder_cos: mean_i max_{j!=i} cos(w_i, w_j)
            p95_max_decoder_cos: 95th percentile over i
            max_decoder_cos: max over i
    """
    W = normalize_decoder_rows(W_dec).to(torch.float32)  # [K, d]
    K = W.shape[0]
    if K < 2:
        return {"avg_max_decoder_cos": 0.0, "p95_max_decoder_cos": 0.0, "max_decoder_cos": 0.0}

    # We'll compute max similarity for each row without ever materializing KxK.
    max_per_i = torch.full((K,), -1.0, device=W.device, dtype=torch.float32)

    # Precompute transpose for matmul blocks.
    WT = W.t().contiguous()  # [d, K]

    for i0 in range(0, K, chunk_size):
        i1 = min(K, i0 + chunk_size)
        block = W[i0:i1]  # [B, d]
        sims = block @ WT  # [B, K]

        # Exclude self-similarity by setting diagonal to -inf for rows in this block.
        row_idx = torch.arange(i0, i1, device=W.device)
        sims[torch.arange(i1 - i0, device=W.device), row_idx] = -1e9

        block_max = sims.max(dim=1).values  # [B]
        max_per_i[i0:i1] = block_max

    # Clamp because we used -1e9 to mask diagonal.
    max_per_i = max_per_i.clamp(min=-1.0, max=1.0)

    return {
        "avg_max_decoder_cos": float(max_per_i.mean().item()),
        "p95_max_decoder_cos": float(torch.quantile(max_per_i, 0.95).item()),
        "max_decoder_cos": float(max_per_i.max().item()),
    }


# -------------------------
# 2) FVU + L0 on an eval stream
# -------------------------

@torch.no_grad()
def eval_on_stream(
    cfg: Config,
    sae,
    *,
    model,
    hook_name: str,
    num_batches: int,
    active_threshold: float,
) -> Dict[str, float]:
    """Evaluate reconstruction + sparsity on a small stream of activations.

    Computes:
      - recon_mse
      - fvu (fraction variance unexplained)
      - l0_mean (avg active latents per token)
      - dead/rare fractions (optional, cheap once you have activations)

    Note: This does a fresh dataset stream (independent of training iterator).
    """
    device = model.cfg.device

    batcher = TokenBatcher(
        tokenizer=model.tokenizer,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        local_text_path=cfg.local_text_path,
        hf_dataset=cfg.hf_dataset,
        hf_dataset_config=cfg.hf_dataset_config,
        hf_split=cfg.hf_split,
        hf_text_field=cfg.hf_text_field,
    )
    it = iter(batcher)

    # Accumulators
    n_total = 0
    sum_sq_err = 0.0
    sum_sq_baseline = 0.0

    # For x_mean we do a streaming mean over activations.
    x_mean = None

    # L0
    sum_active = 0.0

    # Dead/rare feature stats (on eval stream)
    # We'll accumulate active counts per feature.
    feat_active_counts = None

    for _ in range(num_batches):
        tokens = next(it)
        acts = get_activations(model, tokens, hook_name)
        x = flatten_activations(acts).to(torch.float32)  # [N, d]
        N = x.shape[0]

        # Update streaming mean of x
        if x_mean is None:
            x_mean = x.mean(dim=0)
            n_total = N
        else:
            # combine means
            new_total = n_total + N
            x_mean = (x_mean * n_total + x.sum(dim=0)) / float(new_total)
            n_total = new_total

        out = sae(x)
        x_hat = out.x_hat
        a = out.a

        # recon mse numerator
        sq_err = torch.mean((x - x_hat) ** 2).item()
        sum_sq_err += sq_err * N  # weight by N

        # L0
        active = (a > active_threshold).float()
        sum_active += active.sum().item()  # counts all actives over [N, K]

        # per-feature activity (dead/rare)
        if feat_active_counts is None:
            feat_active_counts = active.sum(dim=0).cpu()
        else:
            feat_active_counts += active.sum(dim=0).cpu()

    assert x_mean is not None
    # Baseline variance: E||x - mean(x)||^2
    # Re-run a tiny pass for baseline variance with the final x_mean (cheap).
    # (We could also accumulate second moments, but this is simpler and stable.)
    batcher2 = TokenBatcher(
        tokenizer=model.tokenizer,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        device=device,
        local_text_path=cfg.local_text_path,
        hf_dataset=cfg.hf_dataset,
        hf_dataset_config=cfg.hf_dataset_config,
        hf_split=cfg.hf_split,
        hf_text_field=cfg.hf_text_field,
    )
    it2 = iter(batcher2)
    n2 = 0
    for _ in range(num_batches):
        tokens = next(it2)
        acts = get_activations(model, tokens, hook_name)
        x = flatten_activations(acts).to(torch.float32)
        N = x.shape[0]
        n2 += N
        sum_sq_baseline += torch.mean((x - x_mean) ** 2).item() * N

    recon_mse = sum_sq_err / float(n_total)
    baseline_mse = sum_sq_baseline / float(n2)
    fvu = recon_mse / max(baseline_mse, 1e-12)

    # L0_mean: average active features per token-row (N rows of activations)
    # sum_active counts active entries over [N,K]; divide by N for avg actives per row.
    l0_mean = (sum_active / float(n_total))

    # Dead/rare from eval stream
    p = (feat_active_counts / float(n_total)).numpy()  # activation frequency per feature
    dead_frac = float((p < 1e-6).mean())
    rare_frac = float((p < 1e-4).mean())

    return {
        "recon_mse": float(recon_mse),
        "baseline_mse": float(baseline_mse),
        "fvu": float(fvu),
        "l0_mean": float(l0_mean),
        "dead_frac_p_lt_1e_6": float(dead_frac),
        "rare_frac_p_lt_1e_4": float(rare_frac),
    }


# -------------------------
# 3) Lambda calibration to match target L0
# -------------------------

@torch.no_grad()
def calibrate_lambda_for_target_l0(
    cfg: Config,
    sae,
    *,
    model,
    hook_name: str,
    target_l0: float,
    num_batches: int = 5,
    active_threshold: float = 0.0,
    max_rounds: int = 8,
    tol: float = 1.0,
) -> float:
    """Tune cfg.lambda_base (multiplicatively) to hit a target average L0.

    This is a lightweight calibration pass (no training). It measures L0 under the current SAE.
    For a paper-faithful comparison you typically calibrate lambda during training; but this
    function is still useful for quick iteration and for methods where sparsity is stable.

    Returns:
        new_lambda_base
    """
    lam = cfg.lambda_base

    # Multiplicative search in log space.
    # If L0 too high -> increase lambda
    # If L0 too low  -> decrease lambda
    for _ in range(max_rounds):
        out = eval_on_stream(
            cfg, sae, model=model, hook_name=hook_name,
            num_batches=num_batches, active_threshold=active_threshold
        )
        l0 = out["l0_mean"]
        if abs(l0 - target_l0) <= tol:
            break

        # heuristic multiplicative step
        ratio = (l0 / max(target_l0, 1e-6))
        # if ratio=2 => too many actives => increase lambda by ~sqrt(2) to avoid overshooting
        lam *= float(ratio ** 0.5)
        lam = float(max(lam, 1e-8))

        cfg = replace(cfg, lambda_base=lam)

    return lam


# -------------------------
# Entry point used by train.py
# -------------------------

@torch.no_grad()
def run_quick_eval(cfg: Config, sae, *, model=None, hook_name: Optional[str] = None) -> Dict[str, float]:
    """Paper-aligned quick eval.

    Computes:
      - AvgMax decoder cosine similarity (paper-style)
      - FVU + L0 + dead/rare on a small eval stream (if model+hook_name provided)
    """
    out: Dict[str, float] = {}

    out.update(avg_max_decoder_cosine_similarity(sae.W_dec))

    if model is not None and hook_name is not None:
        out.update(
            eval_on_stream(
                cfg, sae, model=model, hook_name=hook_name,
                num_batches=cfg.eval_num_batches,
                active_threshold=cfg.active_threshold,
            )
        )

    return out
