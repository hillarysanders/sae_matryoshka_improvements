#!/usr/bin/env python3
# toy_absorption.py
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import torch
from tqdm import tqdm

from config import Config
from sae import SparseAutoencoder, normalize_decoder_rows
from sparsity import UniformL1Penalty, FrequencyWeightedL1Penalty
from stoch_avail import StochasticAvailability


ToyMethod = Literal["baseline_l1", "freq_weighted_l1", "stoch_avail_l1"]


# ----------------------------
# Toy data: parent + children
# ----------------------------

@dataclass
class ToyTreeSpec:
    """Simple 2-level tree: one parent feature + C child features.

    Each example contains:
      - parent always on (amplitude ~1)
      - with probability child_prob, one random child is on (amplitude ~1)
      - optionally: a small amount of gaussian noise
    """
    d_model: int = 256
    n_children: int = 8
    n_train: int = 50_000
    n_eval: int = 10_000
    child_prob: float = 0.3
    noise_std: float = 0.02
    seed: int = 0


@torch.no_grad()
def make_orthonormal_features(d_model: int, n_feats: int, device: torch.device) -> torch.Tensor:
    """Return [n_feats, d_model] approximately orthonormal directions."""
    X = torch.randn(n_feats, d_model, device=device)
    # QR on (d_model x n_feats) gives orthonormal columns; transpose back
    Q, _ = torch.linalg.qr(X.t(), mode="reduced")  # [d, n_feats]
    return Q.t().contiguous()  # [n_feats, d]


@torch.no_grad()
def sample_tree_batch(
    U: torch.Tensor,  # [1 + C, d_model] feature directions
    spec: ToyTreeSpec,
    n: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate x and binary indicators of which child was active.

    Returns:
      x: [n, d_model]
      child_idx: [n] with -1 meaning no child active, else in [0..C-1]
    """
    parent = U[0]            # [d]
    children = U[1:]         # [C, d]
    C = children.shape[0]

    # parent always active with amplitude ~1
    parent_amp = torch.ones(n, device=device)

    # choose whether a child is active, and which
    has_child = torch.rand(n, device=device) < spec.child_prob
    child_idx = torch.full((n,), -1, device=device, dtype=torch.long)
    child_idx[has_child] = torch.randint(0, C, (int(has_child.sum().item()),), device=device)

    child_amp = torch.zeros(n, device=device)
    child_amp[has_child] = 1.0

    # Build x
    x = parent_amp.unsqueeze(1) * parent.unsqueeze(0)  # [n, d]
    if has_child.any():
        x[has_child] += child_amp[has_child].unsqueeze(1) * children[child_idx[has_child]]

    # Add noise
    if spec.noise_std > 0:
        x = x + spec.noise_std * torch.randn_like(x)

    return x.to(torch.float32), child_idx


# ----------------------------
# Training loop (toy)
# ----------------------------

def make_penalty(cfg: Config, method: ToyMethod, device: torch.device):
    if method == "baseline_l1":
        return UniformL1Penalty(lambda_base=cfg.lambda_base)
    if method == "freq_weighted_l1":
        return FrequencyWeightedL1Penalty(
            n_latents=cfg.n_latents,
            lambda_base=cfg.lambda_base,
            ema_beta=cfg.fw_ema_beta,
            eps=cfg.fw_eps,
            alpha=cfg.fw_alpha,
            warmup_steps=cfg.fw_warmup_steps,
            clip_min=cfg.fw_clip_min,
            clip_max=cfg.fw_clip_max,
            normalize_mean=cfg.fw_normalize_mean,
            device=device,
        )
    if method == "stoch_avail_l1":
        return UniformL1Penalty(lambda_base=cfg.lambda_base)
    raise ValueError(method)


@torch.no_grad()
def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x_mean = x.mean(dim=0, keepdim=True)
    num = torch.mean((x - x_hat) ** 2).item()
    den = torch.mean((x - x_mean) ** 2).item()
    return float(num / max(den, 1e-12))


@torch.no_grad()
def l0_mean(a: torch.Tensor, thresh: float = 0.0) -> float:
    return float((a > thresh).float().sum(dim=1).mean().item())


@torch.no_grad()
def avg_max_decoder_cos(W_dec: torch.Tensor, chunk: int = 2048) -> float:
    W = normalize_decoder_rows(W_dec).to(torch.float32)
    K = W.shape[0]
    if K < 2:
        return 0.0
    WT = W.t().contiguous()
    max_per = torch.full((K,), -1.0, device=W.device)
    for i0 in range(0, K, chunk):
        i1 = min(K, i0 + chunk)
        sims = W[i0:i1] @ WT
        rows = torch.arange(i0, i1, device=W.device)
        sims[torch.arange(i1 - i0, device=W.device), rows] = -1e9
        max_per[i0:i1] = sims.max(dim=1).values
    return float(max_per.clamp(-1, 1).mean().item())


@torch.no_grad()
def match_latents_to_truth(
    sae: SparseAutoencoder,
    U: torch.Tensor,  # [1+C, d_model] true directions
) -> Tuple[int, torch.Tensor]:
    """Return (parent_latent_index, child_latent_indices[C]).

    We match each true feature direction to the closest decoder row by cosine similarity.
    This is greedy and simple; sufficient for the toy demo.
    """
    W = normalize_decoder_rows(sae.W_dec).to(torch.float32)  # [K, d]
    T = normalize_decoder_rows(U).to(torch.float32)          # [1+C, d]

    # sims: [K, 1+C]
    sims = W @ T.t()
    # For each truth feature, pick best latent
    best_lat_for_truth = sims.argmax(dim=0)  # [1+C]

    parent_lat = int(best_lat_for_truth[0].item())
    child_lats = best_lat_for_truth[1:].clone()  # [C]
    return parent_lat, child_lats


@torch.no_grad()
def absorption_rate_simple(
    a: torch.Tensor,              # [N, K]
    parent_lat: int,
    child_lats: torch.Tensor,     # [C]
    child_idx: torch.Tensor,      # [N] (-1 or child id)
    thresh: float = 0.0,
) -> float:
    """Absorption rate proxy on toy data.

    We say absorption happens on an example if:
      - some child is active (child_idx != -1)
      - that child's matched latent is active
      - but the parent latent is NOT active

    Return fraction of child-active examples where parent is suppressed.
    """
    parent_active = (a[:, parent_lat] > thresh)

    # For each example with child, check matched child latent
    mask = (child_idx != -1)
    if mask.sum().item() == 0:
        return 0.0

    child_ids = child_idx[mask]  # [M]
    child_lat_for_ex = child_lats[child_ids]  # [M]
    ex_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    child_active = (a[ex_idx, child_lat_for_ex] > thresh)
    parent_suppressed = (~parent_active[mask])

    # Only count cases where child latent is actually active (so "child present" is represented)
    denom = child_active.sum().item()
    if denom == 0:
        return 0.0

    num = (child_active & parent_suppressed).sum().item()
    return float(num / denom)


def run_toy(method: ToyMethod, cfg: Config, spec: ToyTreeSpec, device: torch.device) -> Dict[str, float]:
    torch.manual_seed(spec.seed)
    random.seed(spec.seed)

    # Ground-truth directions: [parent + children]
    U = make_orthonormal_features(spec.d_model, 1 + spec.n_children, device=device)

    # SAE
    sae = SparseAutoencoder(
        d_model=spec.d_model,
        n_latents=cfg.n_latents,
        tied_weights=cfg.tied_weights,
        activation=cfg.activation,
        device=device,
        dtype=torch.float32,
    )

    penalty = make_penalty(cfg, method, device=device)

    stoch = None
    if method == "stoch_avail_l1":
        stoch = StochasticAvailability(
            n_latents=cfg.n_latents,
            p_min=cfg.sa_p_min,
            gamma=cfg.sa_gamma,
            inverted=cfg.sa_inverted,
            shared_across_batch=cfg.sa_shared_across_batch,
            device=device,
        )

    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train
    sae.train()
    bs = cfg.batch_size

    steps = max(1, cfg.num_steps)
    pbar = tqdm(range(steps), desc=f"toy train [{method}]", dynamic_ncols=True)
    for step in pbar:
        x, _ = sample_tree_batch(U, spec, n=bs, device=device)

        out = sae(x)
        a = out.a

        # stochastic availability masks only reconstruction
        if method == "stoch_avail_l1":
            assert stoch is not None
            a_m, _ = stoch.mask(a)
            x_hat = sae.decode(a_m)
        else:
            x_hat = out.x_hat

        l_recon = torch.mean((x - x_hat) ** 2)
        l_sparse, sparse_stats = penalty.compute(a, step=step + 1)
        loss = l_recon + l_sparse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, cfg.log_every) == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                recon=f"{l_recon.item():.3e}",
                sparse=f"{l_sparse.item():.3e}",
                lam=f"{sparse_stats.get('lambda_mean', cfg.lambda_base):.2e}",
            )

    # Eval
    sae.eval()
    x_eval, child_idx = sample_tree_batch(U, spec, n=spec.n_eval, device=device)
    out = sae(x_eval)
    a_eval = out.a
    x_hat = out.x_hat

    # Match latents to truth + compute absorption
    parent_lat, child_lats = match_latents_to_truth(sae, U)
    abs_rate = absorption_rate_simple(
        a_eval, parent_lat, child_lats, child_idx, thresh=cfg.active_threshold
    )

    return {
        "fvu": fvu(x_eval, x_hat),
        "l0_mean": l0_mean(a_eval, thresh=cfg.active_threshold),
        "avg_max_decoder_cos": avg_max_decoder_cos(sae.W_dec),
        "absorption_rate": abs_rate,
        "parent_latent": float(parent_lat),
        "mean_lambda": float(cfg.lambda_base),
    }


def main() -> None:
    import tyro

    cfg = tyro.cli(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep toy small + fast by default
    spec = ToyTreeSpec(
        d_model=256,
        n_children=8,
        n_train=50_000,
        n_eval=10_000,
        child_prob=0.3,
        noise_std=0.02,
        seed=cfg.seed,
    )

    # For toy, interpret cfg.num_steps as steps (not tokens); batch_size is examples/step.
    results = {}
    for method in ["baseline_l1", "freq_weighted_l1", "stoch_avail_l1"]:
        results[method] = run_toy(method=method, cfg=cfg, spec=spec, device=device)

    out_path = Path(cfg.out_dir) / f"toy_absorption_{cfg.run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
