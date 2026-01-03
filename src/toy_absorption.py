# toy_absorption.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
from tqdm import tqdm

from config import Config
from sae import SparseAutoencoder, normalize_decoder_rows, renorm_decoder_rows_
from sae import SparseAutoencoder, normalize_decoder_rows
from sparsity import UniformLpPenalty, FrequencyWeightedLpPenalty
from matryoshka_utils import _resolve_matryoshka_ms, _decode_prefix
from eval_metrics import avg_max_decoder_cosine_similarity

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
def make_orthonormal_features(
    d_model: int,
    n_feats: int,
    device: torch.device,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return [n_feats, d_model] approximately orthonormal directions.

    Note: `generator` is optional. In most runs we simply control global seeding.
    This argument exists so you can fully decouple RNG streams if desired.
    """
    assert n_feats <= d_model

    X = torch.randn(n_feats, d_model, device=device, generator=generator)
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

    parent_amp = torch.ones(n, device=device)

    has_child = torch.rand(n, device=device) < spec.child_prob
    child_idx = torch.full((n,), -1, device=device, dtype=torch.long)
    if has_child.any():
        child_idx[has_child] = torch.randint(0, C, (int(has_child.sum().item()),), device=device)

    child_amp = torch.zeros(n, device=device)
    child_amp[has_child] = 1.0

    x = parent_amp.unsqueeze(1) * parent.unsqueeze(0)  # [n, d]
    if has_child.any():
        x[has_child] += child_amp[has_child].unsqueeze(1) * children[child_idx[has_child]]

    if spec.noise_std > 0:
        x = x + spec.noise_std * torch.randn_like(x)

    return x.to(torch.float32), child_idx


# ----------------------------
# Penalty selection (Updated)
# ----------------------------

def make_penalty(cfg: Config, device: torch.device):
    astart = cfg.anneal_start_step if cfg.anneal_start_step is not None else int(0.2 * cfg.num_steps)
    aend = cfg.anneal_end_step if cfg.anneal_end_step is not None else int(0.8 * cfg.num_steps)

    if cfg.sparsity in ("l1_uniform", "p_annealing"):
        return UniformLpPenalty(
            lambda_base=cfg.lambda_base,
            p_start=cfg.p_start,
            p_end=cfg.p_end,
            anneal_start=astart,
            anneal_end=aend,
            eps=cfg.sparsity_eps
        )

    if cfg.sparsity in ("l1_freq_weighted", "p_annealing_freq"):
        return FrequencyWeightedLpPenalty(
            n_latents=cfg.n_latents,
            lambda_base=cfg.lambda_base,
            ema_beta=cfg.fw_ema_beta,
            fw_eps=cfg.fw_eps,
            alpha=cfg.fw_alpha,
            warmup_steps=cfg.fw_warmup_steps,
            clip_min=cfg.fw_clip_min,
            clip_max=cfg.fw_clip_max,
            normalize_mean=cfg.fw_normalize_mean,
            p_start=cfg.p_start,
            p_end=cfg.p_end,
            anneal_start=astart,
            anneal_end=aend,
            sparsity_eps=cfg.sparsity_eps,
            device=device,
        )

    if cfg.sparsity == "batchtopk":
        raise ValueError("toy: sparsity='batchtopk' not implemented yet (later commit).")

    raise ValueError(f"Unknown sparsity: {cfg.sparsity}")


# ----------------------------
# Metrics
# ----------------------------

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
def latent_active_freq(a: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
    """Return per-latent activation frequency p_i = P[a_i > thresh] on this batch."""
    if a.numel() == 0:
        return torch.zeros((a.shape[-1],), device=a.device, dtype=torch.float32)
    return (a > thresh).float().mean(dim=0).to(torch.float32)


# ----------------------------
# Matching helpers
# ----------------------------

@torch.no_grad()
def match_latents_to_truth(
    sae: SparseAutoencoder,
    U: torch.Tensor,  # [1+C, d_model] true directions
) -> Tuple[int, torch.Tensor]:
    """Return (parent_latent_index, child_latent_indices[C]).

    This is cosine-based decoder matching (fast, but can pick dead latents).
    We keep it for backward-compat and comparison.
    """
    W = normalize_decoder_rows(sae.W_dec).to(torch.float32)  # [K, d]
    T = normalize_decoder_rows(U).to(torch.float32)          # [1+C, d]
    sims = W @ T.t()                                         # [K, 1+C]
    best_lat_for_truth = sims.argmax(dim=0)                  # [1+C]
    parent_lat = int(best_lat_for_truth[0].item())
    child_lats = best_lat_for_truth[1:].clone()              # [C]
    return parent_lat, child_lats


@torch.no_grad()
def match_latents_to_truth_by_contribution(
    sae: SparseAutoencoder,
    U: torch.Tensor,              # [1+C, d_model] true directions
    a: torch.Tensor,              # [N, K] eval activations
    child_idx: torch.Tensor,      # [N] (-1 or child id)
    *,
    active_thresh: float = 0.0,
    min_active_frac: float = 1e-4,
) -> Tuple[int, torch.Tensor]:
    """Return (parent_latent_index, child_latent_indices[C]) using contribution matching.

    Intuition:
      For a unit truth direction u, the reconstruction projection is:
        <x_hat, u> = <b_dec, u> + sum_i a_i <w_i, u>
      So latent i's contribution to that projection is a_i * <w_i, u>.

    We choose the latent that maximizes mean contribution on the relevant subset:
      - parent: all examples
      - child j: only examples with child_idx == j

    This tends to avoid dead-latent matching and is more faithful to “who is carrying u”.
    """
    # [K, 1+C] dot products <w_i, u_t> (u_t assumed ~unit from QR construction)
    W = sae.W_dec.to(torch.float32)
    T = U.to(torch.float32)
    dots = W @ T.t()

    p = latent_active_freq(a, thresh=active_thresh)  # [K]
    alive = p > float(min_active_frac)

    def pick(scores: torch.Tensor) -> int:
        # Prefer alive latents (otherwise cosine matching often picks dead rows by accident).
        if alive.any():
            scores2 = scores.clone()
            scores2[~alive] = -1e9
            # If all are masked (shouldn't happen if alive.any()), fall back.
            return int(scores2.argmax().item())
        return int(scores.argmax().item())

    # Parent: use all examples
    parent_scores = (a.to(torch.float32) * dots[:, 0].unsqueeze(0)).mean(dim=0)  # [K]
    parent_lat = pick(parent_scores)

    # Children: use only examples where that child is active
    C = int(U.shape[0] - 1)
    child_lats: List[int] = []
    for j in range(C):
        mask = (child_idx == j)
        if mask.any():
            scores_j = (a[mask].to(torch.float32) * dots[:, 1 + j].unsqueeze(0)).mean(dim=0)  # [K]
            child_lats.append(pick(scores_j))
        else:
            # Rare corner case: no examples of a particular child in eval set.
            # Fall back to cosine match for that child direction.
            _, child_lats_cos = match_latents_to_truth(sae, U)
            child_lats.append(int(child_lats_cos[j].item()))

    return parent_lat, torch.tensor(child_lats, device=a.device, dtype=torch.long)


# ----------------------------
# Absorption metrics
# ----------------------------

@torch.no_grad()
def absorption_rate_simple(
    a: torch.Tensor,              # [N, K]
    parent_lat: int,
    child_lats: torch.Tensor,     # [C]
    child_idx: torch.Tensor,      # [N] (-1 or child id)
    thresh: float = 0.0,
) -> float:
    """Latent-based absorption proxy (can be brittle if 'parent' doesn't map to a single latent)."""
    rate, _stats = absorption_rate_simple_with_stats(
        a, parent_lat, child_lats, child_idx, thresh=thresh
    )
    return rate


@torch.no_grad()
def absorption_rate_simple_with_stats(
    a: torch.Tensor,              # [N, K]
    parent_lat: int,
    child_lats: torch.Tensor,     # [C]
    child_idx: torch.Tensor,      # [N] (-1 or child id)
    thresh: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Same as absorption_rate_simple, but returns diagnostic stats.

    Useful to detect silent metric failure modes:
      - denom == 0 because matched child latents are dead / never active
      - parent_lat is wrong so parent_active is almost always false, etc.
    """
    parent_active = (a[:, parent_lat] > thresh)

    mask = (child_idx != -1)
    n_child = int(mask.sum().item())
    if n_child == 0:
        return 0.0, {"abs_n_child": 0.0, "abs_denom": 0.0, "abs_num": 0.0}

    child_ids = child_idx[mask]                 # [M]
    child_lat_for_ex = child_lats[child_ids]    # [M]
    ex_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    child_active = (a[ex_idx, child_lat_for_ex] > thresh)
    parent_suppressed = (~parent_active[mask])

    denom = float(child_active.sum().item())
    if denom == 0.0:
        return 0.0, {
            "abs_n_child": float(n_child),
            "abs_denom": 0.0,
            "abs_num": 0.0,
        }

    num = float((child_active & parent_suppressed).sum().item())
    return float(num / denom), {
        "abs_n_child": float(n_child),
        "abs_denom": float(denom),
        "abs_num": float(num),
    }


@torch.no_grad()
def absorption_rate_parent_proj(
    x_hat: torch.Tensor,          # [N, d_model] reconstruction
    U_parent: torch.Tensor,       # [d_model] true parent direction (unit)
    child_idx: torch.Tensor,      # [N]
    *,
    min_parent_proj: float = 0.5,
) -> float:
    """
    Projection-based absorption proxy:
      - On child-present examples, compute proj = <x_hat, U_parent>.
      - Count fraction where proj < min_parent_proj.

    This directly measures whether the *reconstruction* has "lost" the parent component,
    without relying on matching a single latent.
    """
    mask = (child_idx != -1)
    if mask.sum().item() == 0:
        return 0.0

    proj = (x_hat @ U_parent)  # [N]
    proj_child = proj[mask]
    num = (proj_child < min_parent_proj).sum().item()
    den = proj_child.numel()

    return float(num / max(den, 1))


@torch.no_grad()
def parent_projection_diagnostics(
    x_hat: torch.Tensor,          # [N, d_model]
    U_parent: torch.Tensor,       # [d_model] unit
    child_idx: torch.Tensor,      # [N]
) -> Dict[str, float]:
    """Useful projection diagnostics (overall + conditional on child presence)."""
    proj = (x_hat @ U_parent).to(torch.float32)  # [N]
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=proj.device)
    qv = torch.quantile(proj, qs).to(torch.float32)

    mask_child = (child_idx != -1)
    mask_nochild = (child_idx == -1)

    out: Dict[str, float] = {
        "parent_proj_min": float(proj.min().item()),
        "parent_proj_mean": float(proj.mean().item()),
        "parent_proj_q01": float(qv[0].item()),
        "parent_proj_q05": float(qv[1].item()),
        "parent_proj_q50": float(qv[2].item()),
        "parent_proj_q95": float(qv[3].item()),
        "parent_proj_q99": float(qv[4].item()),
    }

    if mask_child.any():
        pc = proj[mask_child]
        out.update(
            {
                "parent_proj_mean_child": float(pc.mean().item()),
                "parent_proj_min_child": float(pc.min().item()),
            }
        )
    else:
        out.update({"parent_proj_mean_child": float("nan"), "parent_proj_min_child": float("nan")})

    if mask_nochild.any():
        pn = proj[mask_nochild]
        out.update(
            {
                "parent_proj_mean_nochild": float(pn.mean().item()),
                "parent_proj_min_nochild": float(pn.min().item()),
            }
        )
    else:
        out.update({"parent_proj_mean_nochild": float("nan"), "parent_proj_min_nochild": float("nan")})

    return out


@torch.no_grad()
def child_parent_leakage_stats(
    sae: SparseAutoencoder,
    U_parent: torch.Tensor,      # [d_model] unit
    child_lats: torch.Tensor,    # [C]
) -> Dict[str, float]:
    """How much the matched child latents' decoder vectors align with the parent direction.

    We report dot(w_child, U_parent) (not cosine) because this is the *actual* parent-component
    amplitude contributed by that latent to recon (per unit activation).
    """
    if child_lats.numel() == 0:
        return {
            "child_parent_leak_mean": 0.0,
            "child_parent_leak_min": 0.0,
            "child_parent_leak_max": 0.0,
            "child_parent_leak_p95": 0.0,
        }

    leaks = (sae.W_dec[child_lats].to(torch.float32) @ U_parent.to(torch.float32)).to(torch.float32)  # [C]
    p95 = torch.quantile(leaks, 0.95).item() if leaks.numel() > 1 else leaks.max().item()
    return {
        "child_parent_leak_mean": float(leaks.mean().item()),
        "child_parent_leak_min": float(leaks.min().item()),
        "child_parent_leak_max": float(leaks.max().item()),
        "child_parent_leak_p95": float(p95),
    }


@torch.no_grad()
def parent_latent_suppression(
    sae: SparseAutoencoder,
    a: torch.Tensor,             # [N, K]
    U_parent: torch.Tensor,      # [d_model] unit
    parent_lat: int,
    child_idx: torch.Tensor,     # [N]
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Measure how much a *single chosen parent latent* stops carrying parent on child examples.

    This is still a “single-latent” diagnostic, but it uses contribution (a_i * <w_i, U_parent>)
    rather than a boolean active threshold.

    Returns:
      - parent_lat_dot: <w_parent_lat, U_parent>
      - parent_lat_contrib_mean_child / nochild
      - parent_lat_contrib_suppression: 1 - child/nochild (if nochild != 0)
    """
    dot = float((sae.W_dec[parent_lat].to(torch.float32) @ U_parent.to(torch.float32)).item())
    contrib = a[:, parent_lat].to(torch.float32) * float(dot)  # [N]

    mask_child = (child_idx != -1)
    mask_nochild = (child_idx == -1)

    out: Dict[str, float] = {"parent_lat_dot": float(dot)}

    if mask_child.any():
        out["parent_lat_contrib_mean_child"] = float(contrib[mask_child].mean().item())
    else:
        out["parent_lat_contrib_mean_child"] = float("nan")

    if mask_nochild.any():
        base = float(contrib[mask_nochild].mean().item())
        out["parent_lat_contrib_mean_nochild"] = float(base)
        child_mean = out["parent_lat_contrib_mean_child"]
        if abs(base) > eps and child_mean == child_mean:  # not NaN
            out["parent_lat_contrib_suppression"] = float(1.0 - (float(child_mean) / base))
        else:
            out["parent_lat_contrib_suppression"] = float("nan")
    else:
        out["parent_lat_contrib_mean_nochild"] = float("nan")
        out["parent_lat_contrib_suppression"] = float("nan")

    return out


# ----------------------------
# L0 measurement on toy
# ----------------------------

@torch.no_grad()
def measure_l0_on_toy(
    sae: SparseAutoencoder,
    U: torch.Tensor,
    spec: ToyTreeSpec,
    *,
    device: torch.device,
    batches: int,
    batch_size: int,
    thresh: float,
) -> float:
    vals = []
    for _ in range(batches):
        x, _ = sample_tree_batch(U, spec, n=batch_size, device=device)
        a = sae(x).a
        vals.append(l0_mean(a, thresh=thresh))
    return float(sum(vals) / max(len(vals), 1))


# ----------------------------
# Training (one run)
# ----------------------------

def _train_one(
    cfg: Config,
    *,
    U: torch.Tensor,
    spec: ToyTreeSpec,
    device: torch.device,
    steps: int,
    show_progress: bool,
    seed_offset: int = 1,
) -> SparseAutoencoder:
    # Re-seed so lambda sweeps are comparable (same init & data randomness per candidate)
    train_seed = int(spec.seed) + int(seed_offset)
    torch.manual_seed(train_seed)
    random.seed(train_seed)

    sae = SparseAutoencoder(
        d_model=spec.d_model,
        n_latents=cfg.n_latents,
        tied_weights=cfg.tied_weights,
        activation=cfg.activation,
        device=device,
        dtype=torch.float32,
    )

    penalty = make_penalty(cfg, device=device)

    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sae.train()
    bs = cfg.batch_size

    it = range(steps)
    pbar = (
        tqdm(it, desc=f"toy train [{cfg.sparsity}] lam={cfg.lambda_base:.2e}", dynamic_ncols=True)
        if show_progress else it
    )

    for step in pbar:
        x, _ = sample_tree_batch(U, spec, n=bs, device=device)
        out = sae(x)
        a = out.a

        # Recon path
        if cfg.recon_variant == "standard":
            x_hat = out.x_hat
            l_recon = torch.mean((x - x_hat) ** 2)

        elif cfg.recon_variant == "matryoshka":
            ms = _resolve_matryoshka_ms(cfg)
            recon_terms = []
            for m in ms:
                x_hat_m = _decode_prefix(sae, a, m)
                recon_terms.append(torch.mean((x - x_hat_m) ** 2))
            if getattr(cfg, "matryoshka_recon_agg", "mean") == "sum":
                l_recon = torch.stack(recon_terms).sum()
            else:
                l_recon = torch.stack(recon_terms).mean()
        else:
            raise ValueError(cfg.recon_variant)

        l_sparse, sparse_stats = penalty.compute(a, step=step + 1)
        loss = l_recon + l_sparse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if getattr(cfg, "decoder_unit_norm", False):
            renorm_decoder_rows_(sae, eps=getattr(cfg, "decoder_unit_norm_eps", 1e-8))
            
        if show_progress and (step % max(1, cfg.log_every) == 0):
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                lam=f"{sparse_stats.get('lambda_mean', cfg.lambda_base):.2e}",
                p=f"{sparse_stats.get('p_current', 1.0):.2f}"
            )

    if show_progress:
        # pbar.close() # handled by context or loop end
        pass

    return sae


def _default_lambda_grid(center: float) -> List[float]:
    """
    A simple log grid around center.
    If center is 1e-3, this yields ~[1e-5 .. 1e-2] (capped to avoid trivial collapse in toy).
    """
    base = float(center)
    if base <= 0:
        base = 1e-3
    factors = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
    # Cap to a conservative max for toy so we don't waste sweep budget on obvious collapse.
    grid = sorted({base * f for f in factors})
    return [x for x in grid if x <= 1e-2] or [base]


def _pick_lambda_via_sweep(
    cfg: Config,
    *,
    U: torch.Tensor,
    spec: ToyTreeSpec,
    device: torch.device,
    seed_offset: int,
) -> float:
    """Offline sweep: train runs at candidate lambdas, pick lambda closest to target_l0."""
    assert cfg.target_l0 is not None
    target = float(cfg.target_l0)

    # Candidates: either provided externally (via config) or a reasonable default grid
    cand: Optional[List[float]] = getattr(cfg, "toy_lambda_grid", None)
    if cand is None:
        cand = _default_lambda_grid(cfg.lambda_base)

    # IMPORTANT: The sweep must run for the full duration (cfg.num_steps) 
    # because P-Annealing and Frequency Weighting change dynamics over time.
    sweep_steps = cfg.num_steps 
    
    l0_eval_bs = max(512, cfg.batch_size)
    l0_eval_batches = int(getattr(cfg, "toy_l0_eval_batches", max(10, cfg.calib_batches)))

    best_lam = float(cand[0])
    best_err = float("inf")
    best_fvu = float("inf")

    print(f"[toy] Sweeping lambdas {cand} to hit L0={target}...")

    for lam in cand:
        cfg_s = replace(cfg, lambda_base=float(lam))
        
        sae_s = _train_one(
            cfg_s,
            U=U,
            spec=spec,
            device=device,
            steps=sweep_steps,
            show_progress=False,
            seed_offset=seed_offset,
        )

        with torch.no_grad():
            # Evaluate L0 on a few fresh batches
            cur_l0 = measure_l0_on_toy(
                sae_s, U, spec,
                device=device,
                batches=l0_eval_batches,
                batch_size=l0_eval_bs,
                thresh=cfg.active_threshold,
            )

            # Lightweight FVU estimate (tie-break)
            x_tmp, _ = sample_tree_batch(U, spec, n=spec.n_eval, device=device)
            out_tmp = sae_s(x_tmp)
            if cfg_s.recon_variant == "matryoshka":
                ms = _resolve_matryoshka_ms(cfg_s)
                x_hat_tmp = _decode_prefix(sae_s, out_tmp.a, ms[-1])
            else:
                x_hat_tmp = out_tmp.x_hat
            cur_fvu = fvu(x_tmp, x_hat_tmp)

        err = abs(cur_l0 - target)
        print(f"   lam={lam:.1e} -> L0={cur_l0:.2f} (err={err:.2f})")

        # Choose minimal L0 error; tie-break by lower FVU
        if (err < best_err) or (abs(err - best_err) < 1e-9 and cur_fvu < best_fvu):
            best_err = err
            best_fvu = cur_fvu
            best_lam = float(lam)

    return best_lam


# ----------------------------
# Main toy runner
# ----------------------------

def _toy_debug_enabled(cfg: Config) -> bool:
    # No Config changes required: set env var TOY_DEBUG=1 to print extra diagnostics.
    if os.environ.get("TOY_DEBUG", "").strip() not in ("", "0", "false", "False"):
        return True
    return bool(getattr(cfg, "toy_debug", False))


def run_toy(cfg: Config, spec: ToyTreeSpec, device: torch.device) -> Dict[str, float]:
    # Global seeding for U (toy directions)
    torch.manual_seed(spec.seed)
    random.seed(spec.seed)

    U = make_orthonormal_features(spec.d_model, 1 + spec.n_children, device=device)
    U_parent = U[0].to(torch.float32)  # [d], unit from QR construction

    # Seed offset used for training RNG stream (init + batch sampling).
    seed_offset = int(getattr(cfg, "toy_train_seed_offset", 1))

    # --- UPDATED: Check for all new sparsity modes ---
    do_sweep = (
        cfg.calibrate_l0
        and (cfg.target_l0 is not None)
        and (cfg.sparsity in ("l1_uniform", "l1_freq_weighted", "p_annealing", "p_annealing_freq"))
    )

    if do_sweep:
        best_lam = _pick_lambda_via_sweep(cfg, U=U, spec=spec, device=device, seed_offset=seed_offset)
        cfg = replace(cfg, lambda_base=float(best_lam))
        print(f"[toy sweep] {cfg.recon_variant}+{cfg.sparsity}: picked lambda_base={cfg.lambda_base:.3e} for target_l0={cfg.target_l0}")

    sae = _train_one(
        cfg,
        U=U,
        spec=spec,
        device=device,
        steps=max(1, cfg.num_steps),
        show_progress=True,
        seed_offset=seed_offset,
    )

    # Eval
    sae.eval()
    with torch.no_grad():
        x_eval, child_idx = sample_tree_batch(U, spec, n=spec.n_eval, device=device)
        out = sae(x_eval)
        a_eval = out.a

        if cfg.recon_variant == "matryoshka":
            ms = _resolve_matryoshka_ms(cfg)
            x_hat = _decode_prefix(sae, a_eval, ms[-1])
        else:
            x_hat = out.x_hat

    # -------------------------
    # Decoder cosine matching (old)
    # -------------------------
    parent_lat_cos, child_lats_cos = match_latents_to_truth(sae, U)
    abs_rate_latent_cos, abs_stats_cos = absorption_rate_simple_with_stats(
        a_eval, parent_lat_cos, child_lats_cos, child_idx, thresh=cfg.active_threshold
    )

    # -------------------------
    # Contribution matching (recommended)
    # -------------------------
    parent_lat_contrib, child_lats_contrib = match_latents_to_truth_by_contribution(
        sae,
        U,
        a_eval,
        child_idx,
        active_thresh=cfg.active_threshold,
        min_active_frac=float(getattr(cfg, "toy_min_active_frac", 1e-4)),
    )
    abs_rate_latent_contrib, abs_stats_contrib = absorption_rate_simple_with_stats(
        a_eval, parent_lat_contrib, child_lats_contrib, child_idx, thresh=cfg.active_threshold
    )

    # Projection-based absorption (new, recommended)
    # NOTE: With parent always-on, this is typically ~0 unless training collapses.
    abs_rate_proj = absorption_rate_parent_proj(
        x_hat, U_parent, child_idx, min_parent_proj=float(getattr(cfg, "toy_min_parent_proj", 0.5))
    )

    # Parent projection diagnostics (overall / child vs nochild)
    proj_diag = parent_projection_diagnostics(x_hat, U_parent, child_idx)

    # Bias contribution along parent direction
    bdec_parent = float((sae.b_dec.to(torch.float32) @ U_parent.to(torch.float32)).item())

    # Activity frequencies for matched latents (helps detect dead-latent matching)
    p = latent_active_freq(a_eval, thresh=cfg.active_threshold)
    parent_p_cos = float(p[parent_lat_cos].item()) if p.numel() > 0 else 0.0
    parent_p_contrib = float(p[parent_lat_contrib].item()) if p.numel() > 0 else 0.0

    # Child latent activeness summary (cosine-matched vs contrib-matched)
    def _child_freq_stats(child_lats: torch.Tensor) -> Dict[str, float]:
        if child_lats.numel() == 0:
            return {"child_lat_p_mean": 0.0, "child_lat_p_min": 0.0, "child_lat_p_max": 0.0, "child_lat_unique": 0.0}
        pp = p[child_lats].to(torch.float32)
        return {
            "child_lat_p_mean": float(pp.mean().item()),
            "child_lat_p_min": float(pp.min().item()),
            "child_lat_p_max": float(pp.max().item()),
            "child_lat_unique": float(torch.unique(child_lats).numel()),
        }

    child_freq_cos = _child_freq_stats(child_lats_cos)
    child_freq_contrib = _child_freq_stats(child_lats_contrib)

    # Leakage: how much the matched child latents carry parent direction (dot(w_child, U_parent))
    leak_cos = child_parent_leakage_stats(sae, U_parent, child_lats_cos)
    leak_contrib = child_parent_leakage_stats(sae, U_parent, child_lats_contrib)

    # Parent-latent suppression diagnostics (single-latent, but contribution-based)
    parent_sup_cos = parent_latent_suppression(sae, a_eval, U_parent, parent_lat_cos, child_idx)
    parent_sup_contrib = parent_latent_suppression(sae, a_eval, U_parent, parent_lat_contrib, child_idx)

    # Optional debug prints (set TOY_DEBUG=1)
    if _toy_debug_enabled(cfg):
        # Mirror your earlier temporary printout, but under no_grad and with cleaner formatting.
        proj = (x_hat @ U_parent).to(torch.float32)
        qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=proj.device)
        qv = torch.quantile(proj, qs)
        print(
            "[toy debug] parent proj:",
            float(proj.min().item()),
            float(proj.mean().item()),
            qv.detach().cpu(),
        )
        print("[toy debug] b_dec·U_parent =", bdec_parent)
        print(
            "[toy debug] parent_lat (cos, contrib) =",
            parent_lat_cos,
            parent_lat_contrib,
            "| parent_p (cos, contrib) =",
            f"{parent_p_cos:.3e}",
            f"{parent_p_contrib:.3e}",
        )
        print(
            "[toy debug] abs_denom (cos, contrib) =",
            abs_stats_cos.get("abs_denom", float("nan")),
            abs_stats_contrib.get("abs_denom", float("nan")),
        )

    return {
        # Core metrics
        "fvu": fvu(x_eval, x_hat),
        "l0_mean": l0_mean(a_eval, thresh=cfg.active_threshold),
        # "avg_max_decoder_cos": avg_max_decoder_cos(sae.W_dec),
        "avg_max_decoder_cos": avg_max_decoder_cosine_similarity(sae.W_dec)["avg_max_decoder_cos"],

        # Absorption metrics
        "absorption_rate_latent": abs_rate_latent_cos,                  # backward compat (cosine match)
        "absorption_rate_latent_contrib": abs_rate_latent_contrib,      # recommended (contrib match)
        "absorption_rate_proj": abs_rate_proj,                          # recon-level parent drop

        # Latent matches (indices)
        "parent_latent": float(parent_lat_cos),                         # backward compat
        "parent_latent_contrib": float(parent_lat_contrib),

        # Absorption metric diagnostics (denoms, etc.)
        "absorption_latent_denom": float(abs_stats_cos.get("abs_denom", 0.0)),
        "absorption_latent_denom_contrib": float(abs_stats_contrib.get("abs_denom", 0.0)),
        "absorption_latent_num": float(abs_stats_cos.get("abs_num", 0.0)),
        "absorption_latent_num_contrib": float(abs_stats_contrib.get("abs_num", 0.0)),
        "absorption_n_child": float(abs_stats_cos.get("abs_n_child", 0.0)),

        # Projection diagnostics
        **{k: float(v) for k, v in proj_diag.items()},
        "b_dec_dot_u_parent": float(bdec_parent),

        # Match quality diagnostics (are matched latents alive?)
        "parent_lat_p_active": float(parent_p_cos),
        "parent_lat_p_active_contrib": float(parent_p_contrib),
        
        "child_lat_p_mean": float(child_freq_cos["child_lat_p_mean"]),
        "child_lat_p_min": float(child_freq_cos["child_lat_p_min"]),
        "child_lat_p_max": float(child_freq_cos["child_lat_p_max"]),
        "child_lat_unique": float(child_freq_cos["child_lat_unique"]),
        
        "child_lat_p_mean_contrib": float(child_freq_contrib["child_lat_p_mean"]),
        "child_lat_p_min_contrib": float(child_freq_contrib["child_lat_p_min"]),
        "child_lat_p_max_contrib": float(child_freq_contrib["child_lat_p_max"]),
        "child_lat_unique_contrib": float(child_freq_contrib["child_lat_unique"]),

        # Leakage stats: do child latents' decoder vectors contain parent component?
        "child_parent_leak_mean": float(leak_cos["child_parent_leak_mean"]),
        "child_parent_leak_min": float(leak_cos["child_parent_leak_min"]),
        "child_parent_leak_max": float(leak_cos["child_parent_leak_max"]),
        "child_parent_leak_p95": float(leak_cos["child_parent_leak_p95"]),
        "child_parent_leak_mean_contrib": float(leak_contrib["child_parent_leak_mean"]),
        "child_parent_leak_min_contrib": float(leak_contrib["child_parent_leak_min"]),
        "child_parent_leak_max_contrib": float(leak_contrib["child_parent_leak_max"]),
        "child_parent_leak_p95_contrib": float(leak_contrib["child_parent_leak_p95"]),

        # Parent-latent suppression (single-latent contribution-based diagnostic)
        "parent_lat_dot": float(parent_sup_cos.get("parent_lat_dot", float("nan"))),
        "parent_lat_contrib_mean_child": float(parent_sup_cos.get("parent_lat_contrib_mean_child", float("nan"))),
        "parent_lat_contrib_mean_nochild": float(parent_sup_cos.get("parent_lat_contrib_mean_nochild", float("nan"))),
        "parent_lat_contrib_suppression": float(parent_sup_cos.get("parent_lat_contrib_suppression", float("nan"))),

        "parent_lat_dot_contrib": float(parent_sup_contrib.get("parent_lat_dot", float("nan"))),
        "parent_lat_contrib_mean_child_contrib": float(parent_sup_contrib.get("parent_lat_contrib_mean_child", float("nan"))),
        "parent_lat_contrib_mean_nochild_contrib": float(parent_sup_contrib.get("parent_lat_contrib_mean_nochild", float("nan"))),
        "parent_lat_contrib_suppression_contrib": float(parent_sup_contrib.get("parent_lat_contrib_suppression", float("nan"))),

        # Run metadata
        "lambda_base": float(cfg.lambda_base),
        "p_start": float(cfg.p_start),
        "p_end": float(cfg.p_end),
        "fw_alpha": float(cfg.fw_alpha) if "freq" in cfg.sparsity else 0.0,
    }


def main() -> None:
    import tyro

    cfg = tyro.cli(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = ToyTreeSpec(
        d_model=256,
        n_children=8,
        n_train=50_000,
        n_eval=10_000,
        child_prob=0.3,
        noise_std=0.02,
        seed=cfg.seed,
    )

    results: Dict[str, Dict[str, float]] = {}

    def run_one(tag: str, recon_variant: str, sparsity: str, **kwargs):
        # Helper to override params (e.g., enable p-annealing)
        cfg_i = replace(cfg, recon_variant=recon_variant, sparsity=sparsity, **kwargs)
        print(f"--- Running {tag} ---")
        try:
            results[tag] = run_toy(cfg=cfg_i, spec=spec, device=device)
        except Exception as e:
            print(f"FAILED {tag}: {e}")
            import traceback
            traceback.print_exc()

    # 1. Baseline
    run_one("standard_l1_uniform", "standard", "l1_uniform")
    
    # 2. Idea 1: P-Annealing (1.0 -> 0.5)
    run_one("standard_anneal", "standard", "p_annealing", p_start=1.0, p_end=0.5)
    
    # 3. Idea 2: Freq Weighted
    run_one("standard_freq", "standard", "l1_freq_weighted")
    
    # 4. Combined
    run_one("standard_combined", "standard", "p_annealing_freq", p_start=1.0, p_end=0.5)

    # 5. Matryoshka Comparisons
    run_one("matryoshka_l1_uniform", "matryoshka", "l1_uniform")
    run_one("matryoshka_combined", "matryoshka", "p_annealing_freq", p_start=1.0, p_end=0.5)

    out_path = Path(cfg.out_dir) / f"toy_absorption_{cfg.run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()