# calibration.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torch

from config import Config
from data import TokenBatcher
from tl_activations import get_activations, flatten_activations


@torch.no_grad()
def calibrate_theta_for_target_l0(
    cfg: Config,
    sae,
    *,
    model,
    hook_name: str,
    target_l0: float,
    num_batches: Optional[int] = None,
    samples_per_batch: int = 200_000,
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate a global threshold theta such that:
        E_row[ #(a > theta) ] ≈ target_l0

    We approximate by sampling activation entries uniformly from a large stream
    and picking theta as the (1 - target_l0/K) quantile of sampled entries.
    """
    if num_batches is None:
        # Use your existing calib_batches as default if present; else fall back.
        num_batches = getattr(cfg, "calib_batches", 10)

    device = model.cfg.device
    K = int(cfg.n_latents)
    k = float(target_l0)

    if k <= 0:
        return float("inf"), {"bt_theta": float("inf"), "bt_q": 1.0}
    if k >= K:
        return 0.0, {"bt_theta": 0.0, "bt_q": 0.0}

    q = 1.0 - (k / float(K))
    q = float(min(max(q, 0.0), 1.0))

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

    samples_cpu: List[torch.Tensor] = []
    total_entries_seen = 0

    for _ in range(int(num_batches)):
        tokens = next(it)
        acts = get_activations(model, tokens, hook_name)
        x = flatten_activations(acts).to(torch.float32)

        out = sae(x)
        a = out.a  # [N, K], ReLU outputs

        flat = a.reshape(-1)
        total_entries_seen += int(flat.numel())

        m = min(int(samples_per_batch), int(flat.numel()))
        if m <= 0:
            continue

        # Uniformly sample entries from the flattened activations
        idx = torch.randint(0, flat.numel(), (m,), device=flat.device)
        samp = flat[idx].detach().to("cpu")
        samples_cpu.append(samp)

    if not samples_cpu:
        # Fallback: no samples collected
        return 0.0, {"bt_theta": 0.0, "bt_q": q}

    all_samples = torch.cat(samples_cpu, dim=0).to(torch.float32)  # CPU tensor
    theta = float(torch.quantile(all_samples, q).item())

    stats = {
        "bt_theta": theta,
        "bt_q": q,
        "bt_samples": float(all_samples.numel()),
        "bt_entries_seen": float(total_entries_seen),
    }
    return theta, stats


@torch.no_grad()
def estimate_l0_with_theta(
    cfg: Config,
    sae,
    *,
    model,
    hook_name: str,
    theta: float,
    num_batches: int = 5,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Quickly estimate achieved L0 under global threshold gating."""
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

    n_rows = 0
    active_total = 0.0

    for _ in range(num_batches):
        tokens = next(it)
        acts = get_activations(model, tokens, hook_name)
        x = flatten_activations(acts).to(torch.float32)
        out = sae(x)
        a = out.a

        mask = (a > (theta - eps))
        active_total += float(mask.sum().item())
        n_rows += int(a.shape[0])

    l0_mean = active_total / max(float(n_rows), 1e-12)
    return {"bt_theta_l0_mean": float(l0_mean)}
