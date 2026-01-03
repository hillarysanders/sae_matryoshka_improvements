# train.py
from __future__ import annotations

import json
import random
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from dataclasses import replace
from sae import SparseAutoencoder, recon_loss, renorm_decoder_rows_
from eval_metrics import calibrate_lambda_for_target_l0
from calibration import calibrate_theta_for_target_l0, estimate_l0_with_theta
from config import Config, make_run_dir, save_config
from data import TokenBatcher
from tl_activations import pick_device, pick_dtype, load_tl_model, get_activations, flatten_activations
from sae import SparseAutoencoder, recon_loss
from sparsity import UniformLpPenalty, FrequencyWeightedLpPenalty
from batchtopk import apply_batchtopk
from eval_metrics import run_quick_eval
from mlflow_utils import maybe_init_mlflow, mlflow_log_metrics, mlflow_log_artifact
from matryoshka_utils import _resolve_matryoshka_ms, _decode_prefix

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_penalty(cfg: Config, device: torch.device):
    """Return a sparsity penalty module based on config."""
    
    # Calculate annealing steps defaults if not provided
    astart = cfg.anneal_start_step if cfg.anneal_start_step is not None else int(0.2 * cfg.num_steps)
    aend = cfg.anneal_end_step if cfg.anneal_end_step is not None else int(0.8 * cfg.num_steps)

    # 1. Uniform Weighting (Baseline or Pure P-Annealing)
    if cfg.sparsity in ("l1_uniform", "p_annealing"):
        return UniformLpPenalty(
            lambda_base=cfg.lambda_base,
            p_start=cfg.p_start,
            p_end=cfg.p_end,
            anneal_start=astart,
            anneal_end=aend,
            eps=cfg.sparsity_eps
        )

    # 2. Frequency Weighting (Idea 2 or Combined)
    if cfg.sparsity in ("l1_freq_weighted", "p_annealing_freq"):
        return FrequencyWeightedLpPenalty(
            n_latents=cfg.n_latents,
            lambda_base=cfg.lambda_base,
            
            # Freq params
            ema_beta=cfg.fw_ema_beta,
            fw_eps=cfg.fw_eps,
            alpha=cfg.fw_alpha,
            warmup_steps=cfg.fw_warmup_steps,
            clip_min=cfg.fw_clip_min,
            clip_max=cfg.fw_clip_max,
            normalize_mean=cfg.fw_normalize_mean,
            
            # P-Annealing params
            p_start=cfg.p_start,
            p_end=cfg.p_end,
            anneal_start=astart,
            anneal_end=aend,
            sparsity_eps=cfg.sparsity_eps,
            
            device=device,
        )

    if cfg.sparsity == "batchtopk":
        return None

    raise ValueError(f"Unknown sparsity: {cfg.sparsity}")


def train(cfg: Config) -> Path:
    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir)

    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype, device)
    set_seed(cfg.seed)

    if cfg.sparsity == "batchtopk" and cfg.target_l0 is None:
        raise ValueError("sparsity='batchtopk' requires --target_l0 (e.g. --target_l0 40).")

    model = load_tl_model(cfg.model_name, device=device, dtype=dtype)
    hook_name = cfg.resolved_hook()

    if cfg.recon_variant == "matryoshka":
        ms = _resolve_matryoshka_ms(cfg)
        print(f"[matryoshka] prefix sizes: {ms}")

    # Get d_model by grabbing one forward pass
    toks0 = torch.tensor([[model.tokenizer.bos_token_id] * cfg.seq_len], device=device)
    acts0 = get_activations(model, toks0, hook_name)
    d_model = acts0.shape[-1]

    sae = SparseAutoencoder(
        d_model=d_model,
        n_latents=cfg.n_latents,
        tied_weights=cfg.tied_weights,
        activation=cfg.activation,
        device=device,
        dtype=torch.float32,
    )

    penalty = _make_penalty(cfg, device=device)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = run_dir / "metrics.jsonl"
    f_metrics = metrics_path.open("w", encoding="utf-8")

    mlflow_run = maybe_init_mlflow(cfg)
    mlflow_log_artifact(cfg, str(run_dir / "config.json"))

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

    sae.train()
    pbar = tqdm(total=cfg.num_steps, desc="train", dynamic_ncols=True)
    t0 = time.time()
    tokens_seen = 0

    for step in range(1, cfg.num_steps + 1):
        tokens = next(it)
        tokens_seen += int(tokens.numel())

        with torch.no_grad():
            acts = get_activations(model, tokens, hook_name)
            x = flatten_activations(acts).to(torch.float32)

        out = sae(x)
        a = out.a

        # -------------------------
        # Sparsity: BatchTopK Gating (Architecture-based)
        # -------------------------
        bt_stats: Dict[str, float] = {}
        if cfg.sparsity == "batchtopk":
            a_used, bt_stats = apply_batchtopk(
                a,
                k_per_row=float(cfg.target_l0),
                tie_break=getattr(cfg, "btq_tie_break", "none"),
                eps=getattr(cfg, "btq_eps", 1e-12),
            )
        else:
            a_used = a

        # -------------------------
        # Reconstruction
        # -------------------------
        if cfg.recon_variant == "standard":
            x_hat = sae.decode(a_used) if cfg.sparsity == "batchtopk" else out.x_hat
            l_recon = recon_loss(x, x_hat, cfg.recon_loss)

        elif cfg.recon_variant == "matryoshka":
            ms = _resolve_matryoshka_ms(cfg)
            recon_terms = []
            for m in ms:
                x_hat_m = _decode_prefix(sae, a_used, m)
                recon_terms.append(recon_loss(x, x_hat_m, cfg.recon_loss))

            if getattr(cfg, "matryoshka_recon_agg", "mean") == "sum":
                l_recon = torch.stack(recon_terms).sum()
            else:
                l_recon = torch.stack(recon_terms).mean()
            
            # keep full recon for logging
            x_hat = _decode_prefix(sae, a_used, ms[-1])

        else:
            raise ValueError(f"Unknown recon_variant: {cfg.recon_variant}")

        # -------------------------
        # Sparsity Loss (Penalty-based)
        # -------------------------
        if cfg.sparsity == "batchtopk":
            l_sparse = torch.zeros((), device=x.device, dtype=torch.float32)
            sparse_stats: Dict[str, float] = {"lambda_mean": 0.0}
        else:
            assert penalty is not None
            # Compute Lp / Freq-Weighted penalty
            l_sparse, sparse_stats = penalty.compute(a, step=step)

        loss = l_recon + l_sparse

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)

        opt.step()
        
        # Baseline alignment: constrain decoder directions to unit norm
        dec_norms_pre = None
        if getattr(cfg, "decoder_unit_norm", False):
            dec_norms_pre = renorm_decoder_rows_(
                sae, eps=getattr(cfg, "decoder_unit_norm_eps", 1e-8)
            )

        # ---- Logging ----
        if step == 1 or step % 5 == 0:
            dt = max(time.time() - t0, 1e-6)
            tok_per_s = tokens_seen / dt
            postfix = dict(
                loss=f"{loss.item():.3e}",
                recon=f"{l_recon.item():.3e}",
                sparse=f"{l_sparse.item():.3e}",
                tok_s=f"{tok_per_s:.0f}",
            )
            
            # Add dynamic p logging
            if "p_current" in sparse_stats:
                postfix["p"] = f"{sparse_stats['p_current']:.2f}"
            
            lam_mean = sparse_stats.get("lambda_mean")
            if "bt_l0_mean" in bt_stats:
                postfix["bt_l0"] = f"{bt_stats['bt_l0_mean']:.1f}"
            if lam_mean is not None:
                postfix["lam"] = f"{lam_mean:.2e}"
                
            pbar.set_postfix(**postfix)

        pbar.update(1)

        if step % cfg.log_every == 0 or step == 1:
            row: Dict[str, float] = {
                "step": float(step),
                "loss": float(loss.item()),
                "recon": float(l_recon.item()),
                "sparse": float(l_sparse.item()),
            }
            row.update({k: float(v) for k, v in bt_stats.items()})
            row.update({k: float(v) for k, v in sparse_stats.items()})
            
            if dec_norms_pre is not None:
                row["wdec_norm_pre_mean"] = float(dec_norms_pre.mean().item())
                row["wdec_norm_pre_max"] = float(dec_norms_pre.max().item())
            
            f_metrics.write(json.dumps(row) + "\n")
            f_metrics.flush()

            mlflow_log_metrics(cfg, row, step=step)

        if step % cfg.ckpt_every == 0 or step == cfg.num_steps:
            ckpt_path = run_dir / f"sae_step_{step}.pt"
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "d_model": d_model,
                    "state_dict": sae.state_dict(),
                },
                ckpt_path,
            )
            mlflow_log_artifact(cfg, str(ckpt_path))

    pbar.close()
    f_metrics.close()

    # Evaluation phase at end
    sae.eval()

    # If using L1/Lp modes, optionally calibrate lambda for target L0
    if cfg.calibrate_l0 and cfg.target_l0 is not None and cfg.sparsity != "batchtopk":
        new_lam = calibrate_lambda_for_target_l0(
            cfg, sae,
            model=model,
            hook_name=hook_name,
            target_l0=float(cfg.target_l0),
            num_batches=cfg.calib_batches,
            active_threshold=cfg.active_threshold,
            max_rounds=cfg.calib_rounds,
            tol=cfg.calib_tol,
        )
        cfg = replace(cfg, lambda_base=float(new_lam))
        if penalty is not None and hasattr(penalty, "lambda_base"):
            penalty.lambda_base = float(new_lam)

    # BatchTopK theta calibration
    theta = None
    theta_stats: Dict[str, float] = {}
    if cfg.sparsity == "batchtopk":
        assert cfg.target_l0 is not None
        theta, theta_stats = calibrate_theta_for_target_l0(
            cfg, sae,
            model=model,
            hook_name=hook_name,
            target_l0=float(cfg.target_l0),
            num_batches=getattr(cfg, "calib_batches", 10),
            samples_per_batch=200_000,
            eps=getattr(cfg, "btq_eps", 1e-12),
        )
        theta_stats.update(
            estimate_l0_with_theta(
                cfg, sae,
                model=model,
                hook_name=hook_name,
                theta=float(theta),
                num_batches=5,
                eps=getattr(cfg, "btq_eps", 1e-12),
            )
        )

    eval_out = run_quick_eval(cfg, sae, model=model, hook_name=hook_name, theta=theta)
    if theta_stats:
        eval_out.update(theta_stats)

    (run_dir / "eval.json").write_text(json.dumps(eval_out, indent=2, sort_keys=True))
    mlflow_log_metrics(cfg, {f"eval_{k}": float(v) for k, v in eval_out.items()}, step=cfg.num_steps)

    return run_dir


if __name__ == "__main__":
    import tyro
    cfg = tyro.cli(Config)
    train(cfg)