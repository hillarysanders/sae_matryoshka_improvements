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

from calibration import calibrate_theta_for_target_l0, estimate_l0_with_theta
from stoch_avail import StochasticAvailability
from config import Config, make_run_dir, save_config
from data import TokenBatcher
from tl_activations import pick_device, pick_dtype, load_tl_model, get_activations, flatten_activations
from sae import SparseAutoencoder, recon_loss
from sparsity import UniformL1Penalty, FrequencyWeightedL1Penalty
from batchtopk import apply_batchtopk
from eval_metrics import run_quick_eval
from mlflow_utils import maybe_init_mlflow, mlflow_log_metrics, mlflow_log_artifact


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_penalty(cfg: Config, device: torch.device):
    """Return a sparsity penalty module for L1-based modes.
    For BatchTopK, sparsity is enforced by gating (no penalty loss), so return None.
    """
    if cfg.sparsity == "l1_uniform":
        return UniformL1Penalty(lambda_base=cfg.lambda_base)

    if cfg.sparsity == "l1_freq_weighted":
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

    if cfg.recon_variant == "matryoshka":
        raise ValueError("recon_variant='matryoshka' not implemented yet (later commit).")

    model = load_tl_model(cfg.model_name, device=device, dtype=dtype)
    hook_name = cfg.resolved_hook()

    # Get d_model by grabbing one forward pass (cheap and robust).
    toks0 = torch.tensor([[model.tokenizer.bos_token_id] * cfg.seq_len], device=device)
    acts0 = get_activations(model, toks0, hook_name)
    d_model = acts0.shape[-1]

    sae = SparseAutoencoder(
        d_model=d_model,
        n_latents=cfg.n_latents,
        tied_weights=cfg.tied_weights,
        activation=cfg.activation,
        device=device,
        dtype=torch.float32,  # keep SAE in fp32 for stability
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

    stoch_avail = None
    if cfg.recon_variant == "stoch_avail":
        stoch_avail = StochasticAvailability(
            n_latents=cfg.n_latents,
            p_min=cfg.sa_p_min,
            gamma=cfg.sa_gamma,
            inverted=cfg.sa_inverted,
            shared_across_batch=cfg.sa_shared_across_batch,
            device=device,
        )

    sae.train()
    pbar = tqdm(total=cfg.num_steps, desc="train", dynamic_ncols=True)
    t0 = time.time()
    tokens_seen = 0

    for step in range(1, cfg.num_steps + 1):
        tokens = next(it)  # [batch, seq]
        tokens_seen += int(tokens.numel())

        with torch.no_grad():
            acts = get_activations(model, tokens, hook_name)
            x = flatten_activations(acts).to(torch.float32)

        out = sae(x)
        a = out.a  # [N, K]

        # -------------------------
        # Apply sparsity mechanism to produce activations used for decoding
        # -------------------------
        bt_stats: Dict[str, float] = {}
        if cfg.sparsity == "batchtopk":
            # Gate *before* any recon-variant masking (availability), since BatchTopK is the primary sparsity.
            a_used, bt_stats = apply_batchtopk(
                a,
                k_per_row=float(cfg.target_l0),   # type: ignore[arg-type]
                tie_break=getattr(cfg, "btq_tie_break", "none"),
                eps=getattr(cfg, "btq_eps", 1e-12),
            )
        else:
            a_used = a

        # -------------------------
        # Recon path depends on recon_variant
        # -------------------------
        sa_stats: Dict[str, float] = {}
        if cfg.recon_variant == "stoch_avail":
            assert stoch_avail is not None
            # availability masking only affects reconstruction path
            a_masked, sa_stats = stoch_avail.mask(a_used)
            x_hat = sae.decode(a_masked)
        elif cfg.recon_variant == "standard":
            # decode from the activations used (batchtopk-gated if enabled)
            x_hat = sae.decode(a_used) if cfg.sparsity == "batchtopk" else out.x_hat
        else:
            raise ValueError(f"Unknown recon_variant: {cfg.recon_variant}")

        l_recon = recon_loss(x, x_hat, cfg.recon_loss)

        # -------------------------
        # Sparsity loss term (L1 modes only)
        # -------------------------
        if cfg.sparsity == "batchtopk":
            l_sparse = torch.zeros((), device=x.device, dtype=torch.float32)
            sparse_stats: Dict[str, float] = {"lambda_mean": 0.0}
        else:
            assert penalty is not None
            # For stoch_avail, apply L1 sparsity to the *true* a (ungated/unmasked),
            # matching your earlier behavior.
            l_sparse, sparse_stats = penalty.compute(a, step=step)

        loss = l_recon + l_sparse


        opt.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)

        opt.step()

        # ---- progress bar update every step ----
        if step == 1 or step % 5 == 0:
            dt = max(time.time() - t0, 1e-6)
            tok_per_s = tokens_seen / dt
            postfix = dict(
                loss=f"{loss.item():.3e}",
                recon=f"{l_recon.item():.3e}",
                sparse=f"{l_sparse.item():.3e}",
                tok_s=f"{tok_per_s:.0f}",
            )
            lam_mean = sparse_stats.get("lambda_mean")
            if "bt_l0_mean" in bt_stats:
                postfix["bt_l0"] = f"{bt_stats['bt_l0_mean']:.1f}"
            if lam_mean is not None:
                postfix["lam"] = f"{lam_mean:.2e}"
            if "sa_mask_mean" in sa_stats:
                postfix["mask"] = f"{sa_stats['sa_mask_mean']:.2f}"
            pbar.set_postfix(**postfix)

        pbar.update(1)

        # ---- logging ----
        if step % cfg.log_every == 0 or step == 1:
            row: Dict[str, float] = {
                "step": float(step),
                "loss": float(loss.item()),
                "recon": float(l_recon.item()),
                "sparse": float(l_sparse.item()),
            }
            row.update({k: float(v) for k, v in bt_stats.items()})
            row.update({k: float(v) for k, v in sparse_stats.items()})
            row.update({k: float(v) for k, v in sa_stats.items()})

            f_metrics.write(json.dumps(row) + "\n")
            f_metrics.flush()

            mlflow_log_metrics(cfg, row, step=step)

            print(
                f"[{step:>6d}] loss={row['loss']:.4e} recon={row['recon']:.4e} sparse={row['sparse']:.4e} "
                f"lam_mean={row.get('lambda_mean', float('nan')):.3e}"
            )

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

    # quick eval snapshot at end
    sae.eval()

    # --- BatchTopK inference calibration (global theta) ---
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
        (run_dir / "batchtopk_theta.json").write_text(
            json.dumps({"theta": theta, **theta_stats}, indent=2, sort_keys=True)
        )
        mlflow_log_artifact(cfg, str(run_dir / "batchtopk_theta.json"))

    eval_out = run_quick_eval(cfg, sae, model=model, hook_name=hook_name, theta=theta)
    if theta_stats:
        eval_out.update(theta_stats)

    (run_dir / "eval.json").write_text(json.dumps(eval_out, indent=2, sort_keys=True))
    mlflow_log_metrics(cfg, {f"eval_{k}": float(v) for k, v in eval_out.items()}, step=cfg.num_steps)
    mlflow_log_artifact(cfg, str(run_dir / "eval.json"))

    return run_dir


if __name__ == "__main__":
    import tyro

    cfg = tyro.cli(Config)
    train(cfg)
