# config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
from typing import Literal, Optional, List

# -----------------------------
# Type aliases for clarity
# -----------------------------

Device = Literal["auto", "cpu", "cuda", "mps"]
DType = Literal["auto", "fp32", "fp16", "bf16"]

ReconVariant = Literal["standard", "matryoshka"]

# Sparsity objectives
#
# This repo is testing two knobs that both aim to reduce “feature splitting” / redundancy while
# keeping recon quality acceptable:
#
# This repo is centered on two experimental ideas:
#
#   Idea 1: Concave sparsity via curriculum (a.k.a. "P‑annealing")
#   ----------------------------------------------------------------
#   Replace the usual convex L1 penalty with an Lp penalty where p < 1
#   (e.g. p=0.5 behaves like sqrt), which is “closer” to L0 in spirit.
#
#       L_sparse(a) ~ sum_i (|a_i| + eps)^p
#
#   For p < 1, this is concave / non‑convex. The intuition is:
#     • encourages fewer *active* latents per example (more “winner-take-most”)
#     • tends to concentrate mass into fewer latents instead of many small ones
#     • can improve interpretability but can be harder / “stickier” to train
#
#   To mitigate training pathologies, we *do not* jump straight to p<1:
#     • start at p_start = 1.0 (standard L1; stable gradients early)
#     • anneal linearly to p_end < 1 (e.g. 0.5) after the dictionary starts forming
#     • annealing window: [anneal_start_step, anneal_end_step]
#       (defaults: 20% → 80% of num_steps if unspecified)
#
#   Note on eps:
#     We add `sparsity_eps` inside the power: (|a| + eps)^p.
#     This avoids numerical instability / extreme gradients near zero when p<1.
#
#
#   Idea 2: Frequency‑weighted sparsity (TF‑IDF‑inspired weighting)
#   ----------------------------------------------------------------
#   Maintain an EMA of each latent’s activation frequency:
#
#       freq_i ≈ P[a_i > 0]
#
#   Then scale the per‑latent sparsity coefficient:
#
#       lambda_i = lambda_base * w_i
#       w_i ∝ (freq_i + fw_eps)^(-alpha)
#
#   In the CURRENT implementation (see FrequencyWeightedLpPenalty):
#     • common latents (high freq_i) get *smaller* lambda_i  → cheaper to use
#     • rare latents   (low  freq_i) get *larger*  lambda_i  → more expensive to use
#
#   Why this matches the write-up’s “TF‑IDF / common concepts are easy to ignore” framing:
#     • common factors can be represented without the model “gaming” the sparsity cost
#       by splitting them into many fragments (since common latents are discounted anyway)
#     • rare latents are preserved for genuinely “surprising / informative” concepts
#     • at interpretation time, you can still downweight common latents (TF‑IDF-style)
#       without forcing training-time sparsity to over-fragment common phenomena
#
#   (If you ever want the *opposite* behavior—penalize common latents MORE than rare ones—
#    you’d invert the weighting scheme, e.g. use w_i ∝ (freq_i + eps)^(+alpha) or flip alpha.
#    The current modes are explicitly testing the “discount common / penalize rare” direction.)
#
#   Practical details:
#     • fw_warmup_steps: for the first N steps, all lambdas are uniform = lambda_base
#       (helps stabilize early learning while EMA frequencies are noisy)
#     • fw_clip_min/max: clamp lambda_i to avoid extreme weights
#     • fw_normalize_mean: normalize w so mean(w)=1 (keeps lambda_base comparable)
#
#
# What each cfg.sparsity mode is testing
# -------------------------------------
# 1) "l1_uniform"  (Baseline)
#    - Pure L1 sparsity: p fixed at 1.0, uniform weighting over latents.
#    - This is the control condition: neither Idea 1 nor Idea 2 is active.
#    - Useful for: baseline FVU vs L0 tradeoff, redundancy, dead features, etc.
#
# 2) "l1_freq_weighted"  (Idea 2 only)
#    - p fixed at 1.0 (still convex L1-like shape),
#      but lambda_i varies per latent based on EMA frequency.
#    - Tests: does frequency-weighted penalty reduce “bad feature splitting”
#      / improve interpretability at a given reconstruction quality?
#
# 3) "p_annealing"  (Idea 1 only)
#    - Uniform weights, but p anneals from p_start -> p_end over training.
#    - Tests: can we safely introduce concavity later in training to get more
#      per-example sparsity (closer to L0 behavior) without early-training instability?
#
# 4) "p_annealing_freq"  (Idea 1 + Idea 2 combined)
#    - Combines both mechanisms:
#        • concave sparsity pressure late in training (Idea 1)
#        • frequency-dependent lambda_i (Idea 2)
#    - Tests: synergy. The intended effect is:
#        • concavity encourages fewer active latents per example
#        • freq-weighting makes common “boring” latents cheap, so the model
#          is less incentivized to fragment common factors just to reduce penalty
#
# 5) "batchtopk"  (Reference / alternative to concave penalties)
#    - Not a penalty; it’s a hard gating constraint applied to activations.
#      We keep only the globally top activations across the batch such that:
#         average L0 per row ≈ target_l0
#    - Pros: avoids directly optimizing a concave objective.
#    - Cons: masked latents get zero gradient (risk of dead latents unless you add
#      auxiliary mechanisms). Included mainly as a comparison point vs Idea 1’s
#      “trainable concavity” approach.
#
# Hyperparameter note:
#   - Penalty-based modes ("l1_*", "*annealing*") primarily tune lambda_base
#     (optionally calibrated to a target L0).
#   - "batchtopk" primarily tunes target_l0 (and optionally a threshold theta for eval).
Sparsity = Literal[
    "l1_uniform",
    "l1_freq_weighted",
    "p_annealing",
    "p_annealing_freq",
    "batchtopk",
]

@dataclass
class Config:
    """
    Configuration for training a Sparse Autoencoder (SAE).
    """

    # ============================================================
    # Run bookkeeping / reproducibility
    # ============================================================
    run_name: str = "sae_gemma2b"
    out_dir: str = "runs"
    seed: int = 0

    # ============================================================
    # Model + activation selection
    # ============================================================
    model_name: str = "google/gemma-2b-it"
    layer: int = 12
    hook_point: str = "blocks.{layer}.mlp.hook_post"
    device: Device = "auto"
    dtype: DType = "auto"

    # ============================================================
    # Data / token stream
    # ============================================================
    local_text_path: Optional[str] = None
    hf_dataset: Optional[str] = "wikitext"
    hf_dataset_config: Optional[str] = "wikitext-2-raw-v1"
    hf_split: str = "train"
    hf_text_field: str = "text"
    seq_len: int = 256
    batch_size: int = 2
    num_steps: int = 2_000

    # ============================================================
    # SAE architecture
    # ============================================================
    n_latents: int = 16_384
    tied_weights: bool = False
    activation: Literal["relu"] = "relu"
    recon_loss: Literal["mse"] = "mse"

    # ============================================================
    # Optimization hyperparameters
    # ============================================================
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 100
    ckpt_every: int = 500

    # ------------------------------------------------------------
    # Reconstruction variant
    # ------------------------------------------------------------
    recon_variant: ReconVariant = "standard"

    # ------------------------------------------------------------
    # Sparsity objective
    # ------------------------------------------------------------
    sparsity: Sparsity = "l1_uniform"

    # Base sparsity coefficient lambda
    lambda_base: float = 2e-5

    # ---- P-Annealing Parameters (Idea 1) ----
    # Norm power p. For standard L1, set both to 1.0.
    # For annealing, typically p_start=1.0 -> p_end=0.5.
    p_start: float = 1.0
    p_end: float = 1.0 # change to 0.5 when running annealing experiments.

    # Annealing schedule (in steps).
    # If None, defaults to 20% and 80% of num_steps respectively.
    anneal_start_step: Optional[int] = None
    anneal_end_step: Optional[int] = None

    # Epsilon for numerical stability in concave loss: sum(|z| + eps)^p
    sparsity_eps: float = 1e-6

    # ---- Frequency-weighted sparsity parameters (Idea 2) ----
    fw_ema_beta: float = 0.99
    fw_eps: float = 1e-4
    fw_alpha: float = 0.5

    # Warmup: number of steps before frequency penalty applies.
    # During this period, weights are 1.0 (uniform).
    fw_warmup_steps: int = 200

    fw_clip_min: float = 0.1     # factor (min = 0.1 * lambda_base)
    fw_clip_max: float = 10.0    # factor (max = 10 * lambda_base)
    fw_clip_relative: bool = True
    fw_normalize_mean: bool = True

    decoder_unit_norm: bool = True
    decoder_unit_norm_eps: float = 1e-8
    # ============================================================
    # Evaluation settings
    # ============================================================
    eval_num_batches: int = 10
    active_threshold: float = 0.0
    redundancy_num_pairs: int = 200_000
    redundancy_high_sim: float = 0.95

    target_l0: Optional[float] = None   # for batchtopk
    btq_eps: float = 1e-12
    btq_tie_break: Literal["none","random"] = "none"

    matryoshka_ms: List[int] = field(default_factory=lambda: [])
    matryoshka_recon_agg: Literal["mean","sum"] = "mean"
    matryoshka_include_full: bool = True

    calibrate_l0: bool = False
    calib_rounds: int = 12
    calib_batches: int = 10
    calib_tol: float = 0.5

    # ============================================================
    # MLflow logging
    # ============================================================
    mlflow_uri: Optional[str] = None
    mlflow_experiment: str = "sae_experiments"
    mlflow_enabled: bool = True

    # ============================================================
    # Derived fields
    # ============================================================
    run_dir: str = field(default="", init=False)

    def resolved_hook(self) -> str:
        return self.hook_point.format(layer=self.layer)


def make_run_dir(cfg: Config) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.out_dir) / f"{cfg.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.run_dir = str(run_dir)
    return run_dir


def save_config(cfg: Config, run_dir: Path) -> None:
    path = run_dir / "config.json"
    with path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)