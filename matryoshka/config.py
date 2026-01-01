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

# Which sparsity objective to use:
# - "baseline_l1": standard SAE with uniform L1 sparsity penalty
# - "freq_weighted_l1": frequency-weighted (TF-IDF-style) sparsity penalty (this project)
# config.py

ReconVariant = Literal["standard", "matryoshka", "stoch_avail"]
Sparsity = Literal["l1_uniform", "l1_freq_weighted", "batchtopk"]



@dataclass
class Config:
    """
    Configuration for training a Sparse Autoencoder (SAE) on TransformerLens activations.

    This config controls:
      - which model + activations are used
      - how data is streamed
      - SAE architecture + optimization
      - which sparsity objective is applied
      - what evaluation metrics are computed

    The key experimental switches are `recon_variant` and `sparsity, which selects between the
    different approaches, e.g. standard SAE objective and the frequency-weighted sparsity objective.
    """

    # ============================================================
    # Run bookkeeping / reproducibility
    # ============================================================

    # Human-readable run name (used in output directory names)
    run_name: str = "sae_gemma2b"

    # Root directory where all runs are stored
    out_dir: str = "runs"

    # RNG seed for torch / python
    seed: int = 0

    # ============================================================
    # Model + activation selection
    # ============================================================

    # HuggingFace / TransformerLens model identifier
    model_name: str = "google/gemma-2b-it"

    # Transformer layer index whose activations we train the SAE on
    layer: int = 12

    # Hook name template used by TransformerLens
    # `{layer}` is filled in at runtime by `resolved_hook`
    hook_point: str = "blocks.{layer}.mlp.hook_post"

    # Device placement for the transformer model
    device: Device = "auto"

    # Numeric precision for the transformer model
    # SAE itself is kept in fp32 for stability
    dtype: DType = "auto"

    # ---- stochastic availability knobs ----
    # a: latent activations from the SAE encoder, shape [N, K]
    # p: per-latent availability probabilities, shape [K], with p_i in (0, 1]
    # m: sampled Bernoulli mask, shape [N, K], where m_{n,i} ~ Bernoulli(p_i)
    #
    # Inverted dropout scaling: (m/p) keeps E[m_i/p_i] = 1, so E[a_masked] = a.
    # This makes the masked reconstruction unbiased in expectation and stabilizes training.
    sa_p_min: float = 0.2          # minimum availability probability
    sa_gamma: float = 2.0          # shape of the decay curve (larger => steeper tail)
    sa_inverted: bool = True       # use (m/p)*a rather than m*a
    sa_shared_across_batch: bool = True  # if True: one mask per step; else mask per example

    # ============================================================
    # Data / token stream
    # ============================================================
    # Name of the text field inside each dataset example
    # Most datasets use "text", but some use "content", "body", etc.

    # Data / token stream
    local_text_path: Optional[str] = None

    # Default to a small, stable dataset so train.py works out of the box
    hf_dataset: Optional[str] = "wikitext"
    hf_dataset_config: Optional[str] = "wikitext-2-raw-v1"
    hf_split: str = "train"
    hf_text_field: str = "text"
    
    # Sequence length for each tokenized example
    seq_len: int = 256

    # Number of sequences per batch
    # On a Mac, this is typically 1.
    batch_size: int = 2

    # Total number of training steps (each step = one SAE update)
    num_steps: int = 2_000

    # ============================================================
    # SAE architecture
    # ============================================================

    # Number of latent features in the SAE
    n_latents: int = 16_384

    # If True, encoder weights are tied to decoder weights (W_enc = W_dec^T)
    tied_weights: bool = False

    # Nonlinearity used in the encoder
    activation: Literal["relu"] = "relu"

    # Reconstruction loss type
    recon_loss: Literal["mse"] = "mse"

    # ============================================================
    # Optimization hyperparameters
    # ============================================================

    # Learning rate for SAE parameters
    lr: float = 3e-4

    # Weight decay applied by AdamW
    weight_decay: float = 0.0

    # Global gradient norm clipping (0 disables clipping)
    grad_clip: float = 1.0

    # How often (in steps) to log metrics
    log_every: int = 25

    # How often (in steps) to save checkpoints
    ckpt_every: int = 500

    # ------------------------------------------------------------
    # Reconstruction variant
    # ------------------------------------------------------------
    # Selects how reconstructions are formed from latent activations.
    #
    # Available options:
    #
    # - "standard"
    #     Standard SAE reconstruction: x_hat = W_dec @ z
    #
    # - "matryoshka"
    #     Matryoshka-style reconstruction with hierarchical / nested
    #     decoder structure (see paper for details).
    #
    recon_variant: ReconVariant = "standard"

    # ------------------------------------------------------------
    # Sparsity objective
    # ------------------------------------------------------------
    # Selects which sparsity penalty is applied to latent activations.
    #
    # Available options:
    #
    # - "l1_uniform"
    #     Baseline SAE objective with a uniform L1 penalty on all latents.
    #
    # - "l1_freq_weighted"
    #     Frequency-weighted L1 penalty where each latent i has its own
    #     lambda_i ~ 1 / (p_i + eps)^alpha, with p_i estimated from
    #     historical activation frequency.
    #
    # - "l1_stochastic_availability"
    #     Stochastic availability objective: latents are randomly masked
    #     during training according to learned or fixed availability
    #     probabilities, encouraging redundancy-aware representations.
    #
    sparsity: Sparsity = "l1_uniform"

    
    # Base sparsity coefficient.
    # For baseline_l1: lambda
    # For freq_weighted_l1: mean(lambda_i)
    lambda_base: float = 1e-3

    # ---- Frequency-weighted sparsity parameters ----

    # EMA decay used to estimate latent activation frequencies p_i
    fw_ema_beta: float = 0.99

    # Small constant to avoid division by zero in 1/(p_i + eps)
    fw_eps: float = 1e-4

    # Exponent alpha in lambda_i ~ 1/(p_i + eps)^alpha
    # alpha = 1.0 is aggressive; alpha = 0.5 is smoother
    fw_alpha: float = 0.5

    # Number of initial steps during which sparsity is uniform
    # (helps avoid early collapse / rich-get-richer dynamics)
    fw_warmup_steps: int = 200

    # Lower and upper bounds on lambda_i to prevent extreme penalties
    fw_clip_min: float = 1e-4
    fw_clip_max: float = 1e-2

    # If True, normalize lambda_i so mean(lambda_i) == lambda_base
    fw_normalize_mean: bool = True

    # ============================================================
    # Evaluation settings
    # ============================================================

    # Number of batches used for quick evaluation passes
    eval_num_batches: int = 10

    # Threshold used to count a latent as "active"
    # For ReLU SAEs, >0 is natural
    active_threshold: float = 0.0

    # Number of decoder-vector pairs sampled when estimating redundancy
    redundancy_num_pairs: int = 200_000

    # Cosine similarity threshold above which decoder vectors
    # are considered near-duplicates
    redundancy_high_sim: float = 0.95
    

    # Used when sparsity == "batchtopk"
    target_l0: Optional[float] = None   # for batchtopk
    btq_eps: float = 1e-12              # numerical guard for thresholds
    btq_tie_break: Literal["none","random"] = "none"  # optional
    
    # Used when recon_variant == "matryoshka"
    matryoshka_ms: List[int] = field(default_factory=lambda: [])  # if empty, auto-fill from n_latents
    matryoshka_recon_agg: Literal["mean","sum"] = "mean"
    matryoshka_include_full: bool = True  # whether to always include m = n_latents (usually yes)

    # calibration
    calibrate_l0: bool = False # (whether to run calibration)
    calib_rounds: int = 12
    calib_batches: int = 10
    calib_tol: float = 0.5


    # ============================================================
    # MLflow logging (optional)
    # ============================================================

    # Tracking URI for MLflow (None disables MLflow entirely)
    mlflow_uri: Optional[str] = None

    # Experiment name shown in the MLflow UI
    mlflow_experiment: str = "sae_freq_weighted"

    # Master switch for MLflow logging
    mlflow_enabled: bool = True

    # ============================================================
    # Derived fields (filled at runtime)
    # ============================================================

    # Directory for this specific run (auto-generated)
    run_dir: str = field(default="", init=False)

    def resolved_hook(self) -> str:
        """Return the concrete hook name for the selected layer."""
        return self.hook_point.format(layer=self.layer)

    # def resolve_modes(self: Config) -> Config:
    #     """
    #     Back-compat mapping if cfg.method is set.
    #     Returns cfg with recon_variant/sparsity filled.
    #     """
    #     if self.method is None:
    #         return self

    #     mapping = {
    #         "baseline_l1": ("standard", "l1_uniform"),
    #         "freq_weighted_l1": ("standard", "l1_freq_weighted"),
    #         "stoch_avail_l1": ("stoch_avail", "l1_uniform"),
    #     }
    #     rv, sp = mapping[self.method]
    #     self.recon_variant = rv
    #     self.sparsity = sp
    #     return self
    
# ------------------------------------------------------------
# Run directory + config persistence helpers
# ------------------------------------------------------------

def make_run_dir(cfg: Config) -> Path:
    """Create a unique run directory and store it in cfg.run_dir."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.out_dir) / f"{cfg.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.run_dir = str(run_dir)
    return run_dir


def save_config(cfg: Config, run_dir: Path) -> None:
    """Save the full configuration to JSON for reproducibility."""
    path = run_dir / "config.json"
    with path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)



