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

# Sparsity objectives:
# - "l1_uniform": Standard L1 (p=1.0 fixed, uniform weights)
# - "l1_freq_weighted": L1 with frequency weights (p=1.0 fixed)
# - "p_annealing": Uniform weights, p anneals from p_start -> p_end
# - "p_annealing_freq": Frequency weights, p anneals from p_start -> p_end
# - "batchtopk": Hard sparsity constraint
Sparsity = Literal[
    "l1_uniform", 
    "l1_freq_weighted", 
    "p_annealing", 
    "p_annealing_freq", 
    "batchtopk"
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
    log_every: int = 25
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
    lambda_base: float = 1e-3

    # ---- P-Annealing Parameters (Idea 1) ----
    # Norm power p. For standard L1, set both to 1.0.
    # For annealing, typically p_start=1.0 -> p_end=0.5.
    p_start: float = 1.0
    p_end: float = 1.0
    
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
    
    fw_clip_min: float = 1e-4
    fw_clip_max: float = 1e-2
    fw_normalize_mean: bool = True

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