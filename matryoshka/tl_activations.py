# tl_activations.py
from __future__ import annotations

from typing import Tuple

import torch
from transformer_lens import HookedTransformer


def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if device == "mps":
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        return torch.bfloat16  # generally safe default on modern NVIDIA
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_tl_model(model_name: str, device: torch.device, dtype: torch.dtype) -> HookedTransformer:
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=str(device), dtype=dtype
    )
    model.eval()
    return model


@torch.no_grad()
def get_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,  # [batch, seq]
    hook_name: str,
) -> torch.Tensor:
    """Return activations at hook_name shaped [batch, seq, d_model]."""
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name]
    return acts


def flatten_activations(acts: torch.Tensor) -> torch.Tensor:
    """Flatten [batch, seq, d_model] -> [batch*seq, d_model]."""
    b, s, d = acts.shape
    return acts.reshape(b * s, d).contiguous()
