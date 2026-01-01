import torch
from config import Config
from sae import SparseAutoencoder, normalize_decoder_rows

# ----------------------------
# Matryoshka helpers (fixed-prefix)
# ----------------------------

def _default_matryoshka_ms(n_latents: int) -> list[int]:
    """Reasonable default prefix ladder (small -> larger -> full)."""
    ladder: list[int] = []
    for frac in (1 / 32, 1 / 16, 1 / 8, 1 / 4):
        m = int(round(n_latents * frac))
        ladder.append(max(8, min(m, n_latents)))
    ladder = sorted(set(ladder))
    if not ladder or ladder[-1] != n_latents:
        ladder.append(n_latents)
    return ladder


def _resolve_matryoshka_ms(cfg: Config) -> list[int]:
    ms = list(getattr(cfg, "matryoshka_ms", []) or [])
    if not ms:
        ms = _default_matryoshka_ms(cfg.n_latents)

    ms = sorted(set(int(m) for m in ms))
    ms = [m for m in ms if 1 <= m <= cfg.n_latents]

    if getattr(cfg, "matryoshka_include_full", True):
        if (not ms) or (ms[-1] != cfg.n_latents):
            ms.append(cfg.n_latents)

    if not ms:
        raise ValueError("matryoshka_ms resolved to empty; check config.")
    return ms


def _decode_prefix(sae: SparseAutoencoder, a_used: torch.Tensor, m: int) -> torch.Tensor:
    """Decode using only the first m latents (fixed-prefix Matryoshka)."""
    return a_used[:, :m] @ sae.W_dec[:m] + sae.b_dec
