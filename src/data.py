# data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class TokenBatcher:
    """Produces batches of token IDs shaped [batch, seq_len].

    Default mode: read from a local text file, tokenize with the model tokenizer (via TransformerLens).
    Optional mode: stream from a HuggingFace dataset if `datasets` is installed.
    """

    tokenizer: any  # HookedTransformer.tokenizer (HF tokenizer)
    batch_size: int
    seq_len: int
    device: torch.device

    local_text_path: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_split: str = "train"
    hf_text_field: str = "text"
    hf_streaming: bool = True
    hf_dataset_config: Optional[str] = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.local_text_path:
            yield from self._iter_local_file(self.local_text_path)
        elif self.hf_dataset:
            yield from self._iter_hf_dataset(self.hf_dataset, self.hf_split)
        else:
            raise ValueError("Provide either local_text_path or hf_dataset.")

    def _iter_local_file(self, path: str) -> Iterator[torch.Tensor]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the full file once, then slice. This is simple, not memory-optimal.
        ids = self.tokenizer.encode(text)
        if len(ids) < self.seq_len + 1:
            raise ValueError(f"Not enough tokens in {path} for seq_len={self.seq_len}")

        # Make contiguous sequences; wrap-around for convenience.
        pos = 0
        while True:
            batch = []
            for _ in range(self.batch_size):
                if pos + self.seq_len >= len(ids):
                    pos = 0
                chunk = ids[pos : pos + self.seq_len]
                pos += self.seq_len
                batch.append(chunk)
            toks = torch.tensor(batch, dtype=torch.long, device=self.device)
            yield toks

    def _iter_hf_dataset(self, name: str, split: str) -> Iterator[torch.Tensor]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise ImportError(
                "To use hf_dataset, install datasets: pip install datasets"
            ) from e
        # ds = load_dataset(name, split=split, streaming=self.hf_streaming, config=self.hf_dataset_config)
        ds = load_dataset(
            name,
            self.hf_dataset_config,
            split=split,
            streaming=self.hf_streaming,
        )
        # ds = load_dataset(name, split=split, streaming=self.hf_streaming)
        
        buffer_ids: list[int] = []

        def push_text(t: str) -> None:
            nonlocal buffer_ids
            buffer_ids.extend(self.tokenizer.encode(t))

        for ex in ds:
            t = ex.get(self.hf_text_field)
            if not isinstance(t, str) or not t.strip():
                continue
            push_text(t)

            # Emit as soon as we have enough
            while len(buffer_ids) >= self.batch_size * self.seq_len:
                batch = []
                for _ in range(self.batch_size):
                    chunk = buffer_ids[: self.seq_len]
                    buffer_ids = buffer_ids[self.seq_len :]
                    batch.append(chunk)
                toks = torch.tensor(batch, dtype=torch.long, device=self.device)
                yield toks
