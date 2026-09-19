"""Validated experiment inputs and conservative backend support policy."""

from dataclasses import dataclass
from typing import Literal

import torch

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    revision: str = "main"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = 1
    prompt_length: int = 64
    max_new_tokens: int = 32
    warmups: int = 2
    repetitions: int = 10
    threads: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("batch_size", "prompt_length", "max_new_tokens", "warmups",
                     "repetitions", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Unknown device: {self.device}")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unknown dtype: {self.dtype}")
        if not self.model or not self.revision:
            raise ValueError("model and revision must not be empty")


def unsupported_reason(config: Config) -> str | None:
    """Return a skip reason; CPU low precision is deliberately outside this study."""
    if config.device == "cpu" and config.dtype != "float32":
        return "CPU low precision is outside this portable baseline's support policy"
    if config.device == "cuda":
        if not torch.cuda.is_available():
            return "CUDA is unavailable"
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            return "CUDA bfloat16 is unavailable"
    if config.device == "mps" and not torch.backends.mps.is_available():
        return "MPS is unavailable"
    try:
        x = torch.ones((2, 2), device=config.device, dtype=getattr(torch, config.dtype))
        (x @ x).sum().item()
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        return f"Device/dtype probe failed: {exc}"
    return None
