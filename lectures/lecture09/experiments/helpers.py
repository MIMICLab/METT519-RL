"""Common utilities for Lecture 9 experiments (Actor-Critic)."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE09_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return preferred compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: Iterable[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Move a batch of tensors to the given device."""
    return tuple(t.to(device) for t in batch)
