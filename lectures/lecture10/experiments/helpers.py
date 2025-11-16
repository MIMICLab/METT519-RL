"""Shared utilities for Lecture 10 (PPO) experiments."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE10_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
