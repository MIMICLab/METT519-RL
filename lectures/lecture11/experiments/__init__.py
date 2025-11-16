"""Lecture 11 experiment package helpers."""

from pathlib import Path
from typing import Final

EXPERIMENT_ROOT: Final[Path] = Path(__file__).resolve().parent
RUNS_ROOT: Final[Path] = EXPERIMENT_ROOT / "runs"
FIGURES_ROOT: Final[Path] = EXPERIMENT_ROOT / "figures"

__all__ = [
    "EXPERIMENT_ROOT",
    "RUNS_ROOT",
    "FIGURES_ROOT",
]
