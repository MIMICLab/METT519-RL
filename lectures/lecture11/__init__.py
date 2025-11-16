"""Lecture 11 package marker."""

from pathlib import Path
from typing import Final

LECTURE11_ROOT: Final[Path] = Path(__file__).resolve().parent

__all__ = ["LECTURE11_ROOT"]
