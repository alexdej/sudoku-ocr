"""Shared types for sudoku-ocr."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellInfo:
    """Information extracted from a single sudoku cell.

    Designed for extensibility — future fields may include digit color,
    handwritten vs printed classification, confidence scores, etc.
    """

    row: int
    col: int
    color_image: np.ndarray | None = None
    grayscale_image: np.ndarray | None = None
    digit: int | None = None
    has_digit: bool = False
    is_given: bool | None = None
