"""Read digits from sudoku puzzle images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from .cells import segment_cells
from .grid import detect_grid, detect_grid_size
from .model import SudokuNet
from .types import CellInfo

ImageSource = Union[str, os.PathLike, Image.Image]


def _load_image(source: ImageSource) -> np.ndarray:
    """Load an image source into a BGR numpy array for OpenCV."""
    if isinstance(source, (str, os.PathLike)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {source}")
        return img
    if isinstance(source, Image.Image):
        rgb = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image source type: {type(source)}")


class PuzzleReader:
    """Extracts digits from a sudoku puzzle image."""

    def __init__(self, model: SudokuNet | None = None) -> None:
        self._model = model

    def read_cells(
        self, image: ImageSource, grid_size: int | None = None,
    ) -> tuple[np.ndarray, list[CellInfo]]:
        """Extract cell information from a sudoku puzzle image.

        This is the lower-level method that returns full cell data
        including color crops, useful for downstream analysis and
        visualization.

        Args:
            image: A file path or PIL Image of a sudoku puzzle.
            grid_size: Number of rows/columns, or None to auto-detect.

        Returns:
            A tuple of (color_warped, cells) where color_warped is the
            perspective-corrected image and cells is a list of CellInfo
            objects in row-major order.
        """
        bgr = _load_image(image)
        color_warped, gray_warped = detect_grid(bgr)

        if grid_size is None:
            grid_size = detect_grid_size(gray_warped)

        cells = segment_cells(color_warped, gray_warped, grid_size)

        if self._model is not None:
            for cell in cells:
                if cell.has_digit and cell.grayscale_image is not None:
                    cell.digit = self._model.predict(cell.grayscale_image)

        return color_warped, cells

    def read_digits(
        self, image: ImageSource, grid_size: int | None = None,
    ) -> list[int | None]:
        """Read all cells from a sudoku puzzle image.

        Args:
            image: A file path or PIL Image of a sudoku puzzle.
            grid_size: Number of rows/columns, or None to auto-detect.

        Returns:
            A flat list of values in row-major order.
            Each value is a digit (1-9) or None for empty cells.
        """
        _, cells = self.read_cells(image, grid_size)
        return [cell.digit for cell in cells]
