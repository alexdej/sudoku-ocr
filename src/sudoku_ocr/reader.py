"""Read digits from sudoku puzzle images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from .cells import segment_cells
from .grid import detect_grid, detect_grid_size
from .model import SudokuNet
from .types import CellInfo

if TYPE_CHECKING:
    from PIL import Image as _PILImage

# Union type for image inputs.  PIL.Image.Image is supported when Pillow is
# installed; without it, only file paths are accepted.
ImageSource = Union[str, os.PathLike, "Image.Image"]

_PILLOW_INSTALL_HINT = (
    "Install Pillow to load this image format: pip install Pillow"
)

# Threshold: grids larger than this use the hex model (1-9+A-F) instead of
# the standard model (1-9).
_HEX_GRID_THRESHOLD = 9

# Default model file names inside the weights directory.
_MODEL_STANDARD = "digits_1_9.pt"
_MODEL_HEX      = "digits_hex.pt"
_MODEL_LEGACY   = "digit_classifier.pt"  # fallback for older installs


def _load_image(source: ImageSource) -> np.ndarray:
    """Load an image source into a BGR numpy array for OpenCV."""
    if isinstance(source, (str, os.PathLike)):
        img = cv2.imread(str(source))
        if img is not None:
            return img
        # OpenCV can't decode this format (GIF, WEBP, JFIF, …) — try Pillow.
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                f"Could not load image: {source}\n{_PILLOW_INSTALL_HINT}"
            ) from None
        try:
            pil_img = Image.open(source).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Could not load image: {source}") from exc
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Accept a PIL Image object directly (requires Pillow at call time).
    try:
        from PIL import Image
        if isinstance(source, Image.Image):
            rgb = np.array(source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        pass

    raise TypeError(f"Unsupported image source type: {type(source)}")


class PuzzleReader:
    """Extracts digits from a sudoku puzzle image."""

    def __init__(
        self,
        model: SudokuNet | None = None,
        model_hex: SudokuNet | None = None,
    ) -> None:
        """Create a PuzzleReader.

        Args:
            model: Model for standard grids (≤9×9).  Used for all grids if
                   model_hex is None.
            model_hex: Model for large/hex grids (>9×9, e.g. 16×16).
                       Falls back to model when None.
        """
        self._model = model
        self._model_hex = model_hex

    @classmethod
    def from_weights_dir(cls, weights_dir: str | Path) -> "PuzzleReader":
        """Create a PuzzleReader by loading models from a weights directory.

        Looks for (in order of preference):
          - digits_1_9.pt  → standard model (1-9)
          - digits_hex.pt  → hex model (1-9+A-F)
          - digit_classifier.pt  → legacy fallback (used for both)

        Args:
            weights_dir: Directory containing .pt weight files.

        Returns:
            A PuzzleReader with models loaded from the directory.
        """
        d = Path(weights_dir)

        def _try_load(name: str) -> SudokuNet | None:
            p = d / name
            if p.exists():
                return SudokuNet(p)
            return None

        model     = _try_load(_MODEL_STANDARD) or _try_load(_MODEL_LEGACY)
        model_hex = _try_load(_MODEL_HEX)

        # If only a legacy file exists, use it for both
        if model is None and model_hex is not None:
            model = model_hex

        return cls(model=model, model_hex=model_hex)

    def _get_model(self, grid_size: int) -> SudokuNet | None:
        """Return the appropriate model for the given grid size."""
        if grid_size > _HEX_GRID_THRESHOLD and self._model_hex is not None:
            return self._model_hex
        return self._model

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

        active_model = self._get_model(grid_size)
        if active_model is not None:
            for cell in cells:
                if cell.has_digit and cell.grayscale_image is not None:
                    cell.digit = active_model.predict(cell.grayscale_image)

        return color_warped, cells

    def read_digits_string(
        self, image: ImageSource, grid_size: int | None = None,
    ) -> str:
        """Read puzzle digits as a compact string.

        Args:
            image: A file path or PIL Image of a sudoku puzzle.
            grid_size: Number of rows/columns, or None to auto-detect.

        Returns:
            A string of length grid_size² with digits and '.' for empty cells.
            E.g. "83.469.5.549.876.3..."
        """
        _, cells = self.read_cells(image, grid_size)
        gs = int(len(cells) ** 0.5)
        active_model = self._get_model(gs)
        chars = active_model.chars if active_model is not None else "0123456789"
        return "".join(chars[cell.digit] if cell.digit is not None else "." for cell in cells)

    def read_digits(
        self, image: ImageSource, grid_size: int | None = None,
    ) -> list[int | None]:
        """Read all cells from a sudoku puzzle image.

        Args:
            image: A file path or PIL Image of a sudoku puzzle.
            grid_size: Number of rows/columns, or None to auto-detect.

        Returns:
            A flat list of values in row-major order.
            Each value is a label index (into the active model's chars) or None.
        """
        _, cells = self.read_cells(image, grid_size)
        return [cell.digit for cell in cells]
