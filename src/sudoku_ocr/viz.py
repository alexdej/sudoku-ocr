"""Visualization utilities for debugging sudoku OCR results."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .cells import _find_line_positions
from .types import CellInfo


def draw_overlay(
    color_warped: np.ndarray,
    cells: list[CellInfo],
    grid_size: int | None = None,
    chars: str | None = None,
) -> np.ndarray:
    """Draw detected digits and grid lines over the warped puzzle image.

    Uses the same adaptive line detection as cell segmentation so the
    overlay matches actual cell boundaries.

    Args:
        color_warped: Perspective-corrected color image of the grid.
        cells: List of CellInfo objects with digit predictions.
        grid_size: Number of rows/columns, or None to infer from cells.

    Returns:
        A BGR image with the overlay drawn.
    """
    if grid_size is None:
        grid_size = max(c.row for c in cells) + 1 if cells else 9
    overlay = color_warped.copy()
    h, w = overlay.shape[:2]

    # Find actual line positions (same as cell segmentation uses)
    gray = cv2.cvtColor(color_warped, cv2.COLOR_BGR2GRAY) if len(color_warped.shape) == 3 else color_warped
    row_lines = _find_line_positions(gray, grid_size, "horizontal")
    col_lines = _find_line_positions(gray, grid_size, "vertical")

    # Draw grid lines — thicker at box boundaries
    box_size = int(math.isqrt(grid_size)) or 1
    for i, y in enumerate(row_lines):
        thickness = 2 if i % box_size == 0 else 1
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), thickness)
    for i, x in enumerate(col_lines):
        thickness = 2 if i % box_size == 0 else 1
        cv2.line(overlay, (x, 0), (x, h), (0, 0, 0), thickness)

    # Draw digits centered in their actual cell boundaries
    font = cv2.FONT_HERSHEY_SIMPLEX
    avg_cell_h = h / grid_size
    for cell in cells:
        if cell.digit is None:
            continue

        y1 = row_lines[cell.row]
        y2 = row_lines[cell.row + 1]
        x1 = col_lines[cell.col]
        x2 = col_lines[cell.col + 1]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        label = chars[cell.digit] if chars is not None else str(cell.digit)
        font_scale = avg_cell_h / 50.0
        thickness = max(1, int(font_scale * 2))

        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2

        pad = 4
        cv2.rectangle(
            overlay,
            (tx - pad, ty - text_size[1] - pad),
            (tx + text_size[0] + pad, ty + pad),
            (255, 255, 255),
            -1,
        )
        cv2.putText(overlay, label, (tx, ty), font, font_scale, (0, 0, 200), thickness)

    return overlay
