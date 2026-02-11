"""Cell segmentation and digit extraction from a warped sudoku grid."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import clear_border

from .types import CellInfo

# Minimum fraction of cell area a contour must occupy to be considered a digit
MIN_DIGIT_AREA_RATIO = 0.03

# Padding (fraction of cell size) to trim from cell edges before extraction
CELL_PADDING_RATIO = 0.08

# Minimum cell size (pixels) before upscaling is applied
MIN_CELL_SIZE = 50


def _extract_digit_region(cell_gray: np.ndarray) -> np.ndarray | None:
    """Extract the digit region from a single grayscale cell.

    Returns a cleaned binary image of the digit, or None if the cell is empty.
    """
    thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    cell_area = cell_gray.shape[0] * cell_gray.shape[1]
    if cv2.contourArea(largest) / cell_area < MIN_DIGIT_AREA_RATIO:
        return None

    # Mask to isolate the digit
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return cv2.bitwise_and(thresh, thresh, mask=mask)


def _find_line_positions(
    gray_warped: np.ndarray, grid_size: int, direction: str,
) -> list[int]:
    """Find actual grid line positions along one axis.

    Uses morphological line extraction and projection to find where
    grid lines actually are, rather than assuming uniform spacing.

    Args:
        gray_warped: Grayscale warped grid image.
        grid_size: Expected number of cells along this axis.
        direction: 'horizontal' or 'vertical'.

    Returns:
        Sorted list of pixel positions for each grid line (grid_size + 1 values,
        including the outer edges).
    """
    h, w = gray_warped.shape[:2]
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    # Morphological filter to isolate lines in the given direction
    if direction == "horizontal":
        length = max(w // 12, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        profile = np.sum(lines_img, axis=1).astype(np.float64)
        total = h
    else:
        length = max(h // 12, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        profile = np.sum(lines_img, axis=0).astype(np.float64)
        total = w

    # Find peaks in the profile
    threshold = np.mean(profile) + 0.5 * np.std(profile)
    min_dist = total // (grid_size * 2)  # At least half a cell apart

    positions = []
    in_peak = False
    peak_start = 0
    for i, val in enumerate(profile):
        if val > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif val <= threshold and in_peak:
            in_peak = False
            center = (peak_start + i) // 2
            if not positions or center - positions[-1] >= min_dist:
                positions.append(center)

    # If we're still in a peak at the end
    if in_peak:
        center = (peak_start + total) // 2
        if not positions or center - positions[-1] >= min_dist:
            positions.append(center)

    # We need exactly grid_size + 1 lines (including outer edges).
    # If detection found the right count, use them.
    # Otherwise, ensure outer edges are present and interpolate/trim as needed.
    expected = grid_size + 1

    # Always include the outer edges
    if not positions or positions[0] > total // grid_size:
        positions.insert(0, 0)
    if positions[-1] < total - total // grid_size:
        positions.append(total)

    if len(positions) == expected:
        return positions

    # If we have too many, keep the ones that best match expected spacing
    if len(positions) > expected:
        # Score each subset — but that's expensive. Instead, greedily
        # pick the best-spaced lines.
        result = [positions[0]]
        ideal_step = (positions[-1] - positions[0]) / grid_size
        for i in range(1, grid_size):
            ideal_pos = positions[0] + i * ideal_step
            best = min(positions, key=lambda p: abs(p - ideal_pos))
            if best != result[-1]:
                result.append(best)
        result.append(positions[-1])
        if len(result) == expected:
            return sorted(set(result))

    # Fallback: uniform spacing
    step = total / grid_size
    return [int(round(i * step)) for i in range(expected)]


def segment_cells(
    color_warped: np.ndarray,
    gray_warped: np.ndarray,
    grid_size: int = 9,
) -> list[CellInfo]:
    """Divide a warped grid image into cells and extract digit regions.

    Uses adaptive line detection to find actual grid line positions
    rather than assuming uniform spacing.

    Args:
        color_warped: Perspective-corrected color image of the grid.
        gray_warped: Perspective-corrected grayscale image of the grid.
        grid_size: Number of rows/columns (default 9).

    Returns:
        A list of CellInfo objects, one per cell, in row-major order.
    """
    h, w = gray_warped.shape[:2]

    # Upscale small grids so cells are large enough for reliable extraction
    cell_size = min(w, h) / grid_size
    if cell_size < MIN_CELL_SIZE:
        scale = MIN_CELL_SIZE / cell_size
        new_w, new_h = int(w * scale), int(h * scale)
        gray_warped = cv2.resize(gray_warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        color_warped = cv2.resize(color_warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = new_h, new_w

    # Find actual grid line positions
    row_lines = _find_line_positions(gray_warped, grid_size, "horizontal")
    col_lines = _find_line_positions(gray_warped, grid_size, "vertical")

    cells: list[CellInfo] = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row_lines[row]
            y2 = row_lines[row + 1]
            x1 = col_lines[col]
            x2 = col_lines[col + 1]

            cell_h = y2 - y1
            cell_w = x2 - x1
            if cell_h <= 0 or cell_w <= 0:
                cells.append(CellInfo(row=row, col=col))
                continue

            color_cell = color_warped[y1:y2, x1:x2]

            # Apply padding for digit extraction to reduce grid line interference
            pad_y = int(cell_h * CELL_PADDING_RATIO)
            pad_x = int(cell_w * CELL_PADDING_RATIO)
            inner_gray = gray_warped[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
            digit_region = _extract_digit_region(inner_gray)

            cells.append(
                CellInfo(
                    row=row,
                    col=col,
                    color_image=color_cell,
                    grayscale_image=digit_region,
                    has_digit=digit_region is not None,
                )
            )

    return cells
