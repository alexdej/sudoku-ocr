"""Cell segmentation and digit extraction from a warped sudoku grid."""

from __future__ import annotations

import cv2
import numpy as np

from .grid import _find_line_positions
from .types import CellInfo

# Minimum fraction of cell area a contour must occupy to be considered a digit
MIN_DIGIT_AREA_RATIO = 0.015

# Minimum gap (in pixel values) between the cell mean and the Otsu threshold.
# When Otsu ≈ mean the image is unimodal noise (paper texture, JPEG artifacts)
# with no real foreground/background split — return None immediately.
MIN_BIMODALITY = 5

# Minimum fraction of inner cell height the digit contour must span.
# Filters out noise blobs that occupy only a small corner of the cell.
MIN_DIGIT_HEIGHT_RATIO = 0.40

# Padding (fraction of cell size) to trim from cell edges before extraction
CELL_PADDING_RATIO = 0.08

# Minimum cell size (pixels) before upscaling is applied
MIN_CELL_SIZE = 50


def _clear_border(img: np.ndarray) -> np.ndarray:
    """Zero out any foreground pixels whose connected component touches the border.

    Uses 8-connectivity, matching the behaviour of skimage.segmentation.clear_border.
    """
    n_labels, labels = cv2.connectedComponents(img, connectivity=8)
    h, w = img.shape
    border_labels: set[int] = set()
    border_labels.update(int(v) for v in labels[0, :])
    border_labels.update(int(v) for v in labels[h - 1, :])
    border_labels.update(int(v) for v in labels[:, 0])
    border_labels.update(int(v) for v in labels[:, w - 1])
    border_labels.discard(0)  # 0 is background
    if not border_labels:
        return img.copy()
    cleared = img.copy()
    for label in border_labels:
        cleared[labels == label] = 0
    return cleared


def _extract_digit_region(cell_gray: np.ndarray) -> np.ndarray | None:
    """Extract the digit region from a single grayscale cell.

    Returns a cleaned binary image of the digit, or None if the cell is empty.
    """
    otsu_val, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # If the Otsu threshold is very close to the cell mean the image has no
    # real foreground/background split — it is uniform noise (paper grain, JPEG
    # artifacts, faint shadows).  Splitting such a distribution produces many
    # tiny spurious contours that can pass the downstream area/height/density
    # filters by chance.
    if abs(float(cell_gray.mean()) - float(otsu_val)) < MIN_BIMODALITY:
        return None
    source = _clear_border(thresh)

    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # clear_border eliminated everything — likely a border-touching digit (common in
        # photos where large printed strokes reach the padded cell edge, e.g. the top bar
        # of a "7").  Fall back to the uncleared threshold, filtering only thin elongated
        # border artifacts (grid-line remnants) rather than all border-touching pixels.
        source = thresh
        h_img, w_img = thresh.shape
        raw, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cell_area = h_img * w_img

        def _is_line_artifact(cnt: np.ndarray) -> bool:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x > 0 and y > 0 and x + cw < w_img and y + ch < h_img:
                return False  # fully interior — keep
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            return aspect > 5  # very elongated + border-touching → grid line

        # Reject (a) thin border-touching lines (grid remnants), (b) full-cell background
        # blobs produced by Otsu-BINARY_INV on dark-background cells (> 60 % area),
        # and (c) contours shorter than half the cell (noise is typically short).
        contours = [c for c in raw
                    if not _is_line_artifact(c)
                    and cv2.contourArea(c) / cell_area <= 0.6
                    and cv2.boundingRect(c)[3] / h_img >= 0.5]
        if not contours:
            return None

    largest = max(contours, key=cv2.contourArea)
    cell_area = cell_gray.shape[0] * cell_gray.shape[1]
    if cv2.contourArea(largest) / cell_area < MIN_DIGIT_AREA_RATIO:
        return None

    _, _, bbox_w, ch = cv2.boundingRect(largest)
    if ch / cell_gray.shape[0] < MIN_DIGIT_HEIGHT_RATIO:
        return None

    # Mask to isolate the digit
    mask = np.zeros(source.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    result = cv2.bitwise_and(source, source, mask=mask)

    # Reject sparse noise: real printed digits fill ≥ 20 % of their bounding box.
    # Paper-texture noise has many gaps (~15 %), while digits — even a thin "1" whose
    # bbox is just as narrow as the stroke — score 40–100 %.
    if bbox_w > 0 and ch > 0 and cv2.countNonZero(result) / (bbox_w * ch) < 0.20:
        return None

    return result


def _classify_digit_color(inner_color: np.ndarray, digit_mask: np.ndarray) -> bool | None:
    """Classify a digit as given (True) or fill (False) based on its color.

    Uses the binary digit mask to sample pixel colors from the color cell
    image, then examines HSV values to distinguish:
      - Dark digits (black, dark navy) → given (True)
      - Bright/saturated digits (blue, red) → fill (False)
      - Indeterminate → None

    Args:
        inner_color: Padded color crop of the cell (BGR).
        digit_mask: Binary image of digit pixels (same spatial dimensions).

    Returns:
        True if given, False if fill, None if indeterminate.
    """
    if inner_color.shape[:2] != digit_mask.shape[:2]:
        return None

    digit_pixels = inner_color[digit_mask > 0]
    if len(digit_pixels) < 10:
        return None

    hsv = cv2.cvtColor(digit_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    median_s = float(np.median(hsv[:, 1]))  # saturation 0–255
    median_v = float(np.median(hsv[:, 2]))  # value/brightness 0–255

    # Very dark digit (black) → given
    if median_v < 80:
        return True

    # Bright and saturated digit (blue, red) → fill
    if median_s > 80 and median_v > 130:
        return False

    # Dark but saturated (dark navy, as in some apps' given style) → given
    if median_v < 130:
        return True

    return None


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
            inner_color = color_warped[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
            digit_region = _extract_digit_region(inner_gray)

            # If the inner cell is entirely white but the outer cell has dark
            # pixels, the digit likely sits in the padding zone.  Retry with a
            # minimal strip (3%) so grid-line remnants are still excluded.
            if digit_region is None and float(inner_gray.mean()) > 250:
                outer_cell = gray_warped[y1:y2, x1:x2]
                if int(np.sum(outer_cell < 150)) > 20:
                    slim_y = max(1, int(cell_h * 0.03))
                    slim_x = max(1, int(cell_w * 0.03))
                    slim_gray = gray_warped[y1 + slim_y : y2 - slim_y, x1 + slim_x : x2 - slim_x]
                    slim_color = color_warped[y1 + slim_y : y2 - slim_y, x1 + slim_x : x2 - slim_x]
                    digit_region = _extract_digit_region(slim_gray)
                    if digit_region is not None:
                        inner_color = slim_color

            is_given: bool | None = None
            if digit_region is not None:
                is_given = _classify_digit_color(inner_color, digit_region)

            cells.append(
                CellInfo(
                    row=row,
                    col=col,
                    color_image=color_cell,
                    grayscale_image=digit_region,
                    has_digit=digit_region is not None,
                    is_given=is_given,
                )
            )

    return cells
