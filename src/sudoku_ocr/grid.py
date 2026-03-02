"""Grid detection and perspective correction for sudoku puzzle images."""

from __future__ import annotations

import cv2
import numpy as np


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]  # bottom-left has largest difference
    return rect


def _warp_perspective(
    image: np.ndarray, corners: np.ndarray
) -> np.ndarray:
    """Apply perspective transform to extract a top-down rectangular view."""
    rect = _order_corners(corners)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _is_reasonable_grid(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """Check that detected corners form a roughly square, reasonably-sized grid."""
    ordered = _order_corners(corners.astype(np.float32))
    tl, tr, br, bl = ordered

    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    if width < 1 or height < 1:
        return False

    # Reject if too narrow — sudoku grids are roughly square
    aspect = max(width, height) / min(width, height)
    if aspect > 1.3:
        return False

    # Reject if too small relative to the image
    img_h, img_w = image_shape[:2]
    grid_area = width * height
    image_area = img_w * img_h
    if grid_area / image_area < 0.05:
        return False

    return True


def _find_grid_contour(thresh: np.ndarray) -> np.ndarray | None:
    """Find the largest 4-point contour, assumed to be the puzzle grid.

    Tries progressively looser polygon approximation and falls back to
    minAreaRect for contours that are nearly rectangular but have extra
    vertices (e.g. when text headers merge with the grid border).
    """
    for mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
        contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:10]:
            peri = cv2.arcLength(contour, True)

            # Try progressively larger epsilon to simplify near-rectangular contours
            for eps in (0.02, 0.04, 0.06):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    if _is_reasonable_grid(pts, thresh.shape):
                        return pts

            # Last resort: use minAreaRect for large contours with >4 vertices
            area = cv2.contourArea(contour)
            img_area = thresh.shape[0] * thresh.shape[1]
            if area / img_area > 0.05:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                pts = box.astype(np.float32)
                if _is_reasonable_grid(pts, thresh.shape):
                    return pts

    return None


def _find_grid_hough(thresh: np.ndarray) -> np.ndarray | None:
    """Fallback: find grid boundary using Hough line detection."""
    lines = cv2.HoughLinesP(
        thresh, rho=1, theta=np.pi / 180, threshold=100,
        minLineLength=thresh.shape[1] // 4, maxLineGap=10,
    )
    if lines is None:
        return None

    # Collect all line endpoints
    points = lines.reshape(-1, 2)
    if len(points) < 4:
        return None

    # Find the convex hull and approximate to 4 corners
    hull = cv2.convexHull(points)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)

    # If we can't get exactly 4 points, use bounding rect of all points
    x, y, w, h = cv2.boundingRect(points)
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


KNOWN_GRID_SIZES = (4, 6, 9, 12, 16)


def _count_lines(profile: np.ndarray, min_distance: int) -> int:
    """Count peaks in a 1D projection profile.

    A peak is a local region where the profile exceeds a threshold,
    representing a grid line.
    """
    threshold = np.mean(profile) + 0.5 * np.std(profile)
    above = profile > threshold

    # Find runs of consecutive above-threshold values (each run = one line)
    count = 0
    last_peak = -min_distance
    in_peak = False
    for i, val in enumerate(above):
        if val and not in_peak:
            if i - last_peak >= min_distance:
                count += 1
                last_peak = i
            in_peak = True
        elif not val:
            in_peak = False
    return count


def _extract_lines(thresh: np.ndarray, direction: str) -> np.ndarray:
    """Use morphological operations to isolate grid lines in one direction.

    Args:
        thresh: Binary (inverted) thresholded image.
        direction: 'horizontal' or 'vertical'.

    Returns:
        Binary image containing only the detected lines.
    """
    h, w = thresh.shape
    if direction == "horizontal":
        # Long horizontal kernel suppresses digits, keeps horizontal lines.
        length = max(w // 12, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    else:
        length = max(h // 12, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return lines


def detect_grid_size(gray_warped: np.ndarray) -> int:
    """Detect the grid size from a perspective-corrected grayscale image.

    Isolates grid lines using directional morphological filters, then
    projects onto each axis and counts lines.

    Args:
        gray_warped: Perspective-corrected grayscale image of the grid.

    Returns:
        Detected grid size (e.g. 9 for a standard 9x9 puzzle).

    Raises:
        ValueError: If the grid size could not be determined.
    """
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    h, w = thresh.shape

    # Extract only horizontal and vertical lines (removes digits)
    h_lines_img = _extract_lines(thresh, "horizontal")
    v_lines_img = _extract_lines(thresh, "vertical")

    # Project: sum horizontal line image along columns → one value per row
    h_profile = np.sum(h_lines_img, axis=1).astype(np.float64)
    # Project: sum vertical line image along rows → one value per column
    v_profile = np.sum(v_lines_img, axis=0).astype(np.float64)

    # Minimum distance between lines — prevents double-counting thick lines.
    min_dist_h = h // 25
    min_dist_v = w // 25

    h_count = _count_lines(h_profile, min_dist_h)
    v_count = _count_lines(v_profile, min_dist_v)

    # Grid lines = grid_size + 1, so cells = lines - 1.
    # In photos, one direction may detect only thick box separators while
    # the other detects all lines.  When counts diverge, trust the higher
    # one — it's more likely to have found the actual cell lines.
    h_cells = h_count - 1
    v_cells = v_count - 1

    if abs(h_cells - v_cells) <= 2:
        avg_cells = (h_cells + v_cells) / 2.0
    else:
        avg_cells = float(max(h_cells, v_cells))

    best_size = min(KNOWN_GRID_SIZES, key=lambda s: abs(s - avg_cells))

    # Sanity check — if we're way off from any known size, fail
    if abs(best_size - avg_cells) > 2:
        raise ValueError(
            f"Could not determine grid size: detected ~{avg_cells:.1f} cells "
            f"({h_count} horizontal, {v_count} vertical lines)"
        )

    return best_size


# Minimum size (shorter side in pixels) before upscaling is applied.
# Small images produce blobs in adaptive threshold and lose thin grid lines.
_MIN_IMAGE_DIM = 300


def detect_grid(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect the sudoku grid and return perspective-corrected images.

    Args:
        image: Input image in BGR format.

    Returns:
        A tuple of (color_warped, grayscale_warped) images showing the
        grid in a clean top-down view.

    Raises:
        ValueError: If no grid could be detected in the image.
    """
    h, w = image.shape[:2]
    if min(h, w) < _MIN_IMAGE_DIM:
        scale = _MIN_IMAGE_DIM / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    # Try contour detection first, fall back to Hough lines
    corners = _find_grid_contour(thresh)
    if corners is None:
        corners = _find_grid_hough(thresh)
        if corners is not None and not _is_reasonable_grid(corners, image.shape):
            corners = None
    if corners is None:
        raise ValueError("Could not detect a sudoku grid in the image.")

    corners = corners.astype(np.float32)
    color_warped = _warp_perspective(image, corners)
    gray_warped = _warp_perspective(gray, corners)

    return color_warped, gray_warped
