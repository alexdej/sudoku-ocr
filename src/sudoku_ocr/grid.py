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

    threshold = np.mean(profile) + 0.5 * np.std(profile)
    min_dist = total // (grid_size * 2)

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

    if in_peak:
        center = (peak_start + total) // 2
        if not positions or center - positions[-1] >= min_dist:
            positions.append(center)

    expected = grid_size + 1

    if not positions or positions[0] > total // grid_size:
        positions.insert(0, 0)
    if positions[-1] < total - total // grid_size:
        positions.append(total)

    if len(positions) == expected:
        return positions

    if len(positions) > expected:
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

    step = total / grid_size
    return [int(round(i * step)) for i in range(expected)]


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


def _preprocess_for_detection(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upscale, grayscale, blur, and threshold a BGR image for grid detection.

    Args:
        image: Input image in BGR format.

    Returns:
        (scaled_bgr, gray, thresh_inv) where scaled_bgr is possibly upscaled,
        gray is its grayscale, and thresh_inv is the inverted adaptive threshold
        (white foreground on black background, ready for contour/Hough detection).
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
    thresh_inv = cv2.bitwise_not(thresh)
    return image, gray, thresh_inv


def _find_grid_intersections(
    gray_warped: np.ndarray,
    grid_size: int,
) -> np.ndarray | None:
    """Detect the 2D coordinates of all grid line intersections.

    Uses 1D line positions as seeds, then refines each intersection by
    computing the weighted centroid of morphologically extracted line images
    in a local window around each expected location.

    Args:
        gray_warped: Perspective-corrected grayscale image.
        grid_size: Number of cells per axis.

    Returns:
        Float32 array of shape (grid_size+1, grid_size+1, 2) with (x, y) for
        each intersection, or None if line detection yields the wrong count.
    """
    h, w = gray_warped.shape[:2]

    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh_inv = cv2.bitwise_not(cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    ))
    h_img = _extract_lines(thresh_inv, "horizontal")
    v_img = _extract_lines(thresh_inv, "vertical")

    row_pos = _find_line_positions(gray_warped, grid_size, "horizontal")
    col_pos = _find_line_positions(gray_warped, grid_size, "vertical")

    n = grid_size + 1
    if len(row_pos) != n or len(col_pos) != n:
        return None

    # Dilate both line images slightly so horizontal and vertical strokes
    # overlap at each crossing, then AND them to get a bright spot at each
    # intersection.  3×3 dilation adds ±1 px — enough to bridge sub-pixel gaps
    # while keeping each spot tightly centred on the true crossing.
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cross_img = cv2.bitwise_and(cv2.dilate(h_img, dil_k), cv2.dilate(v_img, dil_k))

    cell_h = h / grid_size
    cell_w = w / grid_size
    # Wide search window: 48 % of cell size so the window still finds lines
    # that deviate significantly from the 1-D average seed (curved paper).
    # Adjacent lines are ~100 % of cell_h apart, so ±48 % is safely clear.
    wy = max(int(cell_h * 0.48), 6)
    wx = max(int(cell_w * 0.48), 6)
    # Fallback strip half-widths for the separate H/V projections.
    ay = max(int(cell_h * 0.20), 2)
    ax = max(int(cell_w * 0.20), 2)

    pts = np.empty((n, n, 2), dtype=np.float32)

    for i, ry in enumerate(row_pos):
        for j, cx in enumerate(col_pos):
            ri, ci = int(round(ry)), int(round(cx))
            y1 = max(0, ri - wy);  y2 = min(h, ri + wy + 1)
            x1 = max(0, ci - wx);  x2 = min(w, ci + wx + 1)

            # Primary: true 2-D centroid of the H∩V crossing region.
            patch = cross_img[y1:y2, x1:x2].astype(np.float64)
            s = patch.sum()
            if s > 0:
                ph, pw = patch.shape
                actual_y = y1 + float(np.dot(np.arange(ph), patch.sum(axis=1)) / s)
                actual_x = x1 + float(np.dot(np.arange(pw), patch.sum(axis=0)) / s)
            else:
                # Fallback: independent H and V 1-D centroids.
                xa1, xa2 = max(0, ci - ax), min(w, ci + ax + 1)
                prof_y = h_img[y1:y2, xa1:xa2].sum(axis=1).astype(np.float64)
                sy = prof_y.sum()
                actual_y = (y1 + float(np.dot(np.arange(len(prof_y)), prof_y) / sy)
                            if sy > 0 else float(ry))

                ya1, ya2 = max(0, ri - ay), min(h, ri + ay + 1)
                prof_x = v_img[ya1:ya2, x1:x2].sum(axis=0).astype(np.float64)
                sx = prof_x.sum()
                actual_x = (x1 + float(np.dot(np.arange(len(prof_x)), prof_x) / sx)
                            if sx > 0 else float(cx))

            pts[i, j] = [actual_x, actual_y]

    return pts


def _build_mesh_remap(
    src_pts: np.ndarray,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense remap arrays from a sparse grid of source intersection points.

    For each output pixel, bilinearly interpolates the corresponding source
    coordinate using the four surrounding intersection points.

    Args:
        src_pts: Shape (n, n, 2) source intersection (x, y) coordinates.
        out_h: Output image height.
        out_w: Output image width.

    Returns:
        (map_x, map_y): float32 arrays of shape (out_h, out_w) for cv2.remap.
    """
    n = src_pts.shape[0]  # grid_size + 1
    tx = np.linspace(0.0, float(out_w - 1), n)
    ty = np.linspace(0.0, float(out_h - 1), n)

    map_x = np.empty((out_h, out_w), dtype=np.float32)
    map_y = np.empty((out_h, out_w), dtype=np.float32)

    for row in range(n - 1):
        for col in range(n - 1):
            tl = src_pts[row,     col    ]
            tr = src_pts[row,     col + 1]
            bl = src_pts[row + 1, col    ]
            br = src_pts[row + 1, col + 1]

            ox1, ox2 = int(round(tx[col])), int(round(tx[col + 1]))
            oy1, oy2 = int(round(ty[row])), int(round(ty[row + 1]))
            ox1c, ox2c = max(0, ox1), min(out_w - 1, ox2)
            oy1c, oy2c = max(0, oy1), min(out_h - 1, oy2)

            if ox2c < ox1c or oy2c < oy1c:
                continue

            xs = np.arange(ox1c, ox2c + 1, dtype=np.float32)
            ys = np.arange(oy1c, oy2c + 1, dtype=np.float32)
            t = np.clip((xs - ox1) / max(ox2 - ox1, 1), 0.0, 1.0)
            s = np.clip((ys - oy1) / max(oy2 - oy1, 1), 0.0, 1.0)
            T, S = np.meshgrid(t, s)

            map_x[oy1c:oy2c + 1, ox1c:ox2c + 1] = (
                (1 - S) * ((1 - T) * tl[0] + T * tr[0])
                +    S  * ((1 - T) * bl[0] + T * br[0])
            )
            map_y[oy1c:oy2c + 1, ox1c:ox2c + 1] = (
                (1 - S) * ((1 - T) * tl[1] + T * tr[1])
                +    S  * ((1 - T) * bl[1] + T * br[1])
            )

    return map_x, map_y


# Only apply the mesh warp when the interior intersections deviate from a flat
# bilinear grid by at least this many pixels (RMS).  Below this the grid is
# already straight enough that the warp would just add interpolation blur.
# Calibrated on sample set: screenshots/flat-paper ≤ 11 px; genuine paper-curl
# (EFJ5JjIXYAEvSse.jpg) ≈ 21 px.
_MIN_WARP_RMS_PX: float = 15.0

# Minimum fraction of expected cells that must be directly detected by the
# contour mesh before it is trusted.  Low coverage means most intersections
# are extrapolated; at that point the centroid method is more reliable.
_CONTOUR_MESH_MIN_COVERAGE: float = 0.30


def _intersection_rms(pts: np.ndarray) -> float:
    """RMS deviation of interior intersections from the bilinear blend of the corners.

    Measures how much the grid curves *internally* (independent of overall
    perspective tilt or margins).  Corner points always score 0 by construction.

    Args:
        pts: Shape (n, n, 2) intersection coordinates.

    Returns:
        RMS pixel error over all interior (non-corner) intersections.
    """
    n = pts.shape[0]
    tl, tr = pts[0, 0], pts[0, -1]
    bl, br = pts[-1, 0], pts[-1, -1]
    sq_err = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if (i == 0 or i == n - 1) and (j == 0 or j == n - 1):
                continue  # skip the four corners themselves
            t = j / (n - 1)
            s = i / (n - 1)
            ux = (1 - s) * ((1 - t) * tl[0] + t * tr[0]) + s * ((1 - t) * bl[0] + t * br[0])
            uy = (1 - s) * ((1 - t) * tl[1] + t * tr[1]) + s * ((1 - t) * bl[1] + t * br[1])
            sq_err += (pts[i, j, 0] - ux) ** 2 + (pts[i, j, 1] - uy) ** 2
            count += 1
    return float(np.sqrt(sq_err / max(count, 1)))


def _find_contour_mesh(
    gray_warped: np.ndarray,
    grid_size: int,
    epsilon: float = 0.05,
) -> tuple[np.ndarray, float] | None:
    """Build an intersection mesh by detecting cell-interior contour holes.

    The key insight is that the dark grid lines form one connected foreground
    region whose *holes* (RETR_CCOMP level-1 contours) are exactly the white
    cell interiors.  Each hole is approximated to a quad, its four corners are
    clustered onto the nearest expected grid intersection, and gaps left by
    cells that couldn't be detected (e.g. digit-clipped or merged holes) are
    filled first by linear extrapolation from pairs of detected neighbors,
    then by the centroid intersection method as a fallback.

    This directly observes where each cell boundary is rather than tracking
    grid-line centroids, making it especially robust for curved or warped paper
    where lines deviate significantly from a uniform grid.

    Args:
        gray_warped: Perspective-corrected grayscale image.
        grid_size:   Number of cells per axis.
        epsilon:     Polygon approximation parameter (fraction of perimeter).

    Returns:
        ``(mesh_pts, coverage)`` where *mesh_pts* is a
        ``(grid_size+1, grid_size+1, 2)`` float32 array of intersection
        coordinates and *coverage* is the fraction of expected cells directly
        detected.  Returns ``None`` if no cell quads are found.
    """
    h, w = gray_warped.shape[:2]
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh_inv = cv2.bitwise_not(cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    ))

    exp_area = (h * w) / grid_size ** 2
    area_lo, area_hi = 0.3 * exp_area, 2.0 * exp_area

    ctrs, hier = cv2.findContours(thresh_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None

    # Level-1 contours (parent index ≥ 0) are holes in the grid line region —
    # exactly the white cell interiors.  Keep only those whose area is roughly
    # one cell's worth to reject noise and the outer image border.
    quads = []
    for i, cnt in enumerate(ctrs):
        if hier[0][i][3] < 0:
            continue
        area = cv2.contourArea(cnt)
        if not (area_lo < area < area_hi):
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))

    if not quads:
        return None

    coverage = len(quads) / grid_size ** 2

    # Cluster detected quad corners onto the nearest expected intersection.
    row_pos = _find_line_positions(gray_warped, grid_size, "horizontal")
    col_pos = _find_line_positions(gray_warped, grid_size, "vertical")
    n = grid_size + 1
    row_arr = np.array(row_pos, dtype=np.float32)
    col_arr = np.array(col_pos, dtype=np.float32)

    accum  = np.zeros((n, n, 2), dtype=np.float64)
    counts = np.zeros((n, n),    dtype=np.int32)
    for q in quads:
        for pt in q:
            i = int(np.argmin(np.abs(row_arr - pt[1])))
            j = int(np.argmin(np.abs(col_arr - pt[0])))
            accum[i, j]  += pt
            counts[i, j] += 1

    detected = counts > 0
    mesh_pts = np.empty((n, n, 2), dtype=np.float32)
    mesh_pts[detected] = (accum[detected] / counts[detected, np.newaxis]).astype(np.float32)

    # Fill gaps: linear extrapolation from pairs of detected neighbors first
    # (follows local curvature); centroid method as fallback; 1-D seed last.
    fallback = _find_grid_intersections(gray_warped, grid_size)
    for i in range(n):
        for j in range(n):
            if detected[i, j]:
                continue
            estimates = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                i1, j1 = i - di, j - dj
                i2, j2 = i - 2 * di, j - 2 * dj
                if (0 <= i1 < n and 0 <= j1 < n and detected[i1, j1]
                        and 0 <= i2 < n and 0 <= j2 < n and detected[i2, j2]):
                    estimates.append(2.0 * mesh_pts[i1, j1] - mesh_pts[i2, j2])
            if estimates:
                mesh_pts[i, j] = np.mean(estimates, axis=0)
            elif fallback is not None:
                mesh_pts[i, j] = fallback[i, j]
            else:
                mesh_pts[i, j] = [col_arr[j], row_arr[i]]

    return mesh_pts, coverage


def refine_grid_warp(
    color_warped: np.ndarray,
    gray_warped: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Correct residual distortion after perspective correction via a mesh warp.

    First attempts a contour-based mesh (``_find_contour_mesh``) that directly
    observes cell-boundary positions, which is especially effective for curved
    or warped paper.  Falls back to the centroid intersection method
    (``_find_grid_intersections``) when cell coverage is too low to trust the
    contour mesh.

    The warp is only applied when the interior intersections deviate from a
    flat bilinear grid by at least ``_MIN_WARP_RMS_PX`` pixels.  For
    screenshots or flat photos the grid is already straight and the images are
    returned unchanged.

    Args:
        color_warped: Perspective-corrected BGR image.
        gray_warped:  Perspective-corrected grayscale image.
        grid_size:    Number of cells per axis.

    Returns:
        (color_refined, gray_refined, intersection_pts) where intersection_pts
        is the (grid_size+1, grid_size+1, 2) detection array (useful for
        visualisation), or None if intersection detection failed entirely.
        When distortion is below the threshold the images are returned
        unchanged and intersection_pts is still provided.
    """
    # Try contour-based mesh first: it directly observes cell boundaries and
    # is more robust than the centroid method for curved/warped paper.
    contour_result = _find_contour_mesh(gray_warped, grid_size)
    if contour_result is not None:
        mesh_pts, coverage = contour_result
        if (coverage >= _CONTOUR_MESH_MIN_COVERAGE
                and _intersection_rms(mesh_pts) >= _MIN_WARP_RMS_PX):
            h, w = gray_warped.shape[:2]
            map_x, map_y = _build_mesh_remap(mesh_pts, h, w)
            color_ref = cv2.remap(
                color_warped, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray_ref = cv2.remap(
                gray_warped, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return color_ref, gray_ref, mesh_pts

    # Fall back to centroid intersection method.
    pts = _find_grid_intersections(gray_warped, grid_size)
    if pts is None:
        return color_warped, gray_warped, None

    if _intersection_rms(pts) < _MIN_WARP_RMS_PX:
        # Grid is already straight — return originals with pts for visualisation.
        return color_warped, gray_warped, pts

    h, w = gray_warped.shape[:2]
    map_x, map_y = _build_mesh_remap(pts, h, w)
    color_ref = cv2.remap(
        color_warped, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    gray_ref = cv2.remap(
        gray_warped, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return color_ref, gray_ref, pts


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
    image, gray, thresh_inv = _preprocess_for_detection(image)

    # Try contour detection first, fall back to Hough lines
    corners = _find_grid_contour(thresh_inv)
    if corners is None:
        corners = _find_grid_hough(thresh_inv)
        if corners is not None and not _is_reasonable_grid(corners, image.shape):
            corners = None
    if corners is None:
        raise ValueError("Could not detect a sudoku grid in the image.")

    corners = corners.astype(np.float32)
    color_warped = _warp_perspective(image, corners)
    gray_warped = _warp_perspective(gray, corners)

    return color_warped, gray_warped
