"""Debug phantom digit detections in printed-digit photos.

For each identified phantom cell, show the contour properties that caused
it to pass all our filters.
"""
import cv2
import numpy as np
from pathlib import Path
from skimage.segmentation import clear_border
import sys
sys.path.insert(0, "src")

from sudoku_ocr.grid import detect_grid, detect_grid_size
from sudoku_ocr.cells import (
    _find_line_positions, _extract_digit_region,
    CELL_PADDING_RATIO, MIN_CELL_SIZE,
    MIN_DIGIT_AREA_RATIO, MIN_DIGIT_HEIGHT_RATIO,
)

# (image, row, col) — all 0-indexed
PHANTOMS = [
    ("samples/photos/6978422072_33ac92fe1a_b.jpg", 4, 5),   # user: r5c6 → 1-indexed
    ("samples/photos/images.jfif",                 0, 0),   # first digit, phantom 6
    ("samples/photos/IMG_6062.JPG",                0, 6),   # phantom 1 in r1 (1-idx)
    ("samples/photos/IMG_6062.JPG",                5, 4),   # first phantom 9 in r6 (1-idx)
    ("samples/photos/IMG_6062.JPG",                5, 5),   # second phantom 9 in r6
]


def load_warped(img_path):
    bgr = cv2.imread(str(img_path))
    color_warped, gray_warped = detect_grid(bgr)
    grid_size = detect_grid_size(gray_warped)
    h, w = gray_warped.shape[:2]
    cell_size = min(w, h) / grid_size
    if cell_size < MIN_CELL_SIZE:
        scale = MIN_CELL_SIZE / cell_size
        gray_warped = cv2.resize(gray_warped, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)
        h, w = gray_warped.shape[:2]
    rl = _find_line_positions(gray_warped, grid_size, "horizontal")
    cl = _find_line_positions(gray_warped, grid_size, "vertical")
    return gray_warped, rl, cl


for img_path, row, col in PHANTOMS:
    print(f"\n{'='*60}")
    print(f"{Path(img_path).name}  r{row}c{col}  (0-indexed)")
    print(f"{'='*60}")

    gray, rl, cl = load_warped(Path(img_path))
    y1, y2 = rl[row], rl[row + 1]
    x1, x2 = cl[col], cl[col + 1]
    cell_h, cell_w = y2 - y1, x2 - x1
    pad_y = int(cell_h * CELL_PADDING_RATIO)
    pad_x = int(cell_w * CELL_PADDING_RATIO)
    inner = gray[y1 + pad_y: y2 - pad_y, x1 + pad_x: x2 - pad_x]

    print(f"cell={cell_h}x{cell_w}  inner={inner.shape}  mean={inner.mean():.1f}")

    thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    otsu_val = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0]
    print(f"Otsu threshold: {otsu_val:.0f}  white_px={cv2.countNonZero(thresh)}")

    source = clear_border(thresh)
    cnts, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"after clear_border: {cv2.countNonZero(source)} white px, {len(cnts)} contours")

    if cnts:
        path = "normal"
    else:
        path = "fallback"
        source = thresh
        h_img, w_img = thresh.shape
        raw, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_area_fb = h_img * w_img
        def _is_line(cnt):
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x > 0 and y > 0 and x + cw < w_img and y + ch < h_img:
                return False
            return max(cw, ch) / max(min(cw, ch), 1) > 5
        cnts = [c for c in raw
                if not _is_line(c)
                and cv2.contourArea(c) / cell_area_fb <= 0.6
                and cv2.boundingRect(c)[3] / h_img >= 0.5]
        print(f"  fallback: {len(raw)} raw → {len(cnts)} kept")

    print(f"path: {path}")
    cell_area = inner.shape[0] * inner.shape[1]
    for i, cnt in enumerate(sorted(cnts, key=cv2.contourArea, reverse=True)[:3]):
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        mask = np.zeros(source.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        density = cv2.countNonZero(cv2.bitwise_and(source, source, mask=mask)) / (bw * bh)
        touching = (x == 0 or y == 0 or x + bw >= source.shape[1] or y + bh >= source.shape[0])
        print(f"  [{i}] area={area:.0f} ({area/cell_area:.3f}), "
              f"bbox={bw}x{bh}, h_ratio={bh/inner.shape[0]:.3f}, "
              f"density={density:.3f}, border={touching}")

    result = _extract_digit_region(inner)
    print(f"_extract_digit_region: {'digit ✓' if result is not None else 'None'}")
