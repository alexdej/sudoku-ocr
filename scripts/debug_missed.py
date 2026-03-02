"""Debug specific cells that should have digits but are reported as empty."""
import cv2
import numpy as np
from pathlib import Path
from skimage.segmentation import clear_border
import sys
sys.path.insert(0, "src")

from sudoku_ocr.grid import detect_grid, detect_grid_size
from sudoku_ocr.cells import (
    _find_line_positions, MIN_DIGIT_AREA_RATIO, MIN_DIGIT_HEIGHT_RATIO,
    CELL_PADDING_RATIO, MIN_CELL_SIZE,
)

TARGETS = [
    ("samples/screenshots/Sudoku_app-in-progress.png", 1, 6),
    ("samples/screenshots/Sudoku_app-in-progress.png", 3, 8),
    ("samples/screenshots/sudoku-com-in-progress.png", 1, 1),
    ("samples/screenshots/sudoku-com-in-progress.png", 1, 6),
]


def get_inner_gray(img_path, target_row, target_col):
    bgr = cv2.imread(str(img_path))
    color_warped, gray_warped = detect_grid(bgr)
    grid_size = detect_grid_size(gray_warped)
    h, w = gray_warped.shape[:2]
    cell_size = min(w, h) / grid_size
    if cell_size < MIN_CELL_SIZE:
        scale = MIN_CELL_SIZE / cell_size
        gray_warped = cv2.resize(gray_warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        h, w = gray_warped.shape[:2]
    row_lines = _find_line_positions(gray_warped, grid_size, "horizontal")
    col_lines = _find_line_positions(gray_warped, grid_size, "vertical")
    r, c = target_row, target_col
    y1, y2 = row_lines[r], row_lines[r+1]
    x1, x2 = col_lines[c], col_lines[c+1]
    cell_h, cell_w = y2 - y1, x2 - x1
    pad_y = int(cell_h * CELL_PADDING_RATIO)
    pad_x = int(cell_w * CELL_PADDING_RATIO)
    return gray_warped[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]


def debug_cell(img_path, row, col):
    print(f"\n{'='*60}")
    print(f"{Path(img_path).name}  r{row}c{col}")
    print(f"{'='*60}")

    inner = get_inner_gray(img_path, row, col)
    h, w = inner.shape
    print(f"inner_gray shape: {inner.shape}, mean: {inner.mean():.1f}")

    thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    print(f"after Otsu (BINARY_INV): {cv2.countNonZero(thresh)} white px / {thresh.size} total")

    source = clear_border(thresh)
    cnts_cb, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"after clear_border: {cv2.countNonZero(source)} white px, {len(cnts_cb)} contours")

    if cnts_cb:
        path = "normal"
        cnts = cnts_cb
    else:
        path = "FALLBACK"
        source = thresh
        raw, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_area = h * w
        def _is_line(cnt):
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x > 0 and y > 0 and x+cw < w and y+ch < h:
                return False
            return max(cw, ch) / max(min(cw, ch), 1) > 5
        cnts = [c for c in raw
                if not _is_line(c)
                and cv2.contourArea(c) / cell_area <= 0.6
                and cv2.boundingRect(c)[3] / h >= 0.5]
        print(f"  fallback: {len(raw)} raw → {len(cnts)} after filters")

    print(f"path: {path}")
    if not cnts:
        print("RESULT: None (no contours after filters)")
        return

    cell_area = h * w
    for i, cnt in enumerate(sorted(cnts, key=cv2.contourArea, reverse=True)[:3]):
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        mask = np.zeros(source.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        masked = cv2.bitwise_and(source, source, mask=mask)
        density = cv2.countNonZero(masked) / (bw * bh) if bw * bh > 0 else 0
        touching = (x == 0 or y == 0 or x+bw >= w or y+bh >= h)
        print(f"  [{i}] area={area:.0f} ({area/cell_area:.3f}), "
              f"bbox={bw}×{bh}, h_ratio={bh/h:.3f}, density={density:.3f}, "
              f"border_touch={touching}")

    largest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, bw, bh = cv2.boundingRect(largest)
    print(f"\nLargest: area_ratio={area/cell_area:.3f} (min={MIN_DIGIT_AREA_RATIO})")
    if area / cell_area < MIN_DIGIT_AREA_RATIO:
        print("RESULT: None — area too small"); return
    print(f"  height_ratio={bh/h:.3f} (min={MIN_DIGIT_HEIGHT_RATIO})")
    if bh / h < MIN_DIGIT_HEIGHT_RATIO:
        print("RESULT: None — height too small"); return
    mask = np.zeros(source.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    result = cv2.bitwise_and(source, source, mask=mask)
    density = cv2.countNonZero(result) / (bw * bh) if bw * bh > 0 else 0
    print(f"  bbox_density={density:.3f} (min=0.20)")
    if density < 0.20:
        print("RESULT: None — bbox density too low"); return
    print("RESULT: digit region returned ✓")


for img_path, row, col in TARGETS:
    debug_cell(img_path, row, col)
