"""
1. Sudoku_app r1c1: confirm area ratio issue
2. sudoku-com: scan all cells where outer has dark pixels but inner is blank
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "src")

from sudoku_ocr.grid import detect_grid, detect_grid_size
from sudoku_ocr.cells import _find_line_positions, CELL_PADDING_RATIO, MIN_CELL_SIZE


def load_warped(img_path):
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
    return gray_warped, grid_size, row_lines, col_lines


# ── 1. Sudoku_app r1c1 area check ──────────────────────────────────────────
print("=== Sudoku_app r1c1 ===")
from skimage.segmentation import clear_border
gray, gs, rl, cl = load_warped(Path("samples/screenshots/Sudoku_app-in-progress.png"))
r, c = 1, 1
y1, y2, x1, x2 = rl[r], rl[r+1], cl[c], cl[c+1]
cell_h, cell_w = y2-y1, x2-x1
pad_y, pad_x = int(cell_h * CELL_PADDING_RATIO), int(cell_w * CELL_PADDING_RATIO)
inner = gray[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]
print(f"inner shape={inner.shape}, mean={inner.mean():.1f}")
thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
source = clear_border(thresh)
cnts, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cell_area = inner.shape[0] * inner.shape[1]
if cnts:
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        print(f"  contour: area={area:.0f} ({area/cell_area:.3f}), bbox={bw}x{bh}, h_ratio={bh/inner.shape[0]:.3f}")
else:
    print("  no contours after clear_border")


# ── 2. sudoku-com: scan for cells where digit is in padding zone ────────────
print("\n=== sudoku-com: cells with content in padding zone but not inner cell ===")
gray, gs, rl, cl = load_warped(Path("samples/screenshots/sudoku-com-in-progress.png"))

for r in range(gs):
    for c in range(gs):
        y1, y2 = rl[r], rl[r+1]
        x1, x2 = cl[c], cl[c+1]
        cell_h, cell_w = y2-y1, x2-x1
        pad_y = int(cell_h * CELL_PADDING_RATIO)
        pad_x = int(cell_w * CELL_PADDING_RATIO)

        outer = gray[y1:y2, x1:x2]
        inner = gray[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]

        outer_dark = int(np.sum(outer < 150))
        inner_dark = int(np.sum(inner < 150))

        if outer_dark > 20 and inner_dark == 0:
            print(f"  r{r}c{c}: outer has {outer_dark} dark pixels (min={outer.min()}), "
                  f"inner has 0 dark pixels  — digit in padding zone!")
        elif outer_dark > 100 and inner_dark < outer_dark * 0.2:
            print(f"  r{r}c{c}: outer={outer_dark} dark px, inner={inner_dark} dark px "
                  f"({100*inner_dark/max(outer_dark,1):.0f}% remain after padding)")
