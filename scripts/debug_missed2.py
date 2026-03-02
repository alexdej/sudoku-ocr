"""Investigate sudoku-com: why are the blue-1 cells showing mean=255 in grayscale?"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "src")

from sudoku_ocr.grid import detect_grid, detect_grid_size
from sudoku_ocr.cells import _find_line_positions, CELL_PADDING_RATIO, MIN_CELL_SIZE

IMG = Path("samples/screenshots/sudoku-com-in-progress.png")
TARGETS = [(1, 1), (1, 6)]

bgr = cv2.imread(str(IMG))
color_warped, gray_warped = detect_grid(bgr)
grid_size = detect_grid_size(gray_warped)
print(f"image size: {bgr.shape[1]}x{bgr.shape[0]}")
print(f"warped size: {gray_warped.shape[1]}x{gray_warped.shape[0]}")
print(f"grid_size: {grid_size}")

h, w = gray_warped.shape[:2]
cell_size = min(w, h) / grid_size
if cell_size < MIN_CELL_SIZE:
    scale = MIN_CELL_SIZE / cell_size
    gray_warped = cv2.resize(gray_warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    color_warped = cv2.resize(color_warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    h, w = gray_warped.shape[:2]
    print(f"upscaled to: {w}x{h}")

row_lines = _find_line_positions(gray_warped, grid_size, "horizontal")
col_lines = _find_line_positions(gray_warped, grid_size, "vertical")

print(f"\nrow_lines: {row_lines}")
print(f"col_lines: {col_lines}")

for r, c in TARGETS:
    print(f"\n--- Cell ({r},{c}) ---")
    y1, y2 = row_lines[r], row_lines[r+1]
    x1, x2 = col_lines[c], col_lines[c+1]
    print(f"cell bounds: y={y1}:{y2}, x={x1}:{x2}  ({y2-y1}x{x2-x1})")

    color_cell = color_warped[y1:y2, x1:x2]
    gray_cell = gray_warped[y1:y2, x1:x2]
    print(f"color cell mean BGR: {color_cell.mean(axis=(0,1)).round(1)}")
    print(f"gray cell mean: {gray_cell.mean():.1f}, min: {gray_cell.min()}, max: {gray_cell.max()}")

    pad_y = int((y2-y1) * CELL_PADDING_RATIO)
    pad_x = int((x2-x1) * CELL_PADDING_RATIO)
    inner_color = color_warped[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]
    inner_gray = gray_warped[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]
    print(f"inner cell ({inner_gray.shape}): mean={inner_gray.mean():.1f}, "
          f"min={inner_gray.min()}, max={inner_gray.max()}")
    print(f"inner color mean BGR: {inner_color.mean(axis=(0,1)).round(1)}")

    # How many dark pixels are in the inner cell?
    for thresh_val in [200, 180, 150, 100, 50]:
        dark_px = np.sum(inner_gray < thresh_val)
        print(f"  pixels < {thresh_val}: {dark_px} ({100*dark_px/inner_gray.size:.1f}%)")
