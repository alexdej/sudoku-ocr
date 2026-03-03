"""Diagnose remaining failing cells after _clear_border change."""
import sys
sys.path.insert(0, "/app/src")

import cv2
import numpy as np
from pathlib import Path
from sudoku_ocr.grid import detect_grid, detect_grid_size, refine_grid_warp
from sudoku_ocr.cells import _clear_border, CELL_PADDING_RATIO
from sudoku_ocr.grid import _find_line_positions

SAMPLES = Path("/app/samples")

def analyze_cell(img_path, subdir, target_row, target_col):
    bgr = cv2.imread(str(SAMPLES / subdir / img_path))
    color_warped, gray_warped = detect_grid(bgr)
    grid_size = detect_grid_size(gray_warped)
    color_warped, gray_warped, _ = refine_grid_warp(color_warped, gray_warped, grid_size)

    row_lines = _find_line_positions(gray_warped, grid_size, "horizontal")
    col_lines = _find_line_positions(gray_warped, grid_size, "vertical")

    row, col = target_row, target_col
    y1, y2 = row_lines[row], row_lines[row+1]
    x1, x2 = col_lines[col], col_lines[col+1]
    cell_h, cell_w = y2-y1, x2-x1

    pad_y = int(cell_h * CELL_PADDING_RATIO)
    pad_x = int(cell_w * CELL_PADDING_RATIO)
    inner_gray = gray_warped[y1+pad_y:y2-pad_y, x1+pad_x:x2-pad_x]
    h_img, w_img = inner_gray.shape

    otsu_val, thresh = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bimodality = abs(float(inner_gray.mean()) - float(otsu_val))
    cell_area = h_img * w_img

    def _clear_border_old(img):
        n_labels, labels = cv2.connectedComponents(img, connectivity=8)
        h, w = img.shape
        border_labels = set()
        border_labels.update(int(v) for v in labels[0,:])
        border_labels.update(int(v) for v in labels[h-1,:])
        border_labels.update(int(v) for v in labels[:,0])
        border_labels.update(int(v) for v in labels[:,w-1])
        border_labels.discard(0)
        cleared = img.copy()
        for label in border_labels:
            cleared[labels == label] = 0
        return cleared

    def describe_source(src, label):
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  {label}: {len(contours)} contours  (tc={cv2.countNonZero(thresh)/cell_area*100:.1f}%  src_px={cv2.countNonZero(src)})")
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(c)
            x,y,cw,ch = cv2.boundingRect(c)
            right = x + cw - 1
            bottom = y + ch - 1
            touches = []
            depths_list = []
            if y == 0: touches.append('T'); depths_list.append(bottom/h_img)
            if bottom == h_img-1: touches.append('B'); depths_list.append((h_img-1-y)/h_img)
            if x == 0: touches.append('L'); depths_list.append(right/w_img)
            if right == w_img-1: touches.append('R'); depths_list.append((w_img-1-x)/w_img)
            touch_str = ''.join(touches) if touches else 'int'
            depth_str = f"min_d={min(depths_list)*100:.0f}%" if depths_list else ""
            mask2 = np.zeros_like(src)
            cv2.drawContours(mask2, [c], -1, 255, -1)
            density = cv2.countNonZero(cv2.bitwise_and(src, src, mask=mask2)) / (cw*ch) if cw*ch > 0 else 0
            print(f"    area={a:.0f}({a/cell_area*100:.1f}%) bbox=({x},{y},{cw},{ch}) h={ch/h_img*100:.0f}% touch={touch_str} {depth_str} dens={density*100:.0f}%")

    print(f"\n{'='*60}")
    print(f"{img_path} r{row}c{col} (cell {h_img}x{w_img}  bimod={bimodality:.1f}):")

    source_old = _clear_border_old(thresh)
    source_new = _clear_border(thresh)

    describe_source(source_old, "OLD (old _clear_border)")
    describe_source(source_new, "NEW (new _clear_border, before large-erase)")

    n_src_labels, src_labels = cv2.connectedComponents(source_new, connectivity=8)
    source_erased = source_new.copy()
    erased_any = False
    for lab in range(1, n_src_labels):
        if np.sum(src_labels == lab) / cell_area > 0.60:
            source_erased[src_labels == lab] = 0
            erased_any = True
    if erased_any:
        describe_source(source_erased, "NEW (after large-blob erase)")

analyze_cell("sudoku-coach-highlighted.png", "screenshots", 3, 6)
analyze_cell("sudoku-coach-wrong.png", "screenshots", 3, 3)
analyze_cell("hq720.jpg", "screenshots", 0, 4)
