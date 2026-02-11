"""Debug hq720.jpg bottom-right cell extraction."""

from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import clear_border

from sudoku_ocr.grid import detect_grid


def main() -> None:
    img = cv2.imread("samples/hq720.jpg")
    color_warped, gray_warped = detect_grid(img)

    h, w = gray_warped.shape[:2]
    step_y = h // 9
    step_x = w // 9
    pad_ratio = 0.08

    print(f"Warped size: {w}x{h}, cell size: {step_x}x{step_y}")
    print(f"\nBottom-right region (rows 6-8, cols 6-8):")

    for row in range(6, 9):
        for col in range(6, 9):
            y1, y2 = row * step_y, (row + 1) * step_y
            x1, x2 = col * step_x, (col + 1) * step_x
            cell = gray_warped[y1:y2, x1:x2]
            color_cell = color_warped[y1:y2, x1:x2]

            pad_y = int(cell.shape[0] * pad_ratio)
            pad_x = int(cell.shape[1] * pad_ratio)
            inner = cell[pad_y:cell.shape[0] - pad_y, pad_x:cell.shape[1] - pad_x]

            # Stats on the grayscale cell
            mean_val = np.mean(inner)
            std_val = np.std(inner)

            # Try thresholding
            thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            white_ratio = np.sum(thresh > 0) / thresh.size

            cleared = clear_border(thresh)
            cleared_ratio = np.sum(cleared > 0) / cleared.size

            contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = max((cv2.contourArea(c) for c in contours), default=0)
            area_ratio = max_area / cleared.size if cleared.size > 0 else 0

            # Also check the color cell for highlight detection
            mean_color = np.mean(color_cell, axis=(0, 1))  # BGR

            print(f"\n  Cell ({row},{col}):")
            print(f"    Gray: mean={mean_val:.1f}, std={std_val:.1f}")
            print(f"    Color BGR mean: ({mean_color[0]:.0f}, {mean_color[1]:.0f}, {mean_color[2]:.0f})")
            print(f"    After thresh: {white_ratio*100:.1f}% white")
            print(f"    After clear_border: {cleared_ratio*100:.1f}% white")
            print(f"    Contours: {len(contours)}, max area ratio: {area_ratio:.4f}")


if __name__ == "__main__":
    main()
