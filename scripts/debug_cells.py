"""Debug script to understand what's being detected in each cell."""

from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import clear_border

from sudoku_ocr.grid import detect_grid


SAMPLES_DIR = Path("samples/screenshots")
# Focus on the over-detecting images
TARGETS = ["rules0-1.png", "sudoku1-max-en.gif", "sudokuexample.png"]


def analyze_cell(cell_gray: np.ndarray, pad_ratio: float = 0.08) -> dict:
    """Analyze a single cell and return diagnostic info."""
    h, w = cell_gray.shape
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)
    inner = cell_gray[pad_y:h - pad_y, pad_x:w - pad_x]

    thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cleared = clear_border(thresh)

    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"has_digit": False, "reason": "no_contours"}

    largest = max(contours, key=cv2.contourArea)
    cell_area = inner.shape[0] * inner.shape[1]
    area_ratio = cv2.contourArea(largest) / cell_area

    x, y, cw, ch = cv2.boundingRect(largest)
    aspect = ch / max(cw, 1)
    center_x = (x + cw / 2) / inner.shape[1]
    center_y = (y + ch / 2) / inner.shape[0]

    # How much of the cell height does the contour span?
    height_ratio = ch / inner.shape[0]
    width_ratio = cw / inner.shape[1]

    return {
        "has_digit": area_ratio >= 0.03,
        "area_ratio": round(area_ratio, 4),
        "aspect": round(aspect, 2),
        "center": (round(center_x, 2), round(center_y, 2)),
        "height_ratio": round(height_ratio, 2),
        "width_ratio": round(width_ratio, 2),
        "num_contours": len(contours),
    }


def debug_image(path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"Debugging: {path.name}")
    print(f"{'=' * 70}")

    img = cv2.imread(str(path))
    if img is None:
        print("  Could not load")
        return

    try:
        color_warped, gray_warped = detect_grid(img)
    except ValueError as e:
        print(f"  Grid detection failed: {e}")
        return

    h, w = gray_warped.shape[:2]
    step_y = h // 9
    step_x = w // 9

    # Collect stats on all detected cells
    detected = []
    for row in range(9):
        for col in range(9):
            y1, y2 = row * step_y, (row + 1) * step_y
            x1, x2 = col * step_x, (col + 1) * step_x
            cell = gray_warped[y1:y2, x1:x2]
            info = analyze_cell(cell)
            if info["has_digit"]:
                detected.append((row, col, info))

    print(f"  Detected {len(detected)} cells with digits")
    print(f"\n  {'Cell':>6} {'Area%':>7} {'H%':>5} {'W%':>5} {'Asp':>5} {'Center':>12} {'#Cnt':>5}")
    print(f"  {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*12} {'-'*5}")
    for row, col, info in detected:
        print(
            f"  ({row},{col})  "
            f"{info['area_ratio']:>6.3f} "
            f"{info['height_ratio']:>5.2f} "
            f"{info['width_ratio']:>5.2f} "
            f"{info['aspect']:>5.2f} "
            f"{str(info['center']):>12} "
            f"{info['num_contours']:>5}"
        )

    # Show distribution of area ratios
    ratios = [info["area_ratio"] for _, _, info in detected]
    if ratios:
        print(f"\n  Area ratio stats: min={min(ratios):.4f} max={max(ratios):.4f} "
              f"median={sorted(ratios)[len(ratios)//2]:.4f}")
        # Histogram of area ratios
        bins = [0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0]
        print(f"  Distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for r in ratios if bins[i] <= r < bins[i+1])
            if count:
                print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {'#' * count} ({count})")


def main() -> None:
    for name in TARGETS:
        path = SAMPLES_DIR / name
        if path.exists():
            debug_image(path)


if __name__ == "__main__":
    main()
