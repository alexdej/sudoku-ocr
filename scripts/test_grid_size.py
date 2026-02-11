"""Test automatic grid size detection on sample images."""

from pathlib import Path

import cv2

from sudoku_ocr.grid import detect_grid, detect_grid_size


SAMPLES_DIR = Path("samples")


def main() -> None:
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        print(f"{path.name:60s}", end="  ")
        try:
            _, gray_warped = detect_grid(img)
            size = detect_grid_size(gray_warped)
            print(f"detected: {size}x{size}")
        except ValueError as e:
            print(f"FAIL: {e}")


if __name__ == "__main__":
    main()
