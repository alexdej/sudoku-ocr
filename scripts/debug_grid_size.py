"""Debug grid size detection on all samples."""

from pathlib import Path

import cv2
import numpy as np

from sudoku_ocr.grid import detect_grid, _extract_lines, _count_lines


SAMPLES_DIR = Path("samples/screenshots")


def debug_image(path: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        return

    try:
        _, gray_warped = detect_grid(img)
    except ValueError:
        print(f"{path.name:50s}  grid detection failed")
        return

    h, w = gray_warped.shape
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    h_lines_img = _extract_lines(thresh, "horizontal")
    v_lines_img = _extract_lines(thresh, "vertical")

    h_profile = np.sum(h_lines_img, axis=1).astype(np.float64)
    v_profile = np.sum(v_lines_img, axis=0).astype(np.float64)

    h_white = np.sum(h_lines_img > 0)
    v_white = np.sum(v_lines_img > 0)

    min_dist = max(min(h, w) // 25, 3)
    h_count = _count_lines(h_profile, min_dist)
    v_count = _count_lines(v_profile, min_dist)

    kernel_len = max(w // 30, 8)

    print(f"{path.name:50s}  warped={w}x{h}  kernel={kernel_len}  "
          f"h_lines={h_count}  v_lines={v_count}  "
          f"h_px={h_white}  v_px={v_white}")


def main() -> None:
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            debug_image(path)


if __name__ == "__main__":
    main()
