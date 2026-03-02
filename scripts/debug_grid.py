"""Debug grid detection on problem images."""

from pathlib import Path

import cv2
import numpy as np

from sudoku_ocr.grid import _find_grid_contour, _find_grid_hough, _order_corners


SAMPLES_DIR = Path("samples/screenshots")
TARGETS = ["images.png", "hq720.jpg"]


def debug_image(path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"Debugging: {path.name}")
    print(f"{'=' * 70}")

    img = cv2.imread(str(path))
    if img is None:
        print("  Could not load")
        return

    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    # Examine top contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    print(f"  Total external contours: {len(contours)}")
    print(f"  Image area: {w * h}")
    print(f"\n  Top 10 contours:")
    for i, c in enumerate(contours[:10]):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        print(f"    #{i}: area={area:.0f} ({area/(w*h)*100:.1f}% of image), "
              f"vertices={len(approx)}, bbox=({x},{y},{cw},{ch}), aspect={aspect:.2f}")

    # Try contour method
    corners = _find_grid_contour(thresh)
    if corners is not None:
        ordered = _order_corners(corners.astype(np.float32))
        tl, tr, br, bl = ordered
        grid_w = np.linalg.norm(tr - tl)
        grid_h = np.linalg.norm(bl - tl)
        print(f"\n  Contour method found grid:")
        print(f"    Corners: TL={tl}, TR={tr}, BR={br}, BL={bl}")
        print(f"    Grid size: {grid_w:.0f}x{grid_h:.0f}")
        print(f"    Grid area ratio: {(grid_w*grid_h)/(w*h)*100:.1f}% of image")
    else:
        print(f"\n  Contour method: no 4-point contour found")

    # Try Hough method
    corners_h = _find_grid_hough(thresh)
    if corners_h is not None:
        ordered_h = _order_corners(corners_h.astype(np.float32))
        tl, tr, br, bl = ordered_h
        grid_w = np.linalg.norm(tr - tl)
        grid_h = np.linalg.norm(bl - tl)
        print(f"\n  Hough method found grid:")
        print(f"    Corners: TL={tl}, TR={tr}, BR={br}, BL={bl}")
        print(f"    Grid size: {grid_w:.0f}x{grid_h:.0f}")
    else:
        print(f"\n  Hough method: failed")

    # For hq720.jpg, also check what happens with RETR_TREE to find nested contours
    contours_tree, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find contours that are roughly square and take up a significant portion of the image
    print(f"\n  Square-ish contours (aspect 0.8-1.2, >10% of image area):")
    for i, c in enumerate(contours_tree):
        area = cv2.contourArea(c)
        if area / (w * h) < 0.10:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        if 0.8 <= aspect <= 1.2:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            print(f"    contour {i}: area={area:.0f} ({area/(w*h)*100:.1f}%), "
                  f"vertices={len(approx)}, bbox=({x},{y},{cw},{ch}), aspect={aspect:.2f}")


def main() -> None:
    for name in TARGETS:
        path = SAMPLES_DIR / name
        if path.exists():
            debug_image(path)


if __name__ == "__main__":
    main()
