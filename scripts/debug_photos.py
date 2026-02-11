"""Debug grid detection and size detection for problem photo samples."""

from pathlib import Path

import cv2
import numpy as np

from sudoku_ocr.grid import (
    detect_grid,
    detect_grid_size,
    _find_grid_contour,
    _find_grid_hough,
    _is_reasonable_grid,
    _extract_lines,
    _count_lines,
)

SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("output")

PROBLEM_PHOTOS = [
    "6978422072_33ac92fe1a_b.jpg",
    "EFJ5JjIXYAEvSse.jpg",
    "images.jfif",
]


def debug_grid_detection(path: Path) -> None:
    """Debug grid detection for a single image."""
    print(f"\n{'='*60}")
    print(f"DEBUG: {path.name}")
    print(f"{'='*60}")

    img = cv2.imread(str(path))
    if img is None:
        print("  Could not load image!")
        return

    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    # Test contour detection
    corners = _find_grid_contour(thresh)
    if corners is not None:
        print(f"  Contour detection: FOUND corners")
        ordered = corners.astype(np.float32)
        widths = [np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])]
        heights = [np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])]
        print(f"    Widths: {widths}")
        print(f"    Heights: {heights}")
    else:
        print(f"  Contour detection: FAILED")
        # Debug: show what contours were found
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print(f"    Total contours: {len(contours)}")
        for i, c in enumerate(contours[:5]):
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            img_area = w * h
            print(f"    #{i}: area={area:.0f} ({area/img_area*100:.1f}% of image), "
                  f"vertices={len(approx)}, bbox={cw}x{ch}, aspect={aspect:.2f}")
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                reasonable = _is_reasonable_grid(pts, thresh.shape)
                print(f"      4-point! reasonable={reasonable}")

        # Also try with RETR_TREE to find nested contours
        contours_tree, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree = sorted(contours_tree, key=cv2.contourArea, reverse=True)
        print(f"\n    RETR_TREE contours: {len(contours_tree)}")
        for i, c in enumerate(contours_tree[:10]):
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            img_area = w * h
            if area / img_area > 0.03 and len(approx) <= 6:
                print(f"    #{i}: area={area:.0f} ({area/img_area*100:.1f}%), "
                      f"vertices={len(approx)}, bbox={cw}x{ch}, aspect={aspect:.2f}")
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    reasonable = _is_reasonable_grid(pts, thresh.shape)
                    print(f"      4-point! reasonable={reasonable}")

    # Hough fallback
    hough_corners = _find_grid_hough(thresh)
    if hough_corners is not None:
        print(f"  Hough detection: FOUND")
        reasonable = _is_reasonable_grid(hough_corners, img.shape)
        print(f"    Reasonable: {reasonable}")
    else:
        print(f"  Hough detection: FAILED")

    # Try full detect_grid
    try:
        color_warped, gray_warped = detect_grid(img)
        wh, ww = gray_warped.shape[:2]
        print(f"\n  Grid detected! Warped size: {ww}x{wh}")
        aspect = max(ww, wh) / min(ww, wh)
        print(f"    Aspect ratio: {aspect:.3f}")

        # Debug grid size detection
        debug_grid_size(gray_warped, path.stem)

    except ValueError as e:
        print(f"\n  Grid detection FAILED: {e}")

        # Try with different preprocessing
        print("\n  Trying alternative preprocessing...")
        # Try with more blur
        blurred2 = cv2.GaussianBlur(gray, (11, 11), 5)
        thresh2 = cv2.adaptiveThreshold(
            blurred2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2,
        )
        thresh2 = cv2.bitwise_not(thresh2)
        corners2 = _find_grid_contour(thresh2)
        if corners2 is not None:
            print("    More blur: contour FOUND!")
        else:
            print("    More blur: contour still failed")

        # Try with Otsu threshold
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        corners3 = _find_grid_contour(thresh3)
        if corners3 is not None:
            print("    Otsu: contour FOUND!")
        else:
            print("    Otsu: contour still failed")

        # Try dilating to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        corners4 = _find_grid_contour(dilated)
        if corners4 is not None:
            print("    Dilated: contour FOUND!")
            pts = corners4.astype(np.float32)
            reasonable = _is_reasonable_grid(pts, dilated.shape)
            print(f"    Reasonable: {reasonable}")
        else:
            print("    Dilated: contour still failed")


def debug_grid_size(gray_warped: np.ndarray, name: str) -> None:
    """Debug grid size detection."""
    h, w = gray_warped.shape[:2]
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    thresh = cv2.bitwise_not(thresh)

    h_lines_img = _extract_lines(thresh, "horizontal")
    v_lines_img = _extract_lines(thresh, "vertical")

    h_profile = np.sum(h_lines_img, axis=1).astype(np.float64)
    v_profile = np.sum(v_lines_img, axis=0).astype(np.float64)

    min_dist_h = h // 25
    min_dist_v = w // 25

    h_count = _count_lines(h_profile, min_dist_h)
    v_count = _count_lines(v_profile, min_dist_v)

    avg_lines = (h_count + v_count) / 2.0
    avg_cells = avg_lines - 1

    print(f"\n  Grid size detection:")
    print(f"    Warped size: {w}x{h}")
    print(f"    H lines: {h_count}, V lines: {v_count}")
    print(f"    Avg cells: {avg_cells:.1f}")

    # Save line extraction images for visual debug
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_hlines.png"), h_lines_img)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_vlines.png"), v_lines_img)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_warped.png"), gray_warped)
    print(f"    Saved debug images to output/")

    # Try with different kernel sizes
    for divisor in [8, 12, 16, 20]:
        h_len = max(w // divisor, 10)
        v_len = max(h // divisor, 10)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        hp = np.sum(h_img, axis=1).astype(np.float64)
        vp = np.sum(v_img, axis=0).astype(np.float64)

        hc = _count_lines(hp, min_dist_h)
        vc = _count_lines(vp, min_dist_v)
        print(f"    Divisor {divisor} (kernel h={h_len}, v={v_len}): H={hc}, V={vc}")

    try:
        detected = detect_grid_size(gray_warped)
        print(f"    Detected grid size: {detected}")
    except ValueError as e:
        print(f"    Grid size detection FAILED: {e}")


def main() -> None:
    for name in PROBLEM_PHOTOS:
        path = SAMPLES_DIR / name
        if path.exists():
            debug_grid_detection(path)
        else:
            print(f"\nSKIPPED: {name} (not found)")


if __name__ == "__main__":
    main()
