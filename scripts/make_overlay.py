"""Generate overlay images for all samples."""

from pathlib import Path

import cv2

from sudoku_ocr import PuzzleReader
from sudoku_ocr.viz import draw_overlay
from sudoku_ocr.model import SudokuNet


SAMPLES_DIR = Path("samples/screenshots")
OUTPUT_DIR = Path("output")
WEIGHTS_PATH = Path("src/sudoku_ocr/weights/digit_classifier.pt")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    model = SudokuNet(WEIGHTS_PATH)
    reader = PuzzleReader(model=model)

    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".jfif", ".gif", ".bmp", ".webp"):
            continue

        print(f"{path.name}...", end=" ")
        try:
            color_warped, cells = reader.read_cells(path)
            grid_size = max(c.row for c in cells) + 1
            digit_count = sum(1 for c in cells if c.digit is not None)
            overlay = draw_overlay(color_warped, cells)
            out_path = OUTPUT_DIR / f"{path.stem}_overlay.png"
            cv2.imwrite(str(out_path), overlay)
            print(f"{grid_size}x{grid_size}, {digit_count} digits → {out_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"FAIL: {e}")


if __name__ == "__main__":
    main()
