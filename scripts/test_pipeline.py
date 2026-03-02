"""End-to-end test of the sudoku OCR pipeline."""

import math
from pathlib import Path

from sudoku_ocr import PuzzleReader
from sudoku_ocr.model import SudokuNet


SAMPLES_DIR = Path("samples/screenshots")
WEIGHTS_PATH = Path("src/sudoku_ocr/weights/digit_classifier.pt")


def format_grid(digits: list[int | None], size: int = 9) -> str:
    """Format a flat digit list as a readable grid."""
    box = int(math.isqrt(size)) or 1
    lines = []
    for row in range(size):
        cells = []
        for col in range(size):
            val = digits[row * size + col]
            cells.append(str(val) if val is not None else ".")
            if (col + 1) % box == 0 and col < size - 1:
                cells.append("|")
        lines.append(" ".join(cells))
        if (row + 1) % box == 0 and row < size - 1:
            lines.append("-" * len(lines[-1]))
    return "\n".join(lines)


def test_image(reader: PuzzleReader, path: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Testing: {path.name}")
    print(f"{'=' * 60}")

    try:
        _, cells = reader.read_cells(path)
    except (ValueError, FileNotFoundError) as e:
        print(f"  FAIL: {e}")
        return

    grid_size = max(c.row for c in cells) + 1 if cells else 9
    digits = [c.digit for c in cells]
    digit_count = sum(1 for d in digits if d is not None)
    empty_count = sum(1 for d in digits if d is None)
    print(f"  Grid: {grid_size}x{grid_size}, Digits found: {digit_count}, Empty: {empty_count}")
    print()
    for line in format_grid(digits, grid_size).splitlines():
        print(f"    {line}")


def main() -> None:
    if not SAMPLES_DIR.exists():
        print(f"No samples directory found at {SAMPLES_DIR}")
        return

    model = SudokuNet(WEIGHTS_PATH)
    reader = PuzzleReader(model=model)

    images = sorted(SAMPLES_DIR.iterdir())
    print(f"Found {len(images)} sample images")

    for path in images:
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".jfif", ".gif", ".bmp", ".webp"):
            test_image(reader, path)

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
