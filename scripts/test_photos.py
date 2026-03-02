"""Quick check: run pipeline on all photos and report results."""
import math
from pathlib import Path
from sudoku_ocr import PuzzleReader
from sudoku_ocr.model import SudokuNet

reader = PuzzleReader(model=SudokuNet(Path("src/sudoku_ocr/weights/digit_classifier.pt")))


def grid_str(result: str, size: int = 9) -> str:
    box = int(math.isqrt(size)) or 1
    rows = []
    for r in range(size):
        row = result[r * size : (r + 1) * size]
        parts = [row[c * box : (c + 1) * box] for c in range(box)]
        rows.append(" | ".join(parts))
        if (r + 1) % box == 0 and r < size - 1:
            rows.append("-" * len(rows[-1]))
    return "\n".join(rows)


for p in sorted(Path("samples/photos").iterdir()):
    print(f"\n{'='*60}")
    print(f"{p.name}")
    print(f"{'='*60}")
    try:
        _, cells = reader.read_cells(p)
        size = max(c.row for c in cells) + 1
        result = "".join(str(c.digit) if c.digit else "." for c in cells)
        digits = sum(1 for c in result if c != ".")
        print(f"grid={size}x{size}  len={len(result)}  digits={digits}")
        print()
        print(grid_str(result, size))
    except Exception as e:
        print(f"FAIL: {e}")
