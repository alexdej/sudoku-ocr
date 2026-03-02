"""Print expected digit strings and given/fill strings for all screenshots.

Used once to generate values to hardcode in tests. Run with:
    python scripts/gen_expected.py
"""
from pathlib import Path
from sudoku_ocr import PuzzleReader
from sudoku_ocr.model import SudokuNet

WEIGHTS = Path("src/sudoku_ocr/weights/digit_classifier.pt")
SAMPLES = Path("samples/screenshots")

reader = PuzzleReader(model=SudokuNet(WEIGHTS))

all_files = sorted(p.name for p in SAMPLES.iterdir() if p.is_file())

print("# --- digit strings (for test_samples.py EXPECTED) ---")
for name in all_files:
    path = SAMPLES / name
    try:
        result = reader.read_digits_string(path)
        print(f'    "{name}": {" " * max(0, 50 - len(name))}"{result}",')
    except Exception as e:
        print(f'    # {name}: ERROR {e}')

print()
print("# --- given/fill strings (for test_given_fill.py EXPECTED) ---")
for name in all_files:
    path = SAMPLES / name
    try:
        _, cells = reader.read_cells(path)
        gf = "".join(
            "." if not c.has_digit
            else ("G" if c.is_given is True else ("F" if c.is_given is False else "?"))
            for c in cells
        )
        print(f'    "{name}": {" " * max(0, 50 - len(name))}"{gf}",')
    except Exception as e:
        print(f'    # {name}: ERROR {e}')
