"""Regression tests: verify given/fill classification on each sample image.

Expected strings are of length grid_size² with one character per cell:
  'G' = given (printed digit, dark/black)
  'F' = fill  (user-entered digit, colored)
  '.' = empty cell

Only images where given/fill is distinguishable by color are included.
Images where all digits are the same dark color (e.g. NYT app) are omitted.

To regenerate expected values after intentional changes:
    python scripts/gen_expected.py  (then update EXPECTED manually)
"""

import pytest
from pathlib import Path

import sudoku_ocr as _pkg
from sudoku_ocr import PuzzleReader

SAMPLES_DIR  = Path(__file__).parent / "testdata" / "images"
WEIGHTS_DIR  = Path(_pkg.__file__).parent / "weights"
del _pkg


def _given_fill_string(reader: PuzzleReader, path: Path) -> str:
    """Return a G/F/./? string parallel to read_digits_string."""
    _, cells = reader.read_cells(path)
    return "".join(
        "." if not c.has_digit
        else ("G" if c.is_given is True else ("F" if c.is_given is False else "?"))
        for c in cells
    )


# 'G' = given, 'F' = fill (correct or wrong), '.' = empty.
# These images all use color to distinguish given vs fill digits.
EXPECTED: dict[str, str] = {
    # sudoku.coach — givens are black, fill digits are blue, wrong fill have dark-red bg
    "sudoku-coach-only-givens.png":   "...G....G.G........G.G.GG...GG..G....GGG.........G.GG...G.G...G...G.G.G.G.....G.G",
    "sudoku-coach-in-progress.png":   "...G....G.G..F....FG.G.GG...GG..GF...GGG....F....G.GG...G.G..FG.F.G.G.G.G...F.G.G",
    "sudoku-coach-highlighted.png":   "...G....G.G..F....FG.G.GG...GG..GF...GGG....F....G.GG...G.G..FG.F.G.G.G.G...F.G.G",
    "sudoku-coach-completed.png":     "GGGFFFFGFGFFGFFFFGGGGFFFFFGFGGFGGFGFFFGFGGGFGGGGFGGGGFFGGGFGGGFGGFFFGFGGGFGFGFFGF",
    # sudoku.com — givens are dark navy, fill digits are bright blue, wrong fill is red
    "sudoku-com-in-progress.png":     ".G....GGGG..F.G..F...GG.GG..GGGGG.GG.GG.GGGGG.GG...G..G..GG...G.G.F..G..GG..GGF..",
    "sudoku-com-wrong.png":           ".G....GGGG..F.G..F...GG.GG..GGGGG.GG.GG.GGGGG.GG...G..G..GG...G.G.FF.G..GG..GGF..",
    # unknown Android app — givens are black, fill digits are blue
    "Sudoku_app-in-progress.png":     "F...GG..GGFFGF.G.GGFF..GFG.FGGGFGFFGF.FFGF...G..GFGGGFFG.G....GGFG.FG.FGG..GG.F..",
}


@pytest.fixture(scope="module")
def reader() -> PuzzleReader:
    return PuzzleReader.from_weights_dir(WEIGHTS_DIR)


@pytest.mark.parametrize("filename,expected", EXPECTED.items())
def test_given_fill(reader: PuzzleReader, filename: str, expected: str) -> None:
    path = SAMPLES_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in tests/testdata/images/")
    result = _given_fill_string(reader, path)
    assert result == expected, (
        f"\n  file    : {filename}"
        f"\n  got     : {result}"
        f"\n  expected: {expected}"
    )
