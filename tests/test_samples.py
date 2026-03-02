"""Smoke tests: OCR each sample image and check output hasn't changed.

Expected strings are digit strings of length grid_size², with '.' for empty cells.
Run `pytest tests/test_samples.py -v` to execute.

To regenerate expected values after intentional changes:
    python scripts/gen_expected.py  (then update SCREENSHOTS/PHOTOS manually)
"""

import pytest
from pathlib import Path

from sudoku_ocr import PuzzleReader
from sudoku_ocr.model import SudokuNet


SAMPLES_DIR  = Path(__file__).parent.parent / "samples"
WEIGHTS_PATH = Path(__file__).parent.parent / "src" / "sudoku_ocr" / "weights" / "digit_classifier.pt"


# NOTE: verify these expected values before treating failures as regressions.
# '.' = empty cell, digits = OCR prediction.
SCREENSHOTS: dict[str, str] = {
    "16x16.png":                                 "..2A..D3...6F.....C.74EF..........6....5.2...1C41....A6.93C....72....C..3.6...9....CF23...4DC.6...9B..C42F7.A.51...DB.1..C...F82CCE...5..9.21...BD.F.1CE48..59...1.249...5DCE....6...F.8..3....BD....572.19....FF9C...4.5....C..........D6C3.2.....1C...FA..34..", # now recognizing hex A-F
    "4363.png":                                  ".7..5..6.4..9.3..1..8...3...5.....4.1.......9.2.....1...4...7..9..1.7..6.8..3..5.",
    "4x4.png":                                   "...4....2..34.12",
    "6x6.png":                                   "62.5.3......5...3..6..2....3463.6...",
    "Sudoku_app-in-progress.png":                "1...97..657926.1.4326..197.6938254172.7914...4..73659274.1....9862.79.419..64.7..",
    "any-tips-for-6x6-v0-wajbc54ipzse1.webp":    "32....56....................12....34",
    "hq720.jpg":                                 "7548.291669317528421849653786752134994576382132194867547638915.582614...139257...",
    "images.png":                                "83.469.5.549.876.367.35.984415.9.37.763.154..928734561.57.4.83.396.28.4..84.73...",
    "nr34xdhhfe8g1.jpeg":                        "564219837372864915189357462835146279941725386726938541698472153417593628253681794",
    "rules0-1.png":                              "8.5..974...3.86.9..9.4.2.6..2.5.3....5.6..9.43.4..862.......2.343.261.5.91.8.8476",
    "sudoku-coach-completed.png":                "739184526152396748684257931526748319413925687897613254265871493978432165341569872",
    "sudoku-coach-highlighted.png":              "...9....4.8..3....46.2.17...39..58...123....6....8.32...5.9..18.2.4.7.3.6...1.5.7",
    "sudoku-coach-in-progress.png":              "...9....4.8..3....46.2.17...39..58...123....6....8.32...5.9..18.2.4.7.3.6...1.5.7",
    "sudoku-coach-only-givens.png":              "...9....4.8........6.2.17...39..5....123.........8.32...5.9...8...4.7.3.6.....5.7",
    "sudoku-coach-wrong.png":                    ".56.8.13..19..65..3.295......C164....357...1.4.1.39..65.76.34...6...7.....429.7..", # C instead of 0: model confusion, 9x9 regression from 16-class training
    "sudoku-com-in-progress.png":                ".2....5317..5.3..4...14.29..52764.18.63.12759.78...4..2..37...5.1.4..9..54..816..",
    "sudoku-com-wrong.png":                      ".2....5317..5.3..4...14.29..52764.18.63.12759.78...4..2..37...5.1.42.9..54..816..",
    "sudoku1-max-en.gif":                        "..5.843..7.46...8.918....541..3....8..3.781....9.45736576..294.......81789.43...2",
    "sudokuexample.png":                         "..2.3......4...697..745...1...2......2....41.87...1...7..62....2..9.3..4...8..259",
    "0eqjbrpfp5dc1.webp":                        "..46.9.8.....3.57...3....46..2.........8..1356..1..42.....518.2.9.........8.7..5.",
    "sudoku-help-v0-rysf2a34qeye1.webp":         ".6..3295725..9..3..93...2...8432.6..52...1..331...9..283.......64.9.3....7.26.3.4",
    # TODO: goofy UIs, need fixes before these can be enabled
    # "android-sudoku.webp":    "",   # grid detection fails entirely
    # "nintendo-sudoku.jpg":    "",   # misdetected as 12x12 (should be 9x9), garbled output
    # "sud1.png":               "",   # detected as 9x9 but all cells empty
}

PHOTOS: dict[str, str] = {
    "hsLhe.jpg":                                                    "...6.47..7.6.....9.....5.8..7..2..938.......543..1..7..5.2.....3.....2.8..23.1...",
    "6978422072_33ac92fe1a_b.jpg":                                  "8...1...9.5.8.7.1...4.9.7...6.7.1.2.5.8.6.1.7.1.5.2.9...7.4.6...8.3.9.4.3...5...8",
    "EFJ5JjIXYAEvSse.jpg":                                         "7234915684685327195917862341769584239421638753852746912.431798E6178493..8.962514.", # still missing a few digits in rows 7-9
    "IMG_6062.JPG":                                                 "....6...24...15......7...9....6..1.7.7.....B.3.6..9....5...8.....149...38...5....", # just missing one digit now: a 6 in r2c7; B instead of 8 is a 16-class regression
    "images.jpg":                                                   ".391.....4.8.6...22..58.7..8.........2...9...3.6....49....1..3..4.3....87.....4..",
    "paper-16x16.jpg":                                              ".C1....6..F.7D.3.1..CDF..E..4......A.3...701F....1FF.A.......2..1..0...6.D7F93....8.3B...0C.16DFB.9..0...382..4C..C.F..56.4.0B...1E5.F.A7..0.C..67A.49E...1..F.D0F2B.C5.E.9A07..C.B1726.F...E..A.E7C......6...3....604C...21B......9..B..FAC....2.5D.8.FB.E4..C.", # now recognizing many A-F hex letters
    "video-screenshot.webp":                                        "14.5............3.6.9.1...54.6..5.7..8.1.2.5..5.6..8.32...9.5.7.9............1.29",
}

HANDWRITTEN: dict[str, str] = {
    "help-sunday-times-very-hard-sudoku-april-7-v0-85e3fsj359tc1.webp": "...15....67..F9..5..5..3.4..59438.67.2......9.689.54.1..7..4.2.24......8..6311...", # 100% of the printed digits, but ~80% of handwritten. good enough for now
    "images (1).jpg":                                               "5.19.3278..982751.2576.1.9.8..79215.192.54.8775..189.2.1527F8.9.78139.2092.A857.1", # one mistake: r8c5 should be a 3
    "images2.jpg":                                                  ".53AC.BE6446A.9123E27AFB85.4B23.156F61.27..3237.2A691.2816413F553E18.6F076A.EB281", # tough one: handwritten candidates cause confusion; now also picking up spurious A-F from 16-class model
    "paper-hand-filled-pencil-2.jpg":                               "2......54......4.7..9.17.5....2985.6.......7...6174..2...83.71.9..7.....73...16.9", # missing a lot of digits including some of the printed ones
    "paper-hand-filled-pencil.jpg":                                 "7C23.7194F63.14E777.1.89.42.3.146947..9875A75.8593274112..98557.7.67.289.98.23416", # missing a few very faint pencil digits; A-F spurious from 16-class model
    "paper-part-filled-pencil-with-notes.webp":                     "..8.....7..94.5.6.4............275585..8A1...98.AC32..6..5...9.79......473.41....", # picked up some of the notes as digits
    "vohuwbzehwhb1.jpg":                                            ".34.4159.547921168.1256....6.417.9.527.53948114.28.6FE7418521..42.1168.7.5....216", # missed some of the handwritten digits
}


@pytest.fixture(scope="module")
def reader() -> PuzzleReader:
    model = SudokuNet(WEIGHTS_PATH)
    return PuzzleReader(model=model)


@pytest.mark.parametrize("filename,expected", SCREENSHOTS.items())
def test_screenshot(reader: PuzzleReader, filename: str, expected: str) -> None:
    path = SAMPLES_DIR / "screenshots" / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in samples/screenshots/")
    result = reader.read_digits_string(path)
    assert result == expected, (
        f"\n  file    : {filename}"
        f"\n  got     : {result}"
        f"\n  expected: {expected}"
    )


@pytest.mark.parametrize("filename,expected", PHOTOS.items())
def test_photo(reader: PuzzleReader, filename: str, expected: str) -> None:
    path = SAMPLES_DIR / "photos" / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in samples/photos/")
    result = reader.read_digits_string(path)
    assert result == expected, (
        f"\n  file    : {filename}"
        f"\n  got     : {result}"
        f"\n  expected: {expected}"
    )


@pytest.mark.parametrize("filename,expected", HANDWRITTEN.items())
def test_handwritten(reader: PuzzleReader, filename: str, expected: str) -> None:
    path = SAMPLES_DIR / "handwritten" / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in samples/handwritten/")
    result = reader.read_digits_string(path)
    assert result == expected, (
        f"\n  file    : {filename}"
        f"\n  got     : {result}"
        f"\n  expected: {expected}"
    )
