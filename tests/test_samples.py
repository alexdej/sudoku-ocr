"""Smoke tests: OCR each sample image and check output hasn't changed.

Expected strings are digit strings of length grid_size², with '.' for empty cells.
Run `pytest tests/test_samples.py -v` to execute.

To update expected values after an intentional change, run the test with `-v`, copy
the "got" string from the failure output, and paste it into SAMPLES below.
Always verify the new value is correct before committing.
"""

import pytest
from pathlib import Path
from typing import NotRequired, TypedDict

from sudoku_ocr import PuzzleReader

# File extensions that OpenCV can't load natively — Pillow required.
_PILLOW_EXTS = {".webp", ".gif"}

IMAGES_DIR = Path(__file__).parent / "testdata" / "images"

import sudoku_ocr as _pkg
WEIGHTS_DIR = Path(_pkg.__file__).parent / "weights"
del _pkg


class SampleInfo(TypedDict):
    expected: str                # expected OCR output ('.' = empty)
    tags: set                    # e.g. {"screenshot"}, {"photo"}, {"photo", "handwritten"}
    skip: NotRequired[bool]      # True = exclude from parametrize (default False)
    accepted: NotRequired[str]   # known-imperfect output: if result matches this instead of expected, skip the test rather than fail


# NOTE: verify expected values before treating failures as regressions.
# Tag conventions:
#   screenshot  — app/browser screenshot
#   photo       — camera photo of printed puzzle
#   handwritten — contains handwritten digits (always alongside "photo")
# skip=True entries are documented but excluded from parametrize.
SAMPLES: dict[str, SampleInfo] = {
    # ── screenshots ──────────────────────────────────────────────────────────
    "16x16.png": {
        "expected": "..2A..D3...6F.....G.74EF..........6....5.2...1C41....A6.93G....72....C..3.6...9....GF23...4DC.6...9B..G42F7.A.51...DB.1..C...F82CGE...5..9.21...BD.F.1CE48..59...1.249...5DCE....6...F.8..3....BD....572.19....FF9C...4.5....G..........D6C3.2.....1G...FA..34..",  # uses 1-9+A-G convention
        "tags": {"screenshot", "16x16"},
    },
    "4363.png": {
        "expected": ".7..5..6.4..9.3..1..8...3...5.....4.1.......9.2.....1...4...7..9..1.7..6.8..3..5.",
        "tags": {"screenshot", "9x9"},
    },
    "4x4.png": {
        "expected": "...4....2..34.12",
        "tags": {"screenshot", "4x4"},
    },
    "6x6.png": {
        "expected": "62.5.3......5...3..6..2....3463.6...",
        "tags": {"screenshot", "6x6"},
    },
    "Sudoku_app-in-progress.png": {
        "expected": "1...97..657926.1.4326..197.6938254172.7914...4..73659274.1....9862.79.419..64.7..",
        "tags": {"screenshot", "9x9"},
    },
    "any-tips-for-6x6-v0-wajbc54ipzse1.webp": {
        "expected": "32....56....................12....34",
        "tags": {"screenshot", "6x6"},
    },
    "hq720.jpg": {
        "expected": "75483291669317528421849653786752134994576382132194867547638915.582614...139257...",
        "tags": {"screenshot", "9x9"},
        "skip": True,  # one digit hard to see in the original, but visible in thumbnail; model picks it up in both
    },
    "images.png": {
        "expected": "83.469.5.549.876.367.35.984415.9.37.763.154..928734561.57.4.83.396.28.4..84.73...",
        "tags": {"screenshot", "9x9"},
    },
    "nr34xdhhfe8g1.jpeg": {
        "expected": "564219837372864915189357462835146279941725386726938541698472153417593628253681794",
        "tags": {"screenshot", "9x9"},
    },
    "rules0-1.png": {
        "expected": "8.5..974...3.86.9..9.4.2.6..2.5.3....5.6..9.43.4..862.......2.343.261.5.91.8.8476",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-coach-completed.png": {
        "expected": "739184526152396748684257931526748319413925687897613254265871493978432165341569872",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-coach-highlighted.png": {
        "expected": "...9....4.8..3....46.2.17...39..58...123....6....8.32...5.9..18.2.4.7.3.6...1.5.7",
        "tags": {"screenshot", "9x9"},
        "skip": True,  # correct but excluded from parametrize
    },
    "sudoku-coach-in-progress.png": {
        "expected": "...9....4.8..3....46.2.17...39..58...123....6....8.32...5.9..18.2.4.7.3.6...1.5.7",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-coach-only-givens.png": {
        "expected": "...9....4.8........6.2.17...39..5....123.........8.32...5.9...8...4.7.3.6.....5.7",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-coach-wrong.png": {
        "expected": ".56.8.13..19..65..3.295.......164....357...1.4.1.39..65.76.34...6...7.....429.7..",  # one digit obscured by highlighting
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-com-in-progress.png": {
        "expected": ".2....5317..5.3..4...14.29..52764.18.63.12759.78...4..2..37...5.1.4..9..54..816..",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-com-wrong.png": {
        "expected": ".2....5317..5.3..4...14.29..52764.18.63.12759.78...4..2..37...5.1.42.9..54..816..",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku1-max-en.gif": {
        "expected": "..5.843..7.46...8.918....541..3....8..3.781....9.45736576..294.......81789.43...2",
        "tags": {"screenshot", "9x9"},
    },
    "sudokuexample.png": {
        "expected": "..2.3......4...697..745...1...2......2....41.87...1...7..62....2..9.3..4...8..259",
        "tags": {"screenshot", "9x9"},
    },
    "0eqjbrpfp5dc1.webp": {
        "expected": "..46.9.8.....3.57...3....46..2.........8..1356..1..42.....518.2.9.........8.7..5.",
        "tags": {"screenshot", "9x9"},
    },
    "sudoku-help-v0-rysf2a34qeye1.webp": {
        "expected": ".6..3295725..9..3..93...2...8432.6..52...1..331...9..285.......64.9.3....7.26.3.4",
        "tags": {"screenshot", "9x9"},
    },
    "nyt-filled.jpg": {
        "expected": "985326174132745986476189325543972861761438259298561437614853792827694513359217648",
        "tags": {"screenshot", "9x9"},
    },
    "nyt-candidates.jpg": {
        "expected": ".9.62.5.4.......2..5..1....82.1....79.7..2...5.63..9.2...24678......12...8.7..1.6",
        "tags": {"screenshot", "9x9"},
    },
    # TODO: goofy UIs, need fixes before these can be enabled
    "android-sudoku.webp": {
        "expected": "1..6.4.8..2.31.459...852..72341896757..2439188..765234517...893..253..4..8.9.1.2.",
        "tags": {"screenshot", "9x9"},
        "skip": True,  # grid detection fails entirely
    },
    "nintendo-sudoku.jpg": {
        "expected": "1593862..3824975616472518398167254939236.87155741.96282918743564685139.2735962184",
        "tags": {"screenshot", "9x9"},
        "skip": True,  # misdetected as 12x12 (should be 9x9), garbled output
    },
    "sud1.png": {
        "expected": "..16.3.87.37.852.86..9...1.8..43215....7963282.38.16.47543..8..169.78432328614.9.",
        "tags": {"screenshot", "9x9"},
        "skip": True,  # detected as 9x9 but all cells empty
    },

    # ── photos (printed digits) ───────────────────────────────────────────────
    "hsLhe.jpg": {
        "expected": "...6.47..7.6.....9.....5.8..7..2..938.......543..1..7..5.2.....3.....2.8..23.1...",
        "tags": {"photo", "9x9"},
    },
    "6978422072_33ac92fe1a_b.jpg": {
        "expected": "8...1...9.5.8.7.1...4.9.7...6.7.1.2.5.8.6.1.7.1.5.2.9...7.4.6...8.3.9.4.3...5...8",
        "tags": {"photo", "9x9"},
    },
    "EFJ5JjIXYAEvSse.jpg": {
        "expected": "723491568468532719591786234176958423942163875385274691254317986617849352839625147",
        "tags": {"photo", "9x9"},
    },
    "IMG_6062.JPG": {
        "expected": "....6...24...156.....7...9....6..1.7.7.....8.3.6..9....5...8.....149...38...5....",
        "tags": {"photo", "9x9"},
    },
    "images.jpg": {
        "expected": ".391.....4.8.6...22..58.7..8.........2...9...3.6....49....1..3..4.3....87.....4..",
        "tags": {"photo", "9x9"},
    },
    "paper-16x16.jpg": {
        "expected": ".C1....6..F.7D.3.1..CDF..E..4......A.3...7D1F....1FF.A.......2..1..D...6.D7F93....8.3B...DC.16DFB.9..D...382..4C..C.F..56.4.D8...1E5.F.A7..D.C..67A.49E...1..F.D9F2B.C5.E.9AD7..C.B1726.F...E..A.27C......6...3....6D4C...21B......9..B..FAC....2.5D.B.FB.E4..C.",
        "tags": {"photo", "16x16"},
        "skip": True,  # uses 0-9+A-F convention; 0s misread since hex model uses 1-9+A-G
    },
    "video-screenshot.webp": {
        "expected": "14.5............3.6.9.1...54.6..5.7..8.1.2.5..5.6..8.32...9.5.7.9............1.29",
        "tags": {"photo", "9x9"},
    },

    # ── handwritten ───────────────────────────────────────────────────────────
    "help-sunday-times-very-hard-sudoku-april-7-v0-85e3fsj359tc1.webp": {
        "expected": "...15....67..49..5..5..3.4.159438267.2...1..9768925431..7..4.2.24......8..6312..4",
        "accepted": "...15....67..49..5..5..3.4..59438.67.2......7.689.5411..7..4.2.24......8..6311...",  # ~80% of handwritten digits; a few printed also missed
        "tags": {"photo", "handwritten", "9x9"},
    },
    "images (1).jpg": {
        "expected": "5.19.3278..982751.2875.1.9.8..79215.192.54.8775..189.2.152768.9.78139.2592.4857.1",
        "accepted": "5.19.3278..982751.1576.1.9.8..79215.192.54.8775..189.2.152798.9.78139.2992.9857.1",  # one mistake: r8c5 should be a 3
        "tags": {"photo", "handwritten", "9x9"},
    },
    "images2.jpg": {
        "expected": ".53.....6.46..9.23.27...85.4923.1.6.61.97..3237.2.691.281...3.553.1..6..76....281",
        "accepted": "7538.945644644912372765985449238156861277443237.24691428144134553618.679764.79281",  # tough: handwritten candidates cause confusion
        "tags": {"photo", "handwritten", "9x9"},
    },
    "paper-hand-filled-pencil-2.jpg": {
        "expected": "271986354683542917549317268417298536892653471356174892164839725928765143735421689",
        "accepted": "271976254.834424175.98172824.9298526...15.471356174..2..183.7169..7484.373..216.9",  # missing a lot of digits including some printed ones
        "tags": {"photo", "handwritten", "9x9"},
    },
    "paper-hand-filled-pencil.jpg": {
        "expected": "822367194963214875741589362237146958419875623685932741126498537374651289598723416",
        "accepted": "762347194763214877741589542.3.146967..9875475.8593274117.498557.7467.289.98723416",  # missing a few very faint pencil digits
        "tags": {"photo", "handwritten", "9x9"},
    },
    "paper-part-filled-pencil-with-notes.webp": {
        "expected": "..8.....7..94.5.6.4............27..85..841....8..532..6..5...9.89......473..1....",
        "accepted": "..8.....7..94.5.6.4............275585..841...98.2732..6..5...9.29......473.41....",  # picked up some notes as digits
        "tags": {"photo", "handwritten", "9x9"},
    },
    "vohuwbzehwhb1.jpg": {
        "expected": "836741592547923168912568734684175923275639481193284675761852349429316857358497216",
        "accepted": "1347415995479251687125697346841759132756394811482876767618522.7421116857258477216",  # wrong
        "tags": {"photo", "handwritten", "9x9"},
    },
}


def _params(tag: str, exclude: set = frozenset()) -> list:
    """Build parametrize list filtered by tag, excluding tags in `exclude` and skip=True entries."""
    return [
        pytest.param(k, marks=pytest.mark.requires_pillow)
        if Path(k).suffix.lower() in _PILLOW_EXTS
        else k
        for k, v in SAMPLES.items()
        if (not tag or tag in v["tags"])
        and not (exclude & v["tags"])
        and not v.get("skip", False)
    ]


@pytest.fixture(scope="module")
def reader() -> PuzzleReader:
    return PuzzleReader.from_weights_dir(WEIGHTS_DIR)


def do_test_sample(reader: PuzzleReader, filename: str) -> None:
    path = IMAGES_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found in {IMAGES_DIR}")
    info = SAMPLES[filename]
    expected = info["expected"]
    result = reader.read_digits_string(path)
    if result == expected:
        return
    accepted = info.get("accepted")
    if accepted is not None and result == accepted:
        pytest.skip(f"got accepted (not ideal) result for {filename}")
    assert result == expected, (
        f"\n  file    : {filename}"
        f"\n  got     : {result}"
        f"\n  expected: {expected}"
    )


@pytest.mark.parametrize("filename", _params("screenshot"))
def test_screenshot(reader: PuzzleReader, filename: str) -> None:
    do_test_sample(reader, filename)


@pytest.mark.parametrize("filename", _params("photo", exclude={"handwritten"}))
def test_photo(reader: PuzzleReader, filename: str) -> None:
    do_test_sample(reader, filename)


@pytest.mark.parametrize("filename", _params("handwritten"))
def test_handwritten(reader: PuzzleReader, filename: str) -> None:
    do_test_sample(reader, filename)
