# sudoku-ocr

OCR for sudoku puzzle images. Extracts digit values from screenshots, photos, and app captures of sudoku puzzles.

## How it works

The pipeline has three stages:

1. **Grid detection** — Finds the puzzle boundary using adaptive thresholding and contour detection (with Hough line fallback), then applies a perspective transform to produce a clean top-down view.
2. **Cell segmentation** — Divides the warped grid into cells, extracts digit regions using Otsu thresholding and border clearing, and filters noise by contour area.
3. **Digit classification** — A small CNN trained on synthetic printed digits classifies each extracted digit. Images are cropped, centered, and binarized to a 28×28 canvas before inference.

## Installation

```bash
pip install -e .
```

For CPU-only PyTorch (smaller install):

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Basic — get digits as a flat list

```python
from sudoku_ocr import PuzzleReader

reader = PuzzleReader.from_weights_dir("src/sudoku_ocr/weights")

digits = reader.read_digits("puzzle.png")
# [None, None, 2, None, 3, None, None, None, None,
#  None, None, 4, None, None, None, 6, 9, 7, ...]

# Convert to row/col
for i, digit in enumerate(digits):
    row, col = divmod(i, 9)
```

### Detailed — get full cell information

```python
color_warped, cells = reader.read_cells("puzzle.png")

for cell in cells:
    print(f"({cell.row},{cell.col}): {cell.digit}")
    # cell.color_image  — BGR crop of the cell (for color analysis)
    # cell.has_digit    — whether a digit was detected
    # cell.is_given     — True = printed given, False = user fill, None = unknown
```

### Compact string representation

```python
s = reader.read_digits_string("puzzle.png")
# "..2.3......4...697..745...1..."  ('.' = empty cell)
```

### Different grid sizes

```python
# 6x6 puzzle
digits = reader.read_digits("puzzle_6x6.png", grid_size=6)
```

## Testing

```bash
docker build -f Dockerfile.test -t sudoku-ocr-test .
docker run --rm sudoku-ocr-test
```

Runs the full sample test suite (screenshots, photos, handwritten) against the
bundled model weights. Pass pytest flags as extra arguments, e.g.
`docker run --rm sudoku-ocr-test pytest tests/ -k screenshot -v`.

## Training

Model weights are included in the repo so retraining is not required for normal use.
To retrain (e.g. after changing the architecture or training data):

```bash
# GPU (recommended) — builds a Docker image with CUDA PyTorch
docker build -f Dockerfile.train -t sudoku-ocr-train .
docker run --rm --gpus all \
  -v ./data:/app/data \
  -v ./src/sudoku_ocr/weights:/app/src/sudoku_ocr/weights \
  sudoku-ocr-train
```

Generates synthetic printed digits, trains three model variants, and writes
weights to `src/sudoku_ocr/weights/`. Delete `data/printed_digits.pt` to force
regeneration of training data.

## Project structure

```
src/sudoku_ocr/
├── reader.py    # PuzzleReader — main interface
├── grid.py      # Grid detection and perspective correction
├── cells.py     # Cell segmentation and digit extraction
├── model.py     # CNN definition and inference
├── viz.py       # Debug overlay visualization
├── types.py     # CellInfo dataclass
└── weights/     # Trained model weights
```
