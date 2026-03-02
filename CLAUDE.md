# Sudoku OCR — Project Instructions

## Running Tests

Always run tests in Docker (no local Python env):

```bash
# Build image (required after any code or sample change; weights are bundled in the wheel)
docker build -f Dockerfile.test -t sudoku-ocr-test .

# Run full test suite
docker run --rm sudoku-ocr-test

# Run a specific test group
docker run --rm sudoku-ocr-test pytest tests/ -v -k screenshot
docker run --rm sudoku-ocr-test pytest tests/ -v -k photo
docker run --rm sudoku-ocr-test pytest tests/ -v -k handwritten
```

`Dockerfile.test` is CPU-only and builds a proper wheel (not an editable install).
The `sudoku-ocr-train` image uses CUDA for training.

## Training

```bash
docker build -f Dockerfile.train -t sudoku-ocr-train .
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/src/sudoku_ocr/weights":/app/src/sudoku_ocr/weights \
  sudoku-ocr-train
```

**Important:** On Windows/git bash, volume mounts require `MSYS_NO_PATHCONV=1` and `$(pwd)/` prefix.
Without this, Docker silently ignores the bind mount and weights are never written to the host.

Training produces multiple model variants in `src/sudoku_ocr/weights/`:
- `digits_1_9.pt`  — 9 classes (1-9), for standard 9×9 grids
- `digits_hex.pt`  — 16 classes (1-9+A-G), for 16×16 grids
- `digits_0_9.pt`  — 10 classes (0-9), optional

`PuzzleReader.from_weights_dir()` auto-selects the right model by grid size.
Model weights (`*.pt`) are tracked in git and bundled with the package. Training script: `scripts/train_pipeline.py`.

## Test Suite Structure

`tests/test_samples.py` has three parametrized test functions:
- `test_screenshot` — app/browser screenshots (21 images, should be ~100% accurate)
- `test_photo` — printed-digit photos taken with a camera (7 images, mostly accurate)
- `test_handwritten` — photos with handwritten digits (7 images, partial accuracy expected)

Samples live in:
- `samples/screenshots/`
- `samples/photos/`
- `samples/handwritten/`

Expected strings use `'.'` for empty cells. Standard 9×9 grids use `1-9`; 16×16 grids use `1-9` and `A-G`.

To update expected values after an intentional change, run the test with `-v`, copy the "got" string from the failure output, and paste it into the test file. Always verify the new value is correct before committing.

## Key Thresholds in `cells.py`

These have been carefully tuned — don't change them without a concrete reason:

| Constant | Value | Purpose |
|---|---|---|
| `MIN_DIGIT_AREA_RATIO` | 0.015 | Min contour area fraction; below this → empty. Thin digits (1, 7) can be as low as 1.7% |
| `MIN_BIMODALITY` | 5 | `abs(cell_mean - otsu_thresh) < 5` → unimodal noise, return None. Prevents phantom detections on paper grain |
| `MIN_DIGIT_HEIGHT_RATIO` | 0.40 | Digit bounding box must span ≥40% of cell height |
| `CELL_PADDING_RATIO` | 0.08 | Trim 8% from cell edges before extraction. Reducing this globally causes grid-line bleed |
| `MIN_CELL_SIZE` | 50 | Upscale cells below this pixel size |

## Version Control

This repo uses **Jujutsu (jj)** colocated with Git.

- Do NOT use `git add` / `git commit` — jj tracks the working copy automatically
- Use `jj describe` to set the description on the current commit
- Use `jj new` to start a new commit on top of the current one
- Use `jj squash` to fold a small fixup into the parent commit
- Use `jj git push` to push to GitHub
- Do NOT include `Co-Authored-By` lines in commit messages
