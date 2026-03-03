# Roadmap

## Current state

**35/35 tests passing** (21 screenshot, 7 photo, 7 handwritten)

### Pipeline

- Grid detection: contour-based (`_find_grid_contour`) with Hough line fallback
- Perspective correction: four-corner warp to top-down view
- **Mesh warp** (`refine_grid_warp`): two-stage distortion correction for curved/warped paper
  - Stage 1: `_find_contour_mesh` — detects cell-interior hole contours (RETR_CCOMP) to directly
    measure cell boundary positions; gaps filled by neighbor linear extrapolation then centroid fallback.
    Applied when cell coverage ≥ 30 % and RMS deviation ≥ 15 px.
  - Stage 2: `_find_grid_intersections` — centroid of the H∩V line crossing image; fallback when
    contour coverage is too low (flat paper, thick lines, few empty cells).
- Grid size detection: morphological line extraction + projection profile peak counting;
  supports 4×4, 6×6, 9×9, 12×12, 16×16
- Cell segmentation: adaptive line positions (not assumed uniform), 8% padding, upscaling for small cells
- Digit extraction: Otsu threshold + bimodality gate + border clearing + contour filters
- CNN classifier: synthetic printed digits, 9 fonts, 5 000 samples/digit, 99.96 % val accuracy;
  separate models for 1–9 (standard) and 1–9+A–F (hex/16×16)
- Given/fill classification: HSV color analysis on digit pixels (`_classify_digit_color`)

### What works well

- Screenshots and clean app images: essentially 100 % digit accuracy
- Printed newspaper/book photos with moderate curl: strong results (EFJ5: 81/81 after mesh warp)
- Handwritten digits: partial recognition (~70–85 % depending on legibility)

### Known gaps

- Unusual UIs: `android-sudoku.webp` (grid detection fails), `nintendo-sudoku.jpg` (size
  misdetected), `sud1.png` (all empty) — commented out in test suite
- Handwritten fill over printed puzzle: candidates/notes sometimes detected as digits
- Aggressive lighting gradients or heavy shadows in photos not yet handled

---

## Testing

- **Pillow-free core suite**: Convert WEBP/GIF samples to PNG/JPG so the core suite runs
  without Pillow. Keep originals as a format-compatibility layer; `requires_pillow` tests would
  then only verify the fallback loader, not full OCR accuracy.

- **CI without Pillow**: Add a test run (`-m "not requires_pillow"`) to confirm Pillow is truly
  optional for the core use case.

---

## Phase 2 — Confidence scores

Add `confidence: float` to `CellInfo` (max softmax probability from the classifier).

**Why:** a low-confidence digit is meaningfully different from an empty cell — it signals
something was detected but couldn't be classified reliably (cursor, smudge, partial crop).
Callers can use this to surface an "unreadable" state rather than silently dropping the cell.

The `digit` field stays as-is; `confidence` is additive and callers that don't need it can
ignore it.

---

## Phase 3 — Handwritten digit support

Some users photograph a partially completed paper puzzle where fill digits are handwritten and
givens are printed. Current classifier handles printed digits only.

Options:
- Two-stage pipeline: classify each digit as printed vs. handwritten, then route to the
  appropriate model (existing CNN for printed; MNIST-trained or EMNIST for handwritten)
- Multi-output model trained on both, with a printed/handwritten head
- `is_given` in `CellInfo` is already the right output field — the implementation changes but
  the API contract stays the same

---

## Phase 3 — Remaining photo challenges

The mesh warp handles paper curvature well. Remaining photo-specific challenges:

- Heavy directional lighting / shadows within cells (digit partially obscured)
- Very low-contrast prints (faded newspaper, worn paper)
- Lens distortion at wide angle (barrel/pincushion) — a radial undistortion pass before
  grid detection would help
- Grids where the outer border is not the dominant contour (large page scans, white borders)
