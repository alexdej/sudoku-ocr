# Roadmap

## Testing

- **Pillow-free core suite**: Convert WEBP/GIF samples to PNG/JPG so they can
  run without Pillow. Keep the originals as a separate format-compatibility
  layer. The `requires_pillow` tests would then only verify that the Pillow
  fallback loads files correctly — not full OCR accuracy — keeping that suite
  minimal.

- **CI without Pillow**: Add a test run that omits Pillow (`-m "not
  requires_pillow"`) to confirm it is truly optional for the core use case.

## Current state

- Grid detection (contour + Hough line fallback), perspective correction
- Cell segmentation, digit extraction
- CNN digit classifier trained on synthetic printed digits (9 fonts, 5k samples/digit)
- Supports 4×4, 6×6, 9×9, 12×12, 16×16 grids
- Jupyter notebook for pipeline visualization and debugging

## Phase 2 — Rich cell annotations

Extend `CellInfo` with a `CellAnnotation` dataclass carrying per-cell metadata:

```python
@dataclass
class CellAnnotation:
    confidence: float        # max softmax probability from classifier
    is_given: bool | None    # True = printed given, False = user fill, None = unknown
    fill_color: tuple | None # dominant cell background color (for given/fill detection)

@dataclass
class CellInfo:
    ...
    annotation: CellAnnotation | None  # None if no model loaded
```

**Why:**
- `confidence` enables an "unreadable" signal — low confidence means something was
  detected but couldn't be classified reliably (e.g. a cursor or smudge obscuring the cell).
  Better to surface this than silently return `None` alongside genuinely empty cells.
- `is_given` is the entry point for given/fill classification (see below).
- Keeps the simple API simple — callers that only need digits can ignore `annotation`.

## Phase 2 — Given/fill classification

Printed sudoku puzzles (newspaper, app screenshot) distinguish pre-filled *givens* from
user-entered *fill* digits — usually by color, sometimes by font weight.

Goal: populate `CellAnnotation.is_given` by detecting this distinction from the cell's
color image, without requiring a second model (pure image analysis first, ML if needed).

Signals to explore:
- Background/foreground color differences between cells
- Font weight / stroke width (given digits are often bolder)

Out of scope until phase 3: handwritten fill digits (see below).

## Phase 3 — Handwritten digit support

Some users photograph a partially completed paper puzzle. Here the fill digits are
handwritten while the givens are printed.

This requires:
- Classifying each digit as printed vs. handwritten (separate from digit identity)
- A handwriting-trained classifier (MNIST or similar) for the fill digits
- Likely a two-stage pipeline or a multi-output model

`is_given` in `CellAnnotation` is already the right home for this result — the
implementation changes but the API contract stays the same.

## Phase 3 — Photo/real-world image support

Current pipeline works well on clean screenshots. Photos introduce:
- Lens distortion
- Lighting gradients and shadows
- Perspective skew beyond what the current warp handles

Likely approach: preprocessing pass (lens correction, contrast normalization) before
the existing grid detection pipeline.
