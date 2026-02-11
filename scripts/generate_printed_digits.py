"""Generate synthetic printed digit images for training.

Renders digits 1-9 using various system fonts with augmentations
(rotation, scale jitter, thickness variation, blur, noise) to produce
28x28 binary images matching the format of _prepare_cell_image().

Output: data/printed_digits/{digit}/*.png  (~2000 per digit)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = Path("data/printed_digits")
TARGET_PER_DIGIT = 2000
CANVAS_SIZE = 28
INNER_SIZE = 20  # digit fits in ~20x20 centered in 28x28

# Common Linux font paths (installed via Dockerfile)
FONT_SEARCH_DIRS = [
    Path("/usr/share/fonts/truetype"),
    Path("C:/Windows/Fonts"),
]

# Font filename patterns to look for
FONT_PATTERNS = [
    # Liberation family
    "LiberationSans-Regular.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSerif-Regular.ttf",
    "LiberationSerif-Bold.ttf",
    "LiberationMono-Regular.ttf",
    "LiberationMono-Bold.ttf",
    # DejaVu family
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf",
    # FreeFont family
    "FreeSans.ttf",
    "FreeSansBold.ttf",
    "FreeSerif.ttf",
    "FreeSerifBold.ttf",
    "FreeMono.ttf",
    "FreeMonoBold.ttf",
    # Windows fallbacks
    "arial.ttf",
    "arialbd.ttf",
    "times.ttf",
    "timesbd.ttf",
    "cour.ttf",
    "courbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "consola.ttf",
    "consolab.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
]


def find_fonts() -> list[Path]:
    """Find available font files on the system."""
    fonts: list[Path] = []
    for search_dir in FONT_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for pattern in FONT_PATTERNS:
            found = list(search_dir.rglob(pattern))
            fonts.extend(found)
    # Deduplicate
    seen: set[str] = set()
    unique: list[Path] = []
    for f in fonts:
        name = f.name.lower()
        if name not in seen:
            seen.add(name)
            unique.append(f)
    return unique


def render_digit(
    digit: int,
    font_path: Path,
    font_size: int,
    rotation: float,
    scale: float,
    thickness: int,
    blur_radius: float,
    noise_prob: float,
) -> np.ndarray:
    """Render a single digit image with augmentations.

    Returns a 28x28 uint8 numpy array (white digit on black background).
    """
    # Render at higher resolution for quality, then downscale
    render_size = 80
    img = Image.new("L", (render_size, render_size), 0)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (render_size - tw) / 2 - bbox[0]
    ty = (render_size - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill=255, font=font)

    # Apply rotation
    if abs(rotation) > 0.5:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor=0)

    # Apply scale jitter
    if abs(scale - 1.0) > 0.01:
        new_size = max(1, int(render_size * scale))
        img = img.resize((new_size, new_size), Image.BICUBIC)
        # Re-center in render_size canvas
        canvas = Image.new("L", (render_size, render_size), 0)
        offset_x = (render_size - new_size) // 2
        offset_y = (render_size - new_size) // 2
        # Crop if scaled up
        if new_size > render_size:
            crop_x = (new_size - render_size) // 2
            crop_y = (new_size - render_size) // 2
            img = img.crop((crop_x, crop_y, crop_x + render_size, crop_y + render_size))
            canvas = img
        else:
            canvas.paste(img, (max(0, offset_x), max(0, offset_y)))
        img = canvas

    # Morphological thickness variation (dilate to thicken)
    if thickness > 0:
        arr = np.array(img)
        kernel_size = 2 * thickness + 1
        from PIL import ImageFilter as _IF
        img = img.filter(ImageFilter.MaxFilter(kernel_size))

    # Apply slight blur
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Downscale to 28x28 via INNER_SIZE centering
    # First crop to bounding box, then fit into INNER_SIZE x INNER_SIZE
    arr = np.array(img)
    coords = np.argwhere(arr > 20)
    if len(coords) == 0:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = arr[y_min:y_max + 1, x_min:x_max + 1]

    # Resize to fit in INNER_SIZE maintaining aspect ratio
    ch, cw = cropped.shape
    max_dim = max(ch, cw)
    if max_dim == 0:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    factor = INNER_SIZE / max_dim
    new_w = max(1, int(cw * factor))
    new_h = max(1, int(ch * factor))
    resized = np.array(
        Image.fromarray(cropped).resize((new_w, new_h), Image.BICUBIC)
    )

    # Center in 28x28 canvas
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    x_off = (CANVAS_SIZE - new_w) // 2
    y_off = (CANVAS_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Binarize with threshold
    canvas = np.where(canvas > 80, 255, 0).astype(np.uint8)

    # Add salt-and-pepper noise
    if noise_prob > 0:
        noise_mask = np.random.random(canvas.shape) < noise_prob
        canvas[noise_mask] = np.where(canvas[noise_mask] > 127, 0, 255).astype(np.uint8)

    return canvas


def main() -> None:
    fonts = find_fonts()
    if not fonts:
        print("ERROR: No fonts found! Install font packages or run in Docker.")
        return

    print(f"Found {len(fonts)} fonts:")
    for f in fonts:
        print(f"  {f.name}")

    random.seed(42)
    np.random.seed(42)

    total_generated = 0
    for digit in range(0, 10):
        digit_dir = OUTPUT_DIR / str(digit)
        digit_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        while count < TARGET_PER_DIGIT:
            font_path = random.choice(fonts)
            font_size = random.randint(40, 65)
            rotation = random.uniform(-10, 10)
            scale = random.uniform(0.85, 1.15)
            thickness = random.choice([0, 0, 0, 1, 1, 2])  # mostly no thickening
            blur_radius = random.choice([0, 0, 0, 0.5, 0.8])
            noise_prob = random.choice([0, 0, 0, 0, 0.005, 0.01])

            img = render_digit(
                digit, font_path, font_size, rotation, scale,
                thickness, blur_radius, noise_prob,
            )

            # Skip if nearly empty
            if img.sum() < 200:
                continue

            out_path = digit_dir / f"{count:05d}.png"
            Image.fromarray(img).save(out_path)
            count += 1

        total_generated += count
        print(f"Digit {digit}: {count} images")

    print(f"\nTotal: {total_generated} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
