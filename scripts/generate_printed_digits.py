"""Generate synthetic printed digit images for training.

Renders digits 0-9 using various system fonts with augmentations
(rotation, scale jitter, thickness variation, blur, noise) to produce
28x28 binary images matching the format of _prepare_cell_image().

Output:
  data/printed_digits.pt               — single tensor file (images + labels)
  data/printed_digits_preview/{0-9}.png — composite grid per digit for inspection
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUTPUT_PT = Path("data/printed_digits.pt")
PREVIEW_DIR = Path("data/printed_digits_preview")
TARGET_PER_DIGIT = 2000
CANVAS_SIZE = 28
INNER_SIZE = 20

FONT_SEARCH_DIRS = [
    Path("/usr/share/fonts/truetype"),
    Path("C:/Windows/Fonts"),
]

FONT_PATTERNS = [
    "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
    "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
    "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf",
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
    "FreeSans.ttf", "FreeSansBold.ttf",
    "FreeSerif.ttf", "FreeSerifBold.ttf",
    "FreeMono.ttf", "FreeMonoBold.ttf",
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf",
    "cour.ttf", "courbd.ttf", "calibri.ttf", "calibrib.ttf",
    "consola.ttf", "consolab.ttf", "verdana.ttf", "verdanab.ttf",
    "georgia.ttf", "georgiab.ttf", "tahoma.ttf", "tahomabd.ttf",
]


def find_fonts() -> list[Path]:
    fonts: list[Path] = []
    for search_dir in FONT_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for pattern in FONT_PATTERNS:
            fonts.extend(search_dir.rglob(pattern))
    seen: set[str] = set()
    unique: list[Path] = []
    for f in fonts:
        name = f.name.lower()
        if name not in seen:
            seen.add(name)
            unique.append(f)
    return unique


def render_digit(
    digit: int, font_path: Path, font_size: int,
    rotation: float, scale: float, thickness: int,
    blur_radius: float, noise_prob: float,
) -> np.ndarray:
    render_size = 80
    img = Image.new("L", (render_size, render_size), 0)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (render_size - tw) / 2 - bbox[0]
    ty = (render_size - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill=255, font=font)

    if abs(rotation) > 0.5:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor=0)

    if abs(scale - 1.0) > 0.01:
        new_size = max(1, int(render_size * scale))
        img = img.resize((new_size, new_size), Image.BICUBIC)
        canvas = Image.new("L", (render_size, render_size), 0)
        if new_size > render_size:
            c = (new_size - render_size) // 2
            img = img.crop((c, c, c + render_size, c + render_size))
            canvas = img
        else:
            off = (render_size - new_size) // 2
            canvas.paste(img, (max(0, off), max(0, off)))
        img = canvas

    if thickness > 0:
        img = img.filter(ImageFilter.MaxFilter(2 * thickness + 1))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    arr = np.array(img)
    coords = np.argwhere(arr > 20)
    if len(coords) == 0:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = arr[y_min:y_max + 1, x_min:x_max + 1]

    ch, cw = cropped.shape
    max_dim = max(ch, cw)
    if max_dim == 0:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    factor = INNER_SIZE / max_dim
    new_w, new_h = max(1, int(cw * factor)), max(1, int(ch * factor))
    resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.BICUBIC))

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    x_off = (CANVAS_SIZE - new_w) // 2
    y_off = (CANVAS_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    canvas = np.where(canvas > 80, 255, 0).astype(np.uint8)

    if noise_prob > 0:
        mask = np.random.random(canvas.shape) < noise_prob
        canvas[mask] = np.where(canvas[mask] > 127, 0, 255).astype(np.uint8)

    return canvas


def save_preview(images: list[np.ndarray], digit: int) -> None:
    n = len(images)
    cols = math.isqrt(n)
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * CANVAS_SIZE, cols * CANVAS_SIZE), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * CANVAS_SIZE:(r + 1) * CANVAS_SIZE,
             c * CANVAS_SIZE:(c + 1) * CANVAS_SIZE] = img
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(PREVIEW_DIR / f"{digit}.png")


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

    all_images: list[np.ndarray] = []
    all_labels: list[int] = []

    t0 = time.time()
    for digit in range(0, 10):
        digit_images: list[np.ndarray] = []
        while len(digit_images) < TARGET_PER_DIGIT:
            img = render_digit(
                digit,
                font_path=random.choice(fonts),
                font_size=random.randint(40, 65),
                rotation=random.uniform(-10, 10),
                scale=random.uniform(0.85, 1.15),
                thickness=random.choice([0, 0, 0, 1, 1, 2]),
                blur_radius=random.choice([0, 0, 0, 0.5, 0.8]),
                noise_prob=random.choice([0, 0, 0, 0, 0.005, 0.01]),
            )
            if img.sum() < 200:
                continue
            digit_images.append(img)

        save_preview(digit_images, digit)
        all_images.extend(digit_images)
        all_labels.extend([digit] * len(digit_images))
        print(f"Digit {digit}: {len(digit_images)} images")

    images_t = torch.from_numpy(np.array(all_images)).float() / 255.0
    images_t = images_t.unsqueeze(1)
    labels_t = torch.tensor(all_labels, dtype=torch.long)

    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images_t, "labels": labels_t}, OUTPUT_PT)
    print(f"\nTotal: {len(all_labels)} images")
    print(f"Saved to {OUTPUT_PT} ({OUTPUT_PT.stat().st_size / 1024 / 1024:.1f}MB)")
    print(f"Previews in {PREVIEW_DIR}/")
    print(f"Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
