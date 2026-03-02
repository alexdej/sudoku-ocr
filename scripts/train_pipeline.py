"""Generate synthetic printed digits + train SudokuNet on printed data only.

Combines digit generation and training in one script. Optimized for GPU
with mixed precision, parallel data loading, and larger batches.

=== HOW TO RUN ===

--- Option 1: Docker with GPU (recommended for isolation) ---

Prerequisites:
  - NVIDIA driver installed on host (you already have this with a 4080)
  - NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    On Windows with Docker Desktop: Settings > Resources > GPU > enable

Build the GPU training image:
  docker build -f Dockerfile.train -t sudoku-ocr-train .

Run with GPU access:
  docker run --rm --gpus all \
    -v ./data:/app/data \
    -v ./src/sudoku_ocr/weights:/app/src/sudoku_ocr/weights \
    sudoku-ocr-train

--- Option 2: Local with CUDA PyTorch ---

Install CUDA-enabled PyTorch (if you only have the CPU version):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Then run directly:
  python scripts/train_pipeline.py

--- Option 3: Docker CPU-only (current Dockerfile, slow) ---

  docker build -t sudoku-ocr .
  docker run --rm \
    -v ./data:/app/data \
    -v ./src/sudoku_ocr/weights:/app/src/sudoku_ocr/weights \
    sudoku-ocr python scripts/train_pipeline.py

=== EXPECTED PERFORMANCE ===
  RTX 4080 (CUDA):  ~2-3 min total (generate + train)
  CPU (Docker):     ~15-20 min total
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import DataLoader, TensorDataset

from sudoku_ocr.model import _SudokuNetCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE_CPU = 128
BATCH_SIZE_GPU = 512  # 4080 has 16GB — this model is tiny, go big
EPOCHS = 15
LR = 1e-3
DATA_DIR = Path("data")
PRINTED_PT = DATA_DIR / "printed_digits.pt"
PREVIEW_DIR = DATA_DIR / "printed_digits_preview"
WEIGHTS_PATH = Path("src/sudoku_ocr/weights/digit_classifier.pt")
TARGET_PER_DIGIT = 5000
CANVAS_SIZE = 28
INNER_SIZE = 20

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------
FONT_SEARCH_DIRS = [
    Path("/usr/share/fonts/truetype"),  # Linux / Docker
    Path("C:/Windows/Fonts"),           # Windows
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


# ---------------------------------------------------------------------------
# Digit rendering
# ---------------------------------------------------------------------------
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
    """Save a grid preview of all generated images for one digit."""
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


# ---------------------------------------------------------------------------
# Step 1: Generate printed digits → single .pt file + preview PNGs
# ---------------------------------------------------------------------------
def generate_printed_digits() -> None:
    if PRINTED_PT.exists():
        data = torch.load(PRINTED_PT, weights_only=True)
        print(f"[generate] {PRINTED_PT} already exists "
              f"({data['images'].shape[0]} images), skipping.")
        print(f"[generate] Delete {PRINTED_PT} to regenerate.")
        return

    fonts = find_fonts()
    if not fonts:
        print("[generate] ERROR: No fonts found!")
        return

    print(f"[generate] Found {len(fonts)} fonts, generating {TARGET_PER_DIGIT} images/digit...")
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
                font_size=random.randint(35, 70),
                rotation=random.uniform(-15, 15),
                scale=random.uniform(0.80, 1.20),
                thickness=random.choice([0, 0, 0, 1, 1, 2, 2, 3]),
                blur_radius=random.choice([0, 0, 0, 0.5, 0.8, 1.2]),
                noise_prob=random.choice([0, 0, 0, 0.005, 0.01, 0.02]),
            )
            if img.sum() < 200:
                continue
            digit_images.append(img)

        save_preview(digit_images, digit)
        all_images.extend(digit_images)
        all_labels.extend([digit] * len(digit_images))

    # Save as single .pt: float32 tensors normalized to [0, 1]
    images_t = torch.from_numpy(np.array(all_images)).float() / 255.0  # (N, 28, 28)
    images_t = images_t.unsqueeze(1)  # (N, 1, 28, 28)
    labels_t = torch.tensor(all_labels, dtype=torch.long)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images_t, "labels": labels_t}, PRINTED_PT)
    print(f"[generate] Done: {len(all_labels)} images in {time.time() - t0:.1f}s")
    print(f"[generate] Saved to {PRINTED_PT} ({PRINTED_PT.stat().st_size / 1024 / 1024:.1f}MB)")
    print(f"[generate] Previews in {PREVIEW_DIR}/")


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(images)
            correct += (output.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total if total > 0 else 0.0


def train() -> None:
    # --- Device setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
        print(f"[train] CUDA: {torch.cuda.get_device_name(0)} ({vram / 1024**3:.0f}GB)")
    else:
        print("[train] WARNING: No CUDA — training on CPU (will be slow)")

    batch_size = BATCH_SIZE_GPU if use_cuda else BATCH_SIZE_CPU
    num_workers = 4 if use_cuda else 0
    print(f"[train] batch_size={batch_size}, num_workers={num_workers}, epochs={EPOCHS}")

    # --- Datasets ---
    if not PRINTED_PT.exists():
        print(f"[train] ERROR: {PRINTED_PT} not found — run generation first.")
        return

    data = torch.load(PRINTED_PT, weights_only=True)
    n = len(data["labels"])

    # Reproducible 90/10 train/val split
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    split = int(n * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]
    train_data = TensorDataset(data["images"][train_idx], data["labels"][train_idx])
    val_data   = TensorDataset(data["images"][val_idx],   data["labels"][val_idx])
    print(f"[train] Printed digits: {len(train_data)} train, {len(val_data)} val")

    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_data, **loader_kwargs)

    # --- Model + optimizer ---
    model = _SudokuNetCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    # Mixed precision: float16 forward/backward, float32 weight updates
    use_amp = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Training loop ---
    t0 = time.time()
    val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images)
                loss = criterion(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (output.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:2d}/{EPOCHS}: loss={avg_loss:.4f} "
              f"train={train_acc:.4f} val={val_acc:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.1e} [{elapsed:.0f}s]")

        scheduler.step(val_acc)

    # --- Save ---
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\n[train] Saved to {WEIGHTS_PATH}")
    print(f"[train] Final val accuracy: {val_acc:.4f}")
    print(f"[train] Total time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Step 1: Generate printed digit training data")
    print("=" * 60)
    generate_printed_digits()

    print()
    print("=" * 60)
    print("Step 2: Train on printed digits")
    print("=" * 60)
    train()


if __name__ == "__main__":
    main()
