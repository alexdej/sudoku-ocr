"""Generate synthetic printed digits + train SudokuNet on MNIST + printed data.

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

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from sudoku_ocr.model import _SudokuNetCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE_CPU = 128
BATCH_SIZE_GPU = 512  # 4080 has 16GB — this model is tiny, go big
EPOCHS = 15
LR = 1e-3
DATA_DIR = Path("data")
PRINTED_DIR = DATA_DIR / "printed_digits"
WEIGHTS_PATH = Path("src/sudoku_ocr/weights/digit_classifier.pt")
TARGET_PER_DIGIT = 2000
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


# ---------------------------------------------------------------------------
# Step 1: Generate printed digits
# ---------------------------------------------------------------------------
def generate_printed_digits() -> None:
    if PRINTED_DIR.exists() and any(PRINTED_DIR.iterdir()):
        count = sum(1 for _ in PRINTED_DIR.rglob("*.png"))
        print(f"[generate] {PRINTED_DIR} already exists with {count} images, skipping.")
        print(f"[generate] Delete {PRINTED_DIR} to regenerate.")
        return

    fonts = find_fonts()
    if not fonts:
        print("[generate] ERROR: No fonts found!")
        return

    print(f"[generate] Found {len(fonts)} fonts, generating {TARGET_PER_DIGIT} images/digit...")
    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    total = 0
    for digit in range(0, 10):
        digit_dir = PRINTED_DIR / str(digit)
        digit_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        while count < TARGET_PER_DIGIT:
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
            Image.fromarray(img).save(digit_dir / f"{count:05d}.png")
            count += 1
        total += count
    print(f"[generate] Done: {total} images in {time.time() - t0:.1f}s")


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
        torch.backends.cudnn.benchmark = True  # autotuner for conv algorithms
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
        print(f"[train] CUDA: {torch.cuda.get_device_name(0)} ({vram / 1024**3:.0f}GB)")
    else:
        print("[train] WARNING: No CUDA — training on CPU (will be slow)")

    batch_size = BATCH_SIZE_GPU if use_cuda else BATCH_SIZE_CPU
    # Workers: 4 is a good default; on Windows > 0 requires if __name__ guard (we have it)
    num_workers = 4 if use_cuda else 0
    print(f"[train] batch_size={batch_size}, num_workers={num_workers}, epochs={EPOCHS}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # --- Datasets ---
    mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    printed_train = None
    if PRINTED_DIR.exists():
        printed_train = datasets.ImageFolder(str(PRINTED_DIR), transform=transform)
        print(f"[train] Printed digits: {len(printed_train)} images")

    if printed_train is not None:
        train_data = ConcatDataset([mnist_train, printed_train])
    else:
        train_data = mnist_train
        print("[train] WARNING: No printed digits — MNIST only")
    print(f"[train] Training samples: {len(train_data)}")

    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=use_cuda,      # pre-stage batches in pinned (page-locked) memory for fast GPU transfer
        num_workers=num_workers,   # parallel data loading processes
        persistent_workers=num_workers > 0,  # keep workers alive between epochs
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    mnist_test_loader = DataLoader(mnist_test, **loader_kwargs)
    printed_test_loader = DataLoader(printed_train, **loader_kwargs) if printed_train else None

    # --- Model + optimizer ---
    model = _SudokuNetCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    # Mixed precision: float16 forward/backward, float32 weight updates
    # Big speedup on Ampere+ GPUs (RTX 30xx/40xx) with tensor cores
    use_amp = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Training loop ---
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

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
        mnist_acc = evaluate(model, mnist_test_loader, device)

        printed_str = ""
        if printed_test_loader is not None:
            printed_acc = evaluate(model, printed_test_loader, device)
            printed_str = f" printed={printed_acc:.4f}"

        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:2d}/{EPOCHS}: loss={avg_loss:.4f} "
              f"train={train_acc:.4f} mnist={mnist_acc:.4f}{printed_str} "
              f"lr={optimizer.param_groups[0]['lr']:.1e} [{elapsed:.0f}s]")

        scheduler.step(mnist_acc)

    # --- Save ---
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\n[train] Saved to {WEIGHTS_PATH}")
    print(f"[train] Final MNIST accuracy: {mnist_acc:.4f}")
    if printed_test_loader is not None:
        print(f"[train] Final printed accuracy: {printed_acc:.4f}")
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
    print("Step 2: Train on MNIST + printed digits")
    print("=" * 60)
    train()


if __name__ == "__main__":
    main()
