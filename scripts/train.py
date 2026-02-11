"""Train SudokuNet on MNIST + synthetic printed digits."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from sudoku_ocr.model import _SudokuNetCNN

BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
DATA_DIR = Path("data")
PRINTED_PT = DATA_DIR / "printed_digits.pt"
WEIGHTS_PATH = Path("src/sudoku_ocr/weights/digit_classifier.pt")


def evaluate(model, loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            correct += (output.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total if total > 0 else 0.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    target_transform = lambda y: torch.tensor(y, dtype=torch.long)

    # MNIST dataset
    mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True,
                                 transform=transform, target_transform=target_transform)
    mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True,
                                transform=transform, target_transform=target_transform)

    # Synthetic printed digits (if available)
    printed_train = None
    if PRINTED_PT.exists():
        data = torch.load(PRINTED_PT, weights_only=True)
        printed_train = TensorDataset(data["images"], data["labels"])
        print(f"Printed digits: {len(printed_train)} images from {PRINTED_PT}")
    else:
        print(f"WARNING: {PRINTED_PT} not found — training on MNIST only")

    # Combine datasets
    if printed_train is not None:
        train_data = ConcatDataset([mnist_train, printed_train])
        print(f"Combined training set: {len(train_data)} "
              f"(MNIST: {len(mnist_train)}, printed: {len(printed_train)})")
    else:
        train_data = mnist_train
        print(f"Training set: {len(train_data)} (MNIST only)")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE)
    printed_test_loader = (
        DataLoader(printed_train, batch_size=BATCH_SIZE)
        if printed_train is not None else None
    )

    model = _SudokuNetCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (output.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        # Evaluate on MNIST test set
        mnist_acc = evaluate(model, mnist_test_loader, device)

        # Evaluate on printed digits (reuse full set as proxy since no held-out split)
        printed_acc_str = ""
        combined_acc = mnist_acc
        if printed_test_loader is not None:
            printed_acc = evaluate(model, printed_test_loader, device)
            printed_acc_str = f" printed_acc={printed_acc:.4f}"
            combined_acc = mnist_acc  # scheduler tracks MNIST accuracy

        print(f"Epoch {epoch + 1}/{EPOCHS}: loss={avg_loss:.4f} "
              f"train_acc={train_acc:.4f} mnist_acc={mnist_acc:.4f}{printed_acc_str} "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

        scheduler.step(mnist_acc)

    # Save weights
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nModel saved to {WEIGHTS_PATH}")
    print(f"Final MNIST test accuracy: {mnist_acc:.4f}")
    if printed_test_loader is not None:
        print(f"Final printed digit accuracy: {printed_acc:.4f}")


if __name__ == "__main__":
    main()
