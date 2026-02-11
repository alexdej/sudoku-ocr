"""CNN model for sudoku digit classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

MODEL_INPUT_SIZE = 28  # Match MNIST dimensions


def _prepare_cell_image(digit_image: np.ndarray) -> torch.Tensor:
    """Resize and normalize a digit image to match MNIST format.

    Centers the digit in a 28x28 frame with padding, matching the
    layout MNIST digits have (centered with ~4px border).

    Args:
        digit_image: Binary image of an extracted digit.

    Returns:
        Float tensor of shape (1, 1, 28, 28) ready for inference.
    """
    # Light morphological cleanup — close small gaps, remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(digit_image, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Crop to bounding box of the digit
    coords = cv2.findNonZero(cleaned)
    if coords is None:
        # Empty image — return zeros
        return torch.zeros(1, 1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = cleaned[y : y + h, x : x + w]

    # Fit into a square with padding (MNIST-style: digit occupies ~20x20
    # centered in 28x28, so ~4px border = ~14% padding per side)
    target_inner = 20
    max_dim = max(w, h)
    scale = target_inner / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28 canvas
    canvas = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.uint8)
    x_off = (MODEL_INPUT_SIZE - new_w) // 2
    y_off = (MODEL_INPUT_SIZE - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    normalized = canvas.astype(np.float32) / 255.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


class _SudokuNetCNN(nn.Module):
    """Small CNN for digit classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SudokuNet:
    """Digit classifier for sudoku cells."""

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._device = torch.device("cpu")
        self._model = _SudokuNetCNN()
        self._model.eval()
        if weights_path is not None:
            self.load(weights_path)

    def load(self, weights_path: str | Path) -> None:
        """Load trained model weights."""
        state = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, digit_image: np.ndarray) -> int:
        """Classify a single digit image.

        Args:
            digit_image: Binary image of an extracted digit.

        Returns:
            Predicted digit (0-9).
        """
        tensor = _prepare_cell_image(digit_image).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        return int(output.argmax(dim=1).item())
