"""Local training utilities shared by baseline and federated modes.

In HydroFed, every simulated hydropower plant trains on its own local
windows. This module contains the basic PyTorch pieces needed for that:
a Dataset wrapper and a simple training loop.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder


class PlantDataset(Dataset):
    """Wrap pre-windowed plant telemetry for PyTorch.

    Expected input shape:
        (num_windows, window_length, n_features)

    Each item returned by the dataset is one telemetry window.
    """

    def __init__(self, windows: np.ndarray) -> None:
        if windows.ndim != 3:
            raise ValueError(
                "windows must have shape "
                "(num_windows, window_length, n_features)"
            )

        if windows.shape[0] == 0:
            raise ValueError("windows must contain at least one window")

        self.windows = torch.as_tensor(windows, dtype=torch.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def train_epochs(
    model: LSTMAutoencoder,
    train_windows: np.ndarray,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Train an autoencoder on normal telemetry windows.

    Returns:
        The average reconstruction loss from the final epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = PlantDataset(train_windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    final_epoch_loss = 0.0

    for _ in range(epochs):
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            total_examples += batch.shape[0]

        final_epoch_loss = total_loss / total_examples

    return float(final_epoch_loss)


def reconstruction_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Score windows without building an autograd graph."""
    loader = DataLoader(PlantDataset(windows), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            scores.append(model.reconstruction_errors(batch.to(device)).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def calibrate_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    """Choose a threshold from normal validation reconstruction errors."""
    values = np.asarray(errors, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("errors must contain finite values")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    return float(np.percentile(values, percentile))
